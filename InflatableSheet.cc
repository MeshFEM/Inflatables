#include "InflatableSheet.hh"

#include <MeshFEM/MSHFieldWriter.hh>
#include <MeshFEM/ParallelAssembly.hh>
#include <MeshFEM/filters/remove_dangling_vertices.hh>

// #include <unsupported/Eigen/MPRealSupport>

// Construct from a triangle mesh representing the "top" sheet.
// The "bottom" sheet is an oppositely oriented copy.
InflatableSheet::InflatableSheet(const std::shared_ptr<Mesh> &inMesh, const std::vector<bool> &fusedVtx)
        : m_topSheetMesh(inMesh), m_useHessianProjectedEnergy(2 * inMesh->numTris(), false) {
    const auto &m = mesh();
    const size_t nv = m.numVertices(),
                 nt = m.numTris();

    if (fusedVtx.empty()) {
        m_fusedVtx.assign(nv, false);
        // Glue the mesh together along its boundary (no individual air channel walls were passed.)
        for (const auto v : m.vertices()) {
            m_fusedVtx[v.index()] = v.isBoundary();
        }
    }
    else { m_fusedVtx = fusedVtx; }

    if (m_fusedVtx.size() != nv) throw std::runtime_error("Incorrect fusedVtx size");

    m_wallVertices.clear();
    for (size_t vi = 0; vi < nv; ++vi) {
        if (m_fusedVtx[vi])
            m_wallVertices.push_back(vi);
    }

    m_airChannelForTri.assign(nt, 0);
    m_numTopSheetWallTris = 0;
    // Completely fused triangles do not belong to an air channel. For now,
    // all fused non-triangles belong to the same air channel (1)
    for (const auto t : m.tris()) {
        bool isWall = m_fusedVtx[t.vertex(0).index()] &&
                      m_fusedVtx[t.vertex(1).index()] &&
                      m_fusedVtx[t.vertex(2).index()];
        m_airChannelForTri[t.index()] = !isWall;
        m_numTopSheetWallTris += isWall;
    }

    {
        // Generate new "reduced vertices" for both the top and bottom sheet
        size_t newReducedVtxIdex = 0;
        m_reducedVtxForVertex.resize(nv, 2);
        for (size_t vi = 0; vi < nv; ++vi) {
            m_reducedVtxForVertex(vi, 0) = newReducedVtxIdex;
            // Only create a new vertex for the bottom sheet if the two sheets are not fused here.
            if (!m_fusedVtx[vi]) ++newReducedVtxIdex;
            m_reducedVtxForVertex(vi, 1) = newReducedVtxIdex;
            ++newReducedVtxIdex;
        }
        m_numReducedVertices = newReducedVtxIdex;
    }

    m_updateB();

    setIdentityDeformation();
    m_updateMaterialProperties();
}

void InflatableSheet::m_updateB() {
    // Generate an orthonormal basis for the tangent plane of each triangle.
    // (Both top and bottom sheets).
    const auto &m = mesh();
    const size_t nt = m.numTris();
    m_B.reserve(2 * nt);

    // First, check if we actually have a plate in the z = 0 plane; in this
    // case we use the global 2D coordinate system's axis vectors as our
    // orthonormal basis to ease specification of anisotropic materials.
    if (std::abs(m.boundingBox().dimensions()[2]) < 1e-16) {
        M32d globalB(M32d::Identity());
        m_B.assign(nt, globalB);
    }
    else {
        m_B.resize(nt);
        for (auto tri : m.elements()) {
            V3d b0 = (tri.node(1)->p - tri.node(0)->p).cast<Real>().normalized();
            V3d b1 = tri->normal().cast<Real>().cross(b0);
            const size_t ti = tri.index();
            m_B[ti] << b0, b1;
        }
    }
    // Bottom sheet uses a flipped basis (negated "b1" axis)
    V2d negate_b1(1.0, -1.0);
    for (size_t ti = 0; ti < nt; ++ti)
        m_B.push_back(m_B[ti] * negate_b1.asDiagonal());
}

void InflatableSheet::setVars(Eigen::Ref<const VXd> vars) {
    BENCHMARK_START_TIMER_SECTION("InflatableSheet setVars");
    if (size_t(vars.size()) != numVars()) throw std::runtime_error("Invalid variable size");
    m_currVars = vars;

    // Compute Jacobian and per-triangle energy under the new deformation
    const size_t nt = mesh().numTris();
    m_J .resize(2 * nt);
    m_JB.resize(2 * nt);
    m_triEnergyDensity.resize(2 * nt);
    m_BtGradLambda.resize(2 * nt);
    m_deformed_normals.resize(3, 2 * nt);
    m_deformed_areas  .resize(2 * nt);

    m_projectedTriEnergyDensity.resize(2 * nt);

    auto process_tri = [&](const size_t ti) {
        const auto &tri = mesh().element(ti);
        // const auto &gradLambda = tri->gradBarycentric();
        const auto gradLambda = tri->gradBarycentric().cast<Real>();
        for (size_t sheetIdx = 0; sheetIdx < 2; ++sheetIdx) {
            size_t sheet_tri_idx = sheetTriIdx(sheetIdx, tri.index());
            M3d triCornerPos;
            getDeformedTriCornerPositions(ti, sheetIdx, triCornerPos);
            M32d &JB = m_JB[sheet_tri_idx];
            M3d  &J  = m_J [sheet_tri_idx];

            m_BtGradLambda[sheet_tri_idx] = m_B[sheet_tri_idx].transpose() * gradLambda;
            J  = triCornerPos * gradLambda.transpose();
            JB = J * m_B[sheet_tri_idx];
            m_triEnergyDensity[sheet_tri_idx].setDeformationGradient(JB);
            if (m_useHessianProjectedEnergy[sheet_tri_idx])
                m_projectedTriEnergyDensity[sheet_tri_idx].setF(JB);

            Real normalSign = (sheetIdx == 0) ? 1.0 : -1.0;
            const V3d n = (triCornerPos.col(1) - triCornerPos.col(0)).cross(triCornerPos.col(2) - triCornerPos.col(0));
            const Real dblA = n.norm();
            m_deformed_areas[sheet_tri_idx] = 0.5 * dblA;
            m_deformed_normals.col(sheet_tri_idx) = (normalSign / dblA) * n;
        }
    };

#if MESHFEM_WITH_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, nt), [&](const tbb::blocked_range<size_t> &b) { for (size_t ti = b.begin(); ti < b.end(); ++ti) process_tri(ti); });
#else
    for (size_t ti = 0; ti < nt; ++ti) process_tri(ti);
#endif
    BENCHMARK_STOP_TIMER_SECTION("InflatableSheet setVars");
}

// Set the current deformed configuration equal to a rigid transformation
// of the top sheet mesh vertex positions "P". This rigid transformation is
// chosen to place the vertex "c" closest to the center of mass at the origin,
// place the furthest vertex "p" from "c" at (p_x, 0, 0) (defining the x axis)
// and place the furthest vertex "q" from the new x axis at "(q_x, q_y, 0)".
// This allows us to efficiently constrain the sheet's rigid motion with 6
// variable pin constraints (c = 0, p_y = p_z = q_z = 0).
void InflatableSheet::setUninflatedDeformation(M3Xd P /* copy modified inside */, bool prepareRigidMotionPinConstraints) {
    VXd vars(numVars());

    if (prepareRigidMotionPinConstraints) {
        // Pick "c" and place it at the origin.
        int c_idx;
        V3d cm = P.rowwise().mean();
        (P.colwise() - cm).colwise().squaredNorm().minCoeff(&c_idx);
        P.colwise() -= P.col(c_idx).eval();

        // Pick "p", defining the unit x axis vector "x_hat"
        int p_idx;
        P.colwise().squaredNorm().maxCoeff(&p_idx);
        V3d x_hat = P.col(p_idx).normalized();

        // Pick "q", defining the unit y axis vector "y_hat"
        int q_idx;
        P.colwise().cross(x_hat).colwise().squaredNorm().maxCoeff(&q_idx);
        // (P - x_hat * (x_hat.transpose() * P)).colwise().squaredNorm().maxCoeff(q_idx);
        V3d y_hat = (P.col(q_idx) - x_hat.dot(P.col(q_idx)) * x_hat).normalized();
        V3d z_hat = x_hat.cross(y_hat);

        M3d R;
        R << x_hat.transpose(),
             y_hat.transpose(),
             z_hat.transpose();

        P = R * P;

        m_rigidMotionPinVars[0] = varIdx(0, c_idx, 0);
        m_rigidMotionPinVars[1] = varIdx(0, c_idx, 1);
        m_rigidMotionPinVars[2] = varIdx(0, c_idx, 2);

        m_rigidMotionPinVars[3] = varIdx(0, p_idx, 1);
        m_rigidMotionPinVars[4] = varIdx(0, p_idx, 2);

        m_rigidMotionPinVars[5] = varIdx(0, q_idx, 2);
    }
    else {
        m_rigidMotionPinVars.fill(0);
    }

    for (const auto v : mesh().vertices()) {
        for (size_t sheetIdx = 0; sheetIdx < 2; ++sheetIdx)
            vars.segment<3>(varIdx(sheetIdx, v.index())) = P.col(v.index()).transpose();
    }

    setVars(vars);
}

// Neumaier sum adapted from Wikipedia
template<typename T>
struct NeumaierSum {
    NeumaierSum(T val = 0) : sum(val) { }

    void accumulate(T term) {
        T newSum = sum + term;
        if (std::abs(sum) >= std::abs(term))
            c += term  + (sum - newSum); // If sum is bigger, low-order digits of "term" are lost.
        else
            c += sum + (term - newSum);  // Else low-order digits of sum are lost
        sum = newSum;
    }

    T result() {  return sum + c; }

    T c = 0; // roundoff error correction
    T sum = 0;
};

// enclosed volume
InflatableSheet::Real InflatableSheet::volume() const {
    Real vol_6 = 0;
    M3d triCornerPos;

    // We expect the reference volume to be close to the current volume and therefore
    // center our volume calculation around it to reduce floating point cancellation.
    vol_6 = -6 * m_referenceVolume;

    NeumaierSum<Real> sum(vol_6);

    for (const auto tri : mesh().elements()) {
        if (m_airChannelForTri[tri.index()] == 0) continue; // only boundaries of the air channels feel pressure

#if 0
        M3d triCornerPosBot;
        // Attempt at a more numerically robust formula--doesn't seem to make much difference,
        // and if anything is slightly less accurate.
        getDeformedTriCornerPositions(tri.index(), 0, triCornerPos);
        getDeformedTriCornerPositions(tri.index(), 1, triCornerPosBot);

        V3d double_nA_top, double_nA_bot;
        double_nA_top = (triCornerPos   .col(1) - triCornerPos   .col(0)).cross(triCornerPos   .col(2) - triCornerPos   .col(0));
        double_nA_bot = (triCornerPosBot.col(2) - triCornerPosBot.col(0)).cross(triCornerPosBot.col(1) - triCornerPosBot.col(0));
        V3d three_c   = (triCornerPos   .rowwise().sum() + triCornerPosBot.rowwise().sum()) / 2;
        Real contrib  = (triCornerPos   .rowwise().sum() - three_c).dot(double_nA_top)
                      + (triCornerPosBot.rowwise().sum() - three_c).dot(double_nA_bot)
                      + three_c.dot(double_nA_bot + double_nA_top);
        vol_6 += contrib / 3;
#else
        getDeformedTriCornerPositions(tri.index(), 0, triCornerPos);
        Real triContrib = triCornerPos.determinant(); // Sum top/bottom sheet contrib first to reduce floating point error
        getDeformedTriCornerPositions(tri.index(), 1, triCornerPos);
        triContrib     -= triCornerPos.determinant();
        sum.accumulate(triContrib);
#endif
    }
    return sum.result() / 6.0 + m_referenceVolume;
    // return vol_6 / 6.0 + m_referenceVolume;
}

InflatableSheet::Real InflatableSheet::energyPressurePotential() const {
    return -(volume() - m_referenceVolume) * m_pressure;
}

InflatableSheet::VXd InflatableSheet::gradientPressurePotential() const {
    VXd result(VXd::Zero(numVars()));

    for (size_t sheetIdx = 0; sheetIdx < 2; ++sheetIdx) {
        for (const auto tri : mesh().elements()) {
            if (m_airChannelForTri[tri.index()] == 0) continue; // only boundaries of the air channels feel pressure
#if 1
            const size_t sheet_tri_idx = sheetTriIdx(sheetIdx, tri.index());
            V3d contrib = (-m_pressure * m_deformed_areas[sheet_tri_idx] / 3.0) * m_deformed_normals.col(sheet_tri_idx);
            for (const auto v : tri.vertices())
                result.segment<3>(varIdx(sheetIdx, v.index())) += contrib;
#else // equivalent version derived more directly from the signed volume pressure potential
            const double normalSign = (sheetIdx == 0) ? 1.0 : -1.0;
            const double signed_pressure_div_6 = normalSign * m_pressure / 6.0;
            M3d triCornerPos;
            getDeformedTriCornerPositions(tri.index(), sheetIdx, triCornerPos);
            for (const auto v : tri.vertices()) {
                result.segment<3>(varIdx(sheetIdx, v.index())) -=
                    signed_pressure_div_6 * triCornerPos.col((v.localIndex() + 1) % 3)
                                     .cross(triCornerPos.col((v.localIndex() + 2) % 3));
            }
#endif
        }
    }
    return result;
}

InflatableSheet::Real InflatableSheet::energy(EnergyType etype) const {
    BENCHMARK_START_TIMER_SECTION("InflatableSheet energy");
    NeumaierSum<Real> sum;
    if ((etype == EnergyType::Full) || (etype == EnergyType::Pressure))
        sum.accumulate(energyPressurePotential());

    if ((etype == EnergyType::Full) || (etype == EnergyType::Elastic)) {
        for (const auto tri : mesh().elements()) {
            for (size_t sheetIdx = 0; sheetIdx < 2; ++sheetIdx) {
                size_t sheet_tri_idx = sheetTriIdx(sheetIdx, tri.index());
                sum.accumulate(tri->volume() * m_triEnergyDensity[sheet_tri_idx].energy());
            }
        }
    }
    BENCHMARK_STOP_TIMER_SECTION("InflatableSheet energy");

    return sum.result();
}

InflatableSheet::VXd InflatableSheet::gradient(EnergyType etype) const {
    BENCHMARK_START_TIMER_SECTION("InflatableSheet gradient");
    VXd result(VXd::Zero(numVars()));

    if ((etype == EnergyType::Full) || (etype == EnergyType::Pressure))
        result += gradientPressurePotential();

    if ((etype == EnergyType::Full) || (etype == EnergyType::Elastic)) {
        auto accumulatePerTriContrib = [this](size_t tri_idx, VXd &out) {
            const auto &tri = mesh().tri(tri_idx);
            for (size_t sheetIdx = 0; sheetIdx < 2; ++sheetIdx) {
                const size_t sheet_tri_idx = sheetTriIdx(sheetIdx, tri.index());
                const auto &ted            = m_triEnergyDensity[sheet_tri_idx];
                const auto &BtGradLambda   = m_BtGradLambda    [sheet_tri_idx];

                M3d dE_dv = ted.denergy() * BtGradLambda;
                for (const auto v : tri.vertices())
                    out.segment<3>(varIdx(sheetIdx, v.index(), 0)) += tri->volume() * dE_dv.col(v.localIndex());
            }
        };

        // assemble_parallel(accumulatePerTriContrib, result, mesh().numElements());
        // The serial version actually seems to be faster... (not enough work is done for each tri).
        const size_t ntri = mesh().numElements();
        for (size_t i = 0; i < ntri; ++i)
            accumulatePerTriContrib(i, result);
    }

    BENCHMARK_STOP_TIMER_SECTION("InflatableSheet gradient");

    return result;
}

SuiteSparseMatrix InflatableSheet::hessianSparsityPattern(Real val) const {
    SuiteSparseMatrix Hsp(numVars(), numVars());
    Hsp.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
    Hsp.Ap.reserve(numVars() + 1);

    auto &Ap = Hsp.Ap;
    auto &Ai = Hsp.Ai;

    const auto &m = mesh();

    // Build the sparsity pattern in compressed form one column (variable) at a time.
    Hsp.Ap.push_back(0);

    auto addRVtx = [&](const size_t idx, const size_t c) { Ai.push_back(3 * idx + c); };

    auto finalizeCol = [&]() {
        const size_t colStart = Ap.back();
        const size_t colEnd = Ai.size();
        Ap.push_back(colEnd);
        std::sort(Ai.begin() + colStart, Ai.begin() + colEnd);
    };

    // The following code assumes the reduced variables are assigned in order
    // of unique vertices of the top/bottom sheet!
    // Specifically, we assume that m_reducedVtxForVertex has contiguous,
    // monotonically increasing entries when read left-right from top to
    // bottom (and repeated indices must happen on the same row).
    for (const auto v : m.vertices()) {
        const size_t vi = v.index();

        // Visit both copies of v (top sheet and non-fused bottom sheet).
        for (size_t v_sheet = 0; v_sheet < 2; ++v_sheet) {
            if ((v_sheet == 1) && m_fusedVtx[vi]) break; // Bottom sheet vtx fused with already visited top sheet vtx

            const size_t rvi = m_reducedVtxForVertex(vi, v_sheet);
            // Visit all variables (columns) corresponding to the current copy of v.
            for (size_t v_comp = 0; v_comp < 3; ++v_comp) {
                // Self-interaction (upper triangle)
                for (size_t c = 0; c <= v_comp; ++c)
                    addRVtx(rvi, c);

                // Interact with neighbors on u_sheet of current v copy.
                auto addNeighborsOnSheet = [&](size_t u_sheet) {
                    for (const auto he : v.incidentHalfEdges()) {
                        const size_t ui = he.tail().index();
                        if ((v_sheet == 0) && (u_sheet == 1) && m_fusedVtx[ui]) continue; // fused vertices on sheet 1 were visited when traversing sheet 0

                        const size_t rui = m_reducedVtxForVertex(ui, u_sheet);
                        // interact with all 3 components of neighbor vertex in upper triangle
                        if (rui < rvi) { addRVtx(rui, 0); addRVtx(rui, 1); addRVtx(rui, 2); }
                    }
                };

                // Interact with neighbors on the same sheet
                addNeighborsOnSheet(v_sheet);
                // Interact with neighbors on opposite (bottom) sheet too if v is fused.
                if (m_fusedVtx[vi]) addNeighborsOnSheet(1);

                finalizeCol();
            }
        }
    }

    Hsp.nz = Ai.size();
    Hsp.Ax.assign(Hsp.nz, val);

    return Hsp;
}

SuiteSparseMatrix InflatableSheet::hessian(EnergyType etype) const {
    SuiteSparseMatrix H = hessianSparsityPattern();
    hessian(H, etype);
    return H;
}

void InflatableSheet::hessian(SuiteSparseMatrix &H, EnergyType etype) const {
    auto assemblePerTriContrib = [&](const size_t ti, SuiteSparseMatrix &Hout) {
        const auto &tri = mesh().element(ti);
        for (size_t sheetIdx = 0; sheetIdx < 2; ++sheetIdx) {
            const size_t sheet_tri_idx = sheetTriIdx(sheetIdx, tri.index());

            if ((etype == EnergyType::Full) || (etype == EnergyType::Pressure)) {
                M3d triCornerPos;
                const double normalSign = (sheetIdx == 0) ? 1.0 : -1.0;
                const double signed_pressure_div_6 = normalSign * m_pressure / 6.0;
                getDeformedTriCornerPositions(tri.index(), sheetIdx, triCornerPos);
                for (    const auto v_b : tri.vertices()) {
                    for (const auto v_a : tri.vertices()) {
                        size_t a = varIdx(sheetIdx, v_a.index(), 0),
                               b = varIdx(sheetIdx, v_b.index(), 0);
                        if (a >= b) continue; // strict upper triangle only (no vertex self-interaction)

                        const size_t vla = v_a.localIndex();
                        const size_t vlb = v_b.localIndex();
                        // Gradient wrt v1 of a triangle's signed volume contribution is:
                        //      d vol / d v1 = v_2 x  v_3
                        // so differentiating again with respect to v_2 or v_3
                        // gives a cross product matrix -[v_3]_x or [v_2]_x, respectively.
                        // The sign here is referred to as ordering_sign below.
                        const size_t vlother = 3 - (vla + vlb);
                        const double ordering_sign = (vlb == ((vla + 1) % 3)) ? -1.0 : 1.0;
                        V3d contrib = (-signed_pressure_div_6 * ordering_sign) * triCornerPos.col(vlother);

                        Hout.addNZ(a + 1, b + 0,  contrib[2]);
                        Hout.addNZ(a + 2, b + 0, -contrib[1]);
                        Hout.addNZ(a + 0, b + 1, -contrib[2]);
                        Hout.addNZ(a + 2, b + 1,  contrib[0]);
                        Hout.addNZ(a + 0, b + 2,  contrib[1]);
                        Hout.addNZ(a + 1, b + 2, -contrib[0]);
                    }
                }
            }
            if (((etype == EnergyType::Full) || (etype == EnergyType::Elastic))) {
                // Accumulate contribution from sheet triangle (tri, sheetIdx)
                // Note: we assume that the variables for the components of a sheet vertex position
                // are contiguous.
                const auto &BtGradLambda = m_BtGradLambda[sheet_tri_idx];

                for (const auto v_b : tri.vertices()) {
                    VSFJ vol_dF_b(0, tri->volume() * BtGradLambda.col(v_b.localIndex()));
                    const size_t b = varIdx(sheetIdx, v_b.index(), 0);
                    for (size_t comp_b = 0; comp_b < 3; ++comp_b) {
                        vol_dF_b.c = comp_b;
                        M32d delta_de;
                        const bool useHPE = m_useHessianProjectedEnergy[sheet_tri_idx];
                        if (useHPE) delta_de = m_projectedTriEnergyDensity[sheet_tri_idx].delta_denergy(vol_dF_b);
                        else        delta_de =          m_triEnergyDensity[sheet_tri_idx].delta_denergy(vol_dF_b);

                        for (const auto v_a : tri.vertices()) {
                            const size_t a = varIdx(sheetIdx, v_a.index(), 0);
                            if (a > b) continue; // upper triangle only
                            if (a == b) Hout.addNZStrip(a, b + comp_b, delta_de.topRows(comp_b + 1) * BtGradLambda.col(v_a.localIndex()));
                            else        Hout.addNZStrip(a, b + comp_b, delta_de                     * BtGradLambda.col(v_a.localIndex()));
                        }
                    }
                }
            }
        }
    };

    assemble_parallel(assemblePerTriContrib, H, mesh().numTris());
}

std::shared_ptr<InflatableSheet::Mesh> InflatableSheet::visualizationMesh(bool duplicateFusedTris) const {
    const size_t nv = mesh().numVertices();
    const size_t nt = mesh().numTris();
    std::vector<V3d> vertices(2 * nv);
    std::vector<MeshIO::IOElement> elements;
    elements.reserve(2 * nt);

    for (size_t sheetIdx = 0; sheetIdx < 2; ++sheetIdx) {
        for (size_t vi = 0; vi < nv; ++vi)
            vertices[vi + sheetIdx * nv] = getDeformedVtxPosition(vi, sheetIdx);
        for (const auto tri : mesh().elements()) {
            if (!duplicateFusedTris && (sheetIdx == 1) && isWallTri(tri.index())) continue;
            elements.emplace_back(tri.vertex(0).index() + sheetIdx * nv,
                                  tri.vertex(1).index() + sheetIdx * nv,
                                  tri.vertex(2).index() + sheetIdx * nv);
            if (sheetIdx == 1) std::swap(elements.back()[0], elements.back()[1]); // flip orientation of bottom sheet tris
        }
    }
    if (!duplicateFusedTris)
        remove_dangling_vertices(vertices, elements);

    // Note: deduplicating the fused trianges can result in a non-manifold mesh
    return std::make_shared<Mesh>(elements, vertices, /* suppressNonmanifoldWarning = */ !duplicateFusedTris);
}

Eigen::MatrixXd InflatableSheet::visualizationField(Eigen::MatrixXd field, bool duplicateFusedTris) {
    const size_t nt        = mesh().numTris();
    const size_t nv        = mesh().numVertices();
          size_t in_size   = field.rows();
    const size_t field_dim = field.cols();
    // std::cout << "Running visualizationField with" << std::endl;
    // std::cout << "nt        = " << nt        << std::endl;
    // std::cout << "nv        = " << nv        << std::endl;
    // std::cout << "in_size   = " << in_size   << std::endl;
    // std::cout << "field_dim = " << field_dim << std::endl;
    // std::cout << "num reduced vertices = " << m_numReducedVertices << std::endl;

    // Duplicate data defined on just the top sheet to the bottom sheet.
    if ((in_size == nt) || (in_size == nv)) {
        field.conservativeResize(2 * in_size, field_dim);
        field.bottomRows(in_size) = field.topRows(in_size);
    }

    // Decode "DoF field" into a per-vertex field on top/bottom sheet
    // (e.g., output of `sheet.gradient()` after reshaping into an `N x 3` matrix)
    if (in_size == m_numReducedVertices) { // "DoF field" 
        Eigen::MatrixXd decodedField(2 * nv, field_dim);
        for (size_t sheetIdx = 0; sheetIdx < 2; ++sheetIdx) {
            for (size_t vi = 0; vi < nv; ++vi)
                decodedField.row(sheetIdx * nv + vi) = field.row(m_reducedVtxForVertex(vi, sheetIdx));
        }
        field.swap(decodedField);
    }

    in_size = field.rows(); // possibly updated!

    // Deduplicate fields defined on top/bottom sheet when duplicateFusedTris = false.
    if (in_size == 2 * nt) {
        if (!duplicateFusedTris) {
            // Deduplicate the field using the same ordering as `visualizationMesh`
            const size_t deduped_tris = numTubeTris() + m_numTopSheetWallTris;
            size_t outTri = nt;
            // Copy the retained portion of the bottom (sheet 1) data over and verify
            // that the data on the duplicated wall tris match the top sheet data.
            for (size_t ti = 0; ti < nt; ++ti) {
                if (isWallTri(ti)) {
                    if (field.row(nt + ti) != field.row(ti))
                        throw std::runtime_error("Inconsistent data on deduplicated top/bottom triangles");
                    continue;
                }
                field.row(outTri++) = field.row(nt + ti);
            }
            if (outTri != deduped_tris) throw std::logic_error("Output size misprediction");
            field.conservativeResize(deduped_tris, field_dim);
        }
    }
    else if (in_size == 2 * nv) {
        if (!duplicateFusedTris) {
            size_t outVtx = nv;
            std::vector<bool> include(nv); // whether to include the bottom sheet copy of a vertex
            for (const auto t: mesh().elements()) {
                if (isWallTri(t.index())) continue;
                include[t.vertex(0).index()] = true;
                include[t.vertex(1).index()] = true;
                include[t.vertex(2).index()] = true;
            }
            for (size_t vi = 0; vi < nv; ++vi) {
                if (!include[vi]) {
                    if (field.row(nv + vi) != field.row(vi))
                        throw std::runtime_error("Inconsistent data on deduplicated top/bottom vertices");
                    continue;
                }
                field.row(outVtx++) = field.row(nv + vi);
            }
            field.conservativeResize(outVtx, field_dim);
        }
    }
    else throw std::runtime_error("Unimplemented/unsupported field type");

    return field;
}

void InflatableSheet::writeDebugMesh(const std::string &path) const {
    const size_t nv = mesh().numVertices();
    const size_t nt = mesh().numTris();
    std::vector<MeshIO::IOVertex > vertices(2 * nv);
    std::vector<MeshIO::IOElement> elements;
    elements.reserve(2 * nt);

    VectorField<double, 3> N(2 * nt);
    SymmetricMatrixField<double, 2> strain(2 * nt); // *2D* rank-2 tensor field

    for (size_t sheetIdx = 0; sheetIdx < 2; ++sheetIdx) {
        for (size_t vi = 0; vi < nv; ++vi)
            vertices[vi + sheetIdx * nv].point = getDeformedVtxPosition(vi, sheetIdx).cast<double>();
        for (const auto tri : mesh().elements()) {
            elements.emplace_back(tri.vertex(0).index() + sheetIdx * nv,
                                  tri.vertex(1).index() + sheetIdx * nv,
                                  tri.vertex(2).index() + sheetIdx * nv);
            if (sheetIdx == 1) std::swap(elements.back()[0], elements.back()[1]); // flip orientation of bottom sheet tris
            size_t sheet_tri_idx = sheetTriIdx(sheetIdx, tri.index());
            N(sheet_tri_idx) = m_deformed_normals.col(sheet_tri_idx).cast<double>();
            strain(sheet_tri_idx) = SymmetricMatrixValue<double, 2>(greenLagrangianStrain(sheetIdx, tri.index()).cast<double>());
        }
    }

    MSHFieldWriter writer(path, vertices, elements);
    writer.addField("Normal",          N, DomainType::PER_ELEMENT);
    writer.addField("G-L Strain", strain, DomainType::PER_ELEMENT);
}

std::pair<std::vector<InflatableSheet::IdxPolyline>, std::vector<InflatableSheet::IdxPolyline>>
InflatableSheet::getFusingPolylines() const {
    std::pair<std::vector<InflatableSheet::IdxPolyline>, std::vector<InflatableSheet::IdxPolyline>> result;
    auto &boundaryLoops = result.first;
    auto &wallCurves = result.second;

    const auto &m = mesh();
    boundaryLoops = m.boundaryLoops();
    // Close the boundary loops
    for (auto &loop : boundaryLoops)
        loop.push_back(loop[0]);

    // Note: the fusing curves can be non-manifold, having vertices with more
    // than two incident wall half-edges; this happens when two extremely close
    // contours end up merging due to how InflatableSheet interprets triangles
    // with three fused corners as wall triangles (shown as + below):
    //                 o
    //    ... o--o--+  |
    //              |  +
    //    ... o--o--+  |
    //                 o
    // Therefore we must track visited half-edges rather than visited
    // vertices when traversing the curves.
    //
    // By convention, we always tranverse the half-edges *outside* the wall triangles
    // (i.e., keeping the wall region to the right) in clockwise order.
    // Therefore, the next half-edge bordering the current wall component
    // can be found by circulating counter-clockwise (through the wall
    // region) until hitting the next fused edge.
    // We currently do not support meshes with "1D wall curves" (fused
    // edges without a wall region on either wide), though these could be
    // handled as a special case.

    std::vector<bool> visited(m.numHalfEdges(), false);
    auto ccwFusedEdge = [&](auto he) {
        assert(isWallBoundaryHE(he.index(), /* include1D = */ false));
        auto curr = he;
        while ((curr = curr.ccw()) != he) {
            if (isWallBoundaryHE(curr.index(), /* include1D = */ false))
                return curr.opposite();
        }
        return decltype(he)(-1, he.mesh());
    };

    auto addCurveOriginatingWithHE = [&](auto he_start) {
        wallCurves.emplace_back();
        auto &curve = wallCurves.back();
        auto he_curr = he_start;
        curve.emplace_back(he_curr.tail().index());
        do {
            visited[he_curr.index()] = true;
            visited[he_curr.opposite().index()] = true;
            curve.emplace_back(he_curr.tip().index());
            if (he_curr.tip().isBoundary()) return he_curr; // Terminate when we hit a boundary

            auto next = ccwFusedEdge(he_curr);
            assert(next);
            he_curr = next;
        } while (!visited[he_curr.index()]); // Or when we return to the start

        return he_curr;
    };

    // Traverse boundary-incident fusing curves, starting with an outbound
    // interior half-edge with a wall triangle to its right (i.e., whose
    // opposite tri is wall).
    for (const auto bv : m.boundaryVertices()) {
        for (const auto he_opp : bv.volumeVertex().incidentHalfEdges()) {
            if (he_opp.isBoundary() || !isWallTri(he_opp.tri().index())) continue;
            auto he = he_opp.opposite();
            if (isWallTri(he.tri().index())) continue;
            auto he_end = addCurveOriginatingWithHE(he);
            assert(he_end.tip().isBoundary()); // We should have terminated at another boundary vertex...
        }
    }

    // Traverse closed fusing curves from an arbitrary unvisited half edge opposite
    // a wall triangle.
    for (const auto he : m.halfEdges()) {
        if (he.isBoundary() || visited[he.index()] || !isWallTri(he.opposite().tri().index())) continue;
        if (isWallTri(he.tri().index())) continue; // skip halfedges inside walls
        auto he_end = addCurveOriginatingWithHE(he);
        assert(he_end == he); // We should have returned to the starting half edge
    }

    // Verify that all fused half-edges are visited (they won't be if we have 1d fusing curves)
    for (const auto he : m.halfEdges()) {
        if (!visited[he.index()] && !he.isBoundary() && isWallBoundaryHE(he.index(), /* include1D = */ false))
            throw std::runtime_error("Not all fused half-edges were visited; does the sheet have 1D fusing curves (unimplemented)?");
    }

    return result;
}

std::list<std::vector<size_t>> InflatableSheet::fusedRegionBooleanIntersectSheetBoundary() const {
    std::list<std::vector<size_t>> result;
    const auto &m = mesh();

    // Predicate testing whether a given boundary half-edge is contained in the
    // boolean intersection we care about.
    auto inBooleanIntersection = [this](auto be_) { return isWallTri(be_.volumeHalfEdge().tri().index()); };

    std::vector<bool> visited(m.numBoundaryEdges(), false);
    for (const auto be : m.boundaryEdges()) {
        if (visited[be.index()]) continue;

        // Traverse a complete boundary loop starting from a halfedge *not* in
        // the boolean intersection (so the first generated polyline is not fragmented)
        auto curr_be = be;
        while ((curr_be = curr_be.next()) != be)
            if (!inBooleanIntersection(curr_be)) break;

        if (inBooleanIntersection(curr_be)) throw std::runtime_error("Couldn't find a boundary edge outside the boolean intersection!");

        std::vector<size_t> *polyline = nullptr;
        while ((curr_be = curr_be.next()) && !visited[curr_be.index()]) {
            visited[curr_be.index()] = true;
            if (inBooleanIntersection(curr_be)) {
                if (!polyline) {
                    result.emplace_back();
                    polyline = &result.back();
                    polyline->push_back(curr_be.tail().volumeVertex().index());
                }
                polyline->push_back(curr_be.tip().volumeVertex().index());
            }
            else {
                // Terminate the current polyline (if it exists)
                polyline = nullptr;
            }
        }
    }

    return result;
}

#include "SheetOptimizer.hh"

#include <MeshFEM/ParallelAssembly.hh>
#include <MeshFEM/Laplacian.hh>

SheetOptimizer::SheetOptimizer(std::shared_ptr<InflatableSheet> s, const Mesh &targetSurface)
        : m_sheet(s), m_targetSurfaceFitter(targetSurface),
          m_collapseBarrier(s->mesh(), 0.1) // Collapse prevention kicks in when element compressed to 1/10 its original area
{
    const auto &m = mesh();
    const size_t nv = m.numVertices();
    // Extract the initial rest vertex positions from the sheet
    MX2d X(nv, 2);
    for (const auto &v : m.vertices())
        X.row(v.index()) = truncateFrom3D<V2d>(v.node()->p).transpose();

    m_currVars.resize(numVars());
    size_t ndv = 2 * nv;
    m_currVars.head(numEquilibriumVars()) = sheet().getVars();
    m_currVars.tail(ndv) = Eigen::Map<VXd>(X.data(), ndv);

    // Area in the original (unoptimized) rest mesh. This is used for scaling the
    // vertex contributions to the fitting energy.
    VXd vertexArea(VXd::Zero(nv));
    for (const auto &e : m.elements())
        for (const auto &v : e.vertices())
            vertexArea[v.index()] += e->volume() / 3.0;

    const auto &wallVtxs = sheet().wallVertices();
    m_wallVtxOnBoundary.resize(wallVtxs.size());
    VXd queryPtArea(wallVtxs.size());
    for (size_t i = 0; i < wallVtxs.size(); ++i) {
        m_wallVtxOnBoundary[i] = m.vertex(wallVtxs[i]).isBoundary();
        queryPtArea[i] = vertexArea[wallVtxs[i]];
    }
    m_targetSurfaceFitter.setQueryPtWeights(queryPtArea);

    m_targetSurfaceFitter.updateClosestPoints(sheet().deformedWallVertexPositions(), m_wallVtxOnBoundary);
    m_restLaplacian = SuiteSparseMatrix(Laplacian::construct<1>(m));

    // Initialize the remaining state
    setVars(m_currVars);
}

SheetOptimizer::Real SheetOptimizer::energy(EnergyType etype) const {
    Real result = 0.0;
    if ((etype == EnergyType::Full) || (etype == EnergyType::Simulation))
        result += weight(EnergyType::Simulation) * m_sheet->energy();

    if ((etype == EnergyType::Full) || (etype == EnergyType::Fitting))
        result += weight(EnergyType::Fitting) * m_targetSurfaceFitter.energy();

    if ((etype == EnergyType::Full) || (etype == EnergyType::Smoothing)) {
        const auto &restPos = restPositionsFromVariables(m_currVars);
        for (size_t comp = 0; comp < 2; ++comp) {
            auto X = restPos.col(comp).cast<double>().eval();
            result += (0.5 * weight(EnergyType::Smoothing)) * X.dot(m_restLaplacian.apply(X));
        }
    }

    if ((etype == EnergyType::Full) || (etype == EnergyType::CollapseBarrier))
        result += m_collapseBarrier.energy();

    return result;
}

SheetOptimizer::VXd SheetOptimizer::gradient(EnergyType etype) const {
    VXd g = VXd::Zero(numVars());
    auto gradRestPositions  = restPositionsFromVariables(g);
    auto gradDeformedConfig = g.head(numEquilibriumVars());

    const auto &s = sheet();
    const auto &teds = s.triEnergyDensities();
    const auto &BtGradLambdas = s.shapeFunctionGradients();
    const auto &m = mesh();

    const Real simWeight = weight(EnergyType::Simulation),
               fitWeight = weight(EnergyType::Fitting),
            smoothWeight = weight(EnergyType::Smoothing);

    if ((etype == EnergyType::Full) || (etype == EnergyType::Simulation)) {
        for (const auto &tri : m.elements()) {
            const auto &gradLambda = tri->gradBarycentric();
            for (size_t sheetIdx = 0; sheetIdx < 2; ++sheetIdx) {
                const size_t sheet_tri_idx = s.sheetTriIdx(sheetIdx, tri.index());
                const auto &ted          = teds         [sheet_tri_idx];
                const auto &J            = s.deformationGradient3D(sheet_tri_idx);
                const auto &BtGradLambda = BtGradLambdas[sheet_tri_idx];
                for (const auto &v : tri.vertices()) {
                    for (size_t c = 0; c < 2; ++c) {
                        // We consider perturbing rest vertex position v in direction e_c in R^2.
                        // This induces the velocity lambda_v e_c, with Jacobian e_c \otimes B^T grad lambda_v
                        Real contrib = -ted.denergy(J.col(c) * BtGradLambda.col(v.localIndex()).transpose());
                        // Velocity divergence term: e_c . grad lambda_v
                        contrib += ted.energy() * gradLambda(c, v.localIndex());
                        gradRestPositions(v.index(), c) += simWeight * contrib * tri->volume();
                    }
                }
            }
        }

        gradDeformedConfig += simWeight * m_sheet->gradient();
    }

    if ((etype == EnergyType::Full) || (etype == EnergyType::Fitting)) {
        auto gFit = (fitWeight * m_targetSurfaceFitter.gradient()).eval();
        const auto &wv = s.wallVertices();
        for (size_t i = 0; i < wv.size(); ++i)
            gradDeformedConfig.segment<3>(s.varIdx(0, wv[i], 0)) += gFit.row(i).transpose().cast<Real>();
    }

    if ((etype == EnergyType::Full) || (etype == EnergyType::Smoothing)) {
        const auto &restPos = restPositionsFromVariables(m_currVars);
        for (size_t comp = 0; comp < 2; ++comp) {
            auto X = restPos.col(comp).cast<double>().eval();
            gradRestPositions.col(comp) += (smoothWeight * m_restLaplacian.apply(X)).cast<Real>();
        }
    }

    if ((etype == EnergyType::Full) || (etype == EnergyType::CollapseBarrier))
        m_collapseBarrier.accumulateGradient(gradRestPositions);

    return g;
}

SuiteSparseMatrix SheetOptimizer::hessianSparsityPattern(Real val) const {
    SuiteSparseMatrix Hsp(numVars(), numVars());
    Hsp.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
    Hsp.Ap.reserve(numVars() + 1);

    auto &Ap = Hsp.Ap;
    auto &Ai = Hsp.Ai;

    const auto &s = sheet();
    const auto &m = mesh();

    auto Hsp_sim = s.hessianSparsityPattern();
    Ap = Hsp_sim.Ap;
    Ai = Hsp_sim.Ai;

    const size_t dvo = designVarOffset();
    const size_t nv = m.numVertices();

    auto designVariable = [dvo, nv](size_t vtx, size_t c) { return dvo + c * nv + vtx; };

    // Add columns corresponding to the design variables
    // (equilibrium-design and design-design blocks).
    auto addDeformationVar = [&](size_t defVar)                    { Ai.push_back(defVar); };
    auto addRestPosVar     = [&](const size_t vtx, const size_t c) { Ai.push_back(designVariable(vtx, c)); };

    auto finalizeCol = [&]() {
        const size_t colStart = Ap.back();
        const size_t colEnd = Ai.size();
        Ap.push_back(colEnd);
        std::sort(Ai.begin() + colStart, Ai.begin() + colEnd);
    };

    // Add a column for each component of each vertex's 2D rest position
    for (size_t v_comp = 0; v_comp < 2; ++v_comp) {
        for (const auto &v : m.vertices()) {
            const size_t vi = v.index();
            auto interactWithDeformedVertex = [&](size_t ui) {
                for (size_t u_sheet = 0; u_sheet < 2; ++u_sheet) {
                    if ((u_sheet == 1) && s.isFusedVtx(ui)) break; // Bottom sheet vtx fused with already visited top sheet vtx
                    for (size_t u_comp = 0; u_comp < 3; ++u_comp)
                        addDeformationVar(s.varIdx(u_sheet, ui, u_comp));
                }
            };

            // Interact with the deformation variables for vertices in stencil.
            interactWithDeformedVertex(vi);
            for (const auto &he : v.incidentHalfEdges())
                interactWithDeformedVertex(he.tail().index());

            // Interact with the rest positions of this vertex (upper tri).
            for (size_t u_comp = 0; u_comp <= v_comp; ++u_comp)
                addRestPosVar(vi, u_comp);

            // Interact with the rest positions of neighbors (upper tri)
            for (const auto &he : v.incidentHalfEdges()) {
                const size_t ui = he.tail().index();
                for (size_t u_comp = 0; u_comp < 2; ++u_comp) {
                    if (designVariable(ui, u_comp) > designVariable(vi, v_comp)) continue;
                    addRestPosVar(ui, u_comp);
                }
            }

            finalizeCol();
        }
    }

    Hsp.nz = Ai.size();
    Hsp.Ax.assign(Hsp.nz, val);

    return Hsp;
}

SuiteSparseMatrix SheetOptimizer::hessian(EnergyType etype) const {
    SuiteSparseMatrix H = hessianSparsityPattern();
    hessian(H, etype);
    return H;
}

void SheetOptimizer::hessian(SuiteSparseMatrix &H, EnergyType etype) const {
    const auto &s = sheet();
    const auto &m = mesh();

    const size_t nv = m.numVertices();
    const size_t dvo = designVarOffset();
    auto designVariable = [dvo, nv](size_t vtx, size_t c) { return dvo + c * nv + vtx; };

    const auto &teds = s.triEnergyDensities();
    const auto &BtGradLambdas = sheet().shapeFunctionGradients();

    const Real simWeight = weight(EnergyType::Simulation),
               fitWeight = weight(EnergyType::Fitting),
            smoothWeight = weight(EnergyType::Smoothing);

    // WARNING: Simulation energy must remain first so that our weights scale
    // the Hessian properly.
    if ((etype == EnergyType::Full) || (etype == EnergyType::Simulation)) {
        s.hessian(H);

        for (const auto &tri : m.elements()) {
            for (size_t sheetIdx = 0; sheetIdx < 2; ++sheetIdx) {
                const size_t sheet_tri_idx = s.sheetTriIdx(sheetIdx, tri.index());
                const auto &ted            = teds         [sheet_tri_idx];
                const auto &J              = s.deformationGradient3D(sheet_tri_idx);
                const auto &BtGradLambda   = BtGradLambdas[sheet_tri_idx];
                const auto &gradLambda     = tri->gradBarycentric();

                size_t hint = std::numeric_limits<int>::max(); // not size_t max which would be come -1 on cast to int!
                for (size_t comp_b = 0; comp_b < 2; ++comp_b) {
                    for (const auto &v_b : tri.vertices()) {
                        // We consider perturbing rest vertex position v in direction e_c in R^2.
                        // This induces the velocity lambda_v e_c, with Jacobian e_c \otimes B^T grad lambda_v
                        M32d J_grad_v_b = J.col(comp_b) * BtGradLambda.col(v_b.localIndex()).transpose();
                        const Real div_v_b  = gradLambda(comp_b, v_b.localIndex());

                        const size_t b = designVariable(v_b.index(), comp_b);

                        for (size_t comp_a = 0; comp_a < 3; ++comp_a) {
                            for (const auto &v_a : tri.vertices()) {
                                // equilibrium-design block
                                {
                                    const size_t a = s.varIdx(sheetIdx, v_a.index(), comp_a);

                                    VSFJ dF_a(comp_a, BtGradLambda.col(v_a.localIndex()));
                                    VSFJ div_v_b_dF_a_minus_dF_a_grad_v_b(comp_a, div_v_b                               * BtGradLambda.col(v_a.localIndex())
                                                                                 - gradLambda(comp_b, v_a.localIndex()) * BtGradLambda.col(v_b.localIndex()));

                                    using DeltaF = typename VSFJ::MatrixType;
                                    Real contrib = -ted.d2energy(J_grad_v_b,                 dF_a.operator DeltaF())
                                                  + ted. denergy(div_v_b_dF_a_minus_dF_a_grad_v_b.operator DeltaF());
                                    hint = H.addNZ(a, b, contrib * tri->volume(), hint);
                                }

                                // design-design block
                                {
                                    const size_t a = designVariable(v_a.index(), comp_a);
                                    if (a > b) continue;

                                    M32d J_grad_v_a = J.col(comp_a) * BtGradLambda.col(v_a.localIndex()).transpose();

                                    const Real div_v_a              = gradLambda(comp_a, v_a.localIndex());
                                    const Real tr_grad_v_a_grad_v_b = gradLambda(comp_a, v_b.localIndex())
                                                                    * gradLambda(comp_b, v_a.localIndex());

                                    Real contrib = ted.d2energy(J_grad_v_a, J_grad_v_b)
                                                 + ted. denergy(J.col(comp_a) * (gradLambda(comp_b, v_a.localIndex()) * BtGradLambda.col(v_b.localIndex()).transpose())
                                                              + J.col(comp_b) * (gradLambda(comp_a, v_b.localIndex()) * BtGradLambda.col(v_a.localIndex()).transpose())
                                                              - J_grad_v_a * div_v_b
                                                              - J_grad_v_b * div_v_a)
                                                 + ted.energy() * (div_v_a * div_v_b - tr_grad_v_a_grad_v_b);

                                    hint = H.addNZ(a, b, contrib * tri->volume(), hint);
                                }
                            }
                        }
                    }
                }
            }
        }
        using Real2 = typename std::remove_reference_t<decltype(H)>::value_type;
        Eigen::Map<Eigen::Matrix<Real2, Eigen::Dynamic, 1>>(H.Ax.data(), H.Ax.size()) *= simWeight;
    }

    if ((etype == EnergyType::Full) || (etype == EnergyType::Fitting)) {
        const auto &wv = s.wallVertices();
        for (size_t i = 0; i < wv.size(); ++i) {
            const auto &vtxHess = (fitWeight * m_targetSurfaceFitter.vtx_hessian(i)).eval();
            const size_t varOffset = s.varIdx(0, wv[i], 0);

            size_t hint = std::numeric_limits<int>::max(); // not size_t max which would be come -1 on cast to int!
            for (size_t comp_a = 0; comp_a < 3; ++comp_a)
                for (size_t comp_b = comp_a; comp_b < 3; ++comp_b)
                    hint = H.addNZ(varOffset + comp_a, varOffset + comp_b, vtxHess(comp_a, comp_b), hint);
        }
    }

    if ((etype == EnergyType::Full) || (etype == EnergyType::Smoothing)) {
        size_t hint = std::numeric_limits<int>::max(); // not size_t max which would be come -1 on cast to int!
        for (const auto &triplet : m_restLaplacian) {
            for (size_t c = 0; c < 2; ++c)
                hint = H.addNZ(designVariable(triplet.i, c), designVariable(triplet.j, c), smoothWeight * triplet.v, hint);
        }
    }

    if ((etype == EnergyType::Full) || (etype == EnergyType::CollapseBarrier))
        m_collapseBarrier.accumulateHessian(H, designVariable);
}

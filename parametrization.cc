#include "parametrization.hh"

#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/Laplacian.hh>
#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/MeshIO.hh>
#include <complex>
#include <set>

#include <MeshFEM/ParallelAssembly.hh>
#include "circular_mean.hh"
#include "subdivide_triangle.hh"
#include "curvature.hh"

namespace parametrization {

struct SPSDSystemSolver : public SPSDSystem<Real> {
    using Base = SPSDSystem<Real>;
    using Base::Base;
};

// Compute a least-squares conformal parametrization with the global scale factor
// chosen to minimize the L2 norm of the pointwise area distortion.
UVMap lscm(const Mesh &mesh) {
    const size_t nv = mesh.numVertices();
    UVMap uv(nv, 2);

    TripletMatrix<> K(2 * nv, 2 * nv);
    K.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;

    // Assemble (upper triangle of) LSCM matrix K =
    // [L   A] = [L   A]
    // [A^T L]   [-A  L]
    // where L_ij = int grad phi_i . grad phi_j dA          is the P1 FEM Laplacian and
    //       A_ij = int n . (grad phi_j x grad phi_i) dA    is the skew symmetric "parametric area calculator"
    //            = int s_ij (1 / 2A) dA = sum_T s_ij|_T / 2
    //       s_ij|_T = 1 if local(i) == local(j) + 1, -1 if local(i) == local(j) - 1, 0 otherwise   (this is as evaluated on a particular triangle T)
    // This is the quadratic form for [u v] giving the LSCM energy.
    // Note: the interior edge contributions to the "area calculator" matrix cancel out, and it can be written as an integral over the boundary.
    // However, if we want to support varying triangle weights as recommended in Spectral Conformal Parametrization,
    // we need to compute the per-triangle contribution. (This seems to actually be a bad idea though--probably they just need to incorporate a mass matrix in their generalized eigenvalue problem.)
    for (auto tri : mesh.elements()) {
        const auto &gradLambda = tri->gradBarycentric();
        for (auto ni : tri.nodes()) {
            for (auto nj : tri.nodes()) {
                if (ni.index() > nj.index()) continue; // lower triangle
                // Symmetric Laplacian blocks
                const Real val = gradLambda.col(ni.localIndex()).dot(gradLambda.col(nj.localIndex())) * tri->volume();
                K.addNZ(     ni.index(),      nj.index(), val); // (u, u) block
                K.addNZ(nv + ni.index(), nv + nj.index(), val); // (v, v) block

                // Skew symmetric A block (u, v)
                if (ni.localIndex() == nj.localIndex()) continue;
                int s = (ni.localIndex() == (nj.localIndex() + 1) % 3) ? 1.0 : -1.0;
                K.addNZ(ni.index(), nv + nj.index(),  0.5 * s);
                K.addNZ(nj.index(), nv + ni.index(), -0.5 * s);
            }
        }
    }

    SPSDSystemSolver Ksys(K);

    // Pin down the null-space (scale, rotation) by fixing two vertices' UVs: vertex 0 and the vertex furthest from it.
    {
        Point3D p0 = mesh.node(0)->p;
        Real furthestDist = 0;
        size_t furthestIdx = 0;

        for (auto n : mesh.nodes()) {
            Real dist = (n->p - p0).norm();
            if (dist > furthestDist) {
                furthestDist = dist;
                furthestIdx = n.index();
            }
        }

        std::vector<size_t>    fixedVars = {0, furthestIdx, nv, nv + furthestIdx};
        std::vector<Real> fixedVarValues = {0.0, furthestDist, 0.0, 0.0};

        Ksys.fixVariables(fixedVars, fixedVarValues);
    }

    Eigen::VectorXd soln;
    Ksys.solve(Eigen::VectorXd::Zero(2 * nv), soln);
    Eigen::Map<Eigen::VectorXd>(uv.data(), 2 * nv) = soln;

    // Compute per-triangle areas before and after parametrization
    Eigen::VectorXd origArea(mesh.numTris()), paramArea(mesh.numTris());
    for (const auto t : mesh.elements()) {
        origArea[t.index()] = t->volume();
        std::array<Point2D, 3> poly;
        for (auto v : t.vertices())
            poly[v.localIndex()] = uv.row(v.index()).transpose();
        paramArea[t.index()] = area(poly);
    }

    // Scale the full parametrization to minimize the squared difference in areas
    // min_s 1/2 ||s paramArea - origArea||^2 ==> (s paramArea - origArea) . paramArea = 0 ==> s = (origArea . paramArea) / ||paramArea||^2
    uv *= std::sqrt(origArea.dot(paramArea) / paramArea.squaredNorm());

    return uv;
}

NDMap harmonic(const Mesh &mesh, NDMap &boundaryData) {
    const size_t nbn = mesh.numBoundaryNodes(),
                 nn  = mesh.numNodes();
    if (size_t(boundaryData.rows()) != nbn) throw std::runtime_error("Invalid boundary data size");
    size_t numComponents = boundaryData.cols();

    NDMap result(nn, numComponents);

    auto L = Laplacian::construct(mesh);
    L.sumRepeated();
    L.needs_sum_repeated = false;
    SPSDSystemSolver Lsys(L);

    // Avoid resetting the SPSDSystemSolver and fixing variables anew for each component solve
    // by always fixing the boundary variables to "0" and directly computing the "load"
    // contributed by these constraints
    std::vector<size_t> fixedVars(nbn);
    for (auto bn : mesh.boundaryNodes())
        fixedVars[bn.index()] = bn.volumeNode().index();
    Lsys.fixVariables(fixedVars, std::vector<double>(nbn, 0.0));
    std::vector<double> negDirichletValues(nn, 0.0);
    std::vector<double> soln;

    for (size_t c = 0; c < numComponents; ++c) {
        for (auto bn : mesh.boundaryNodes())
            negDirichletValues[bn.volumeNode().index()] = -boundaryData(bn.index(), c);
        auto rhs = L.apply(negDirichletValues);
        Lsys.solve(rhs, soln);

        for (auto n : mesh.nodes()) {
            auto bn = n.boundaryNode();
            result(n.index(), c) = bn ? boundaryData(bn.index(), c) : soln[n.index()];
        }
    }

    return result;
}

void Parametrizer::setUV(Eigen::Ref<const UVMap> uv) {
    const auto &m = mesh();
    if (size_t(uv.rows()) != m.numVertices()) throw std::runtime_error("Invalid parametrization size");
    m_uv = uv;

    // Update the cached Jacobians and count flips
    const size_t nt = m.numTris();
    m_J.resize(nt);
    M23d f_restrict_T;
    m_flipCount = 0;
    for (const auto &tri : m.elements()) {
        f_restrict_T.col(0) = m_uv.row(tri.vertex(0).index());
        f_restrict_T.col(1) = m_uv.row(tri.vertex(1).index());
        f_restrict_T.col(2) = m_uv.row(tri.vertex(2).index());
        auto &J = m_J[tri.index()];
        J = f_restrict_T * tri->gradBarycentric().transpose();
        if ((J * m_B[tri.index()]).determinant() < 0) ++m_flipCount;
    }

    parametrizationUpdated(); // Notify derived class that the parametrization has been updated (invalidate cache)
}

Eigen::VectorXd Parametrizer::perVertexLeftStretchAngles(double /* agreementThreshold */) const {
    const auto &m = mesh();
    Eigen::VectorXd result(m.numVertices());

    std::vector<double> twiceIncidentAngles;
    for (const auto &v : m.vertices()) {
        twiceIncidentAngles.clear();
        for (const auto &he : v.incidentHalfEdges())
            if (he.tri()) twiceIncidentAngles.push_back(2 * leftStretchAngle(he.tri().index()));
        result[v.index()] = 0.5 * circularMean(twiceIncidentAngles);
    }

    return result;
}

Eigen::VectorXd Parametrizer::perVertexAlphas() const {
    const auto &m = mesh();
    Eigen::VectorXd result(m.numVertices());
    const Eigen::VectorXd &alphas = getAlphas();

    for (auto v : mesh().vertices()) {
        double &alpha = result[v.index()];
        alpha = 0;
        size_t tri_valence = 0;
        for (auto he : v.incidentHalfEdges()) {
            if (!he.tri()) continue;
            alpha += alphas[he.tri().index()];
            ++tri_valence;
        }
        alpha /= tri_valence;
    }

    return result;
}

std::tuple<std::shared_ptr<Mesh>, UVMap>
Parametrizer::upsampledUV(size_t nsubdiv) const {
    std::tuple<std::shared_ptr<Mesh>, UVMap> result;

    std::vector<MeshIO::IOVertex > subVertices;
    std::vector<MeshIO::IOElement> subElements;
    aligned_std_vector<V2d> subUV;

    const auto &m = mesh();
    PointGluingMap indexForPoint;
    for (const auto &tri : m.elements()) {
        auto newPt = [&](const Point3D &p, double lambda_0, double lambda_1, double lambda_2) {
            subVertices.emplace_back(p);
            subUV.push_back(lambda_0 * m_uv.row(tri.vertex(0).index()) +
                            lambda_1 * m_uv.row(tri.vertex(1).index()) +
                            lambda_2 * m_uv.row(tri.vertex(2).index()));
            return subVertices.size() - 1;
        };

        subdivide_triangle(nsubdiv,
                tri.vertex(0).node()->p,
                tri.vertex(1).node()->p,
                tri.vertex(2).node()->p,
                indexForPoint,
                newPt, [&](size_t i0, size_t i1, size_t i2) { subElements.emplace_back(i0, i1, i2); });
    }

    std::get<0>(result) = std::make_shared<Mesh>(subElements, subVertices);
    auto &fineUV = std::get<1>(result);
    fineUV.resize(subUV.size(), 2);

    for (size_t i = 0; i < subUV.size(); ++i)
        fineUV.row(i) = subUV[i];

    return result;
}

std::tuple<std::shared_ptr<Mesh>, Eigen::VectorXd, Eigen::VectorXd>
Parametrizer::upsampledVertexLeftStretchAnglesAndMagnitudes(size_t nsubdiv, double agreementThreshold) const {
    std::tuple<std::shared_ptr<Mesh>, Eigen::VectorXd, Eigen::VectorXd> result;

    std::vector<MeshIO::IOVertex > subVertices;
    std::vector<MeshIO::IOElement> subElements;

    auto coarseVertexAngles = perVertexLeftStretchAngles(agreementThreshold);
    auto coarseVertexAlphas = perVertexAlphas();

    std::vector<double> subAngles, subAlphas;

    // Until we have implemented a weighted angle averaging algorithm,
    // implement the rational barycentric coordinate weights by duplicating the
    // corresponding angles (inefficient).
    const size_t barycentricDenominator = nsubdiv + 1;
    std::vector<double> cornerAngleVec(barycentricDenominator);

    const auto &m = mesh();
    PointGluingMap indexForPoint;
    // size_t triIdx = 0;
    for (const auto &tri : m.elements()) {
        // bool verbose = (triIdx++ == 6467);

        auto newPt = [&](const Point3D &p, double lambda_0, double lambda_1, double lambda_2) {
            cornerAngleVec.clear();
            auto replicateAngle = [&](size_t corner, double lambda) {
                const size_t numerator = std::round(lambda * barycentricDenominator);
                double angle = 2.0 * coarseVertexAngles[tri.vertex(corner).index()]; // average 2x the angle to account for 2-RoSy
                for (size_t i = 0; i < numerator; ++i) cornerAngleVec.push_back(angle);
            };

            replicateAngle(0, lambda_0);
            replicateAngle(1, lambda_1);
            replicateAngle(2, lambda_2);

            assert(cornerAngleVec.size() == barycentricDenominator);

            subVertices.emplace_back(p);
            subAngles.push_back(0.5 * circularMean(cornerAngleVec));
            subAlphas.push_back(lambda_0 * coarseVertexAlphas[tri.vertex(0).index()] +
                                lambda_1 * coarseVertexAlphas[tri.vertex(1).index()] +
                                lambda_2 * coarseVertexAlphas[tri.vertex(2).index()]);

            // if (verbose) {
            //     std::cout << "pt " << p.transpose() << " mean " << subAngles.back() << " from";
            //     for (double v : cornerAngleVec)
            //         std::cout << "\t" << v;
            //     std::cout << std::endl;
            // }

            return subVertices.size() - 1;
        };

        subdivide_triangle(nsubdiv,
                padTo3D(m_uv.row(tri.vertex(0).index()).transpose().eval()),
                padTo3D(m_uv.row(tri.vertex(1).index()).transpose().eval()),
                padTo3D(m_uv.row(tri.vertex(2).index()).transpose().eval()),
                indexForPoint,
                newPt, [&](size_t i0, size_t i1, size_t i2) { subElements.emplace_back(i0, i1, i2); });
    }

    std::get<0>(result) = std::make_shared<Mesh>(subElements, subVertices);
    std::get<1>(result) = Eigen::Map<Eigen::VectorXd>(subAngles.data(), subAngles.size());
    std::get<2>(result) = Eigen::Map<Eigen::VectorXd>(subAlphas.data(), subAlphas.size());
    return result;
}

LocalGlobalParametrizer::LocalGlobalParametrizer(const std::shared_ptr<Mesh> &inMesh, const UVMap &uvInit)
    : Parametrizer(inMesh)
{
    setUV(uvInit);
    // Constant Laplacian matrix used throughout local/global iterations
    L = std::make_unique<SPSDSystemSolver>(Laplacian::construct(mesh()));
    L->fixVariables(std::vector<size_t>{0}, std::vector<Real>{0.0}); // fix first vertex's coordinate in u or v axis.
}

// Replace the parametrization, updating the local-global energy energy (running the local step)
void LocalGlobalParametrizer::m_localStep() {
    BENCHMARK_START_TIMER_SECTION("Local step");
    const auto &m = mesh();
    const size_t ne = m.numElements();
    m_J.resize(ne);
    m_M_Bt.resize(ne);
    m_R.resize(ne);
    m_U.resize(ne);
    m_alpha.resize(ne);
    m_lambda.resize(ne, 2);

    // Local step: compute closest admissible Jacobian U R(theta) [alpha 0; 0 1] R(theta)^T
    // and construct the RHS for the global step.
    auto process_tri = [&](const size_t ti) {
        M2d JB = m_J[ti] * m_B[ti];

        // Decompose JB = U R Lambda R^T where "U" is a post-stretch rotation
        // in the parametric domain, and R Lambda R^T describes how material is
        // stretched in the (b0, b1) tangent plane.
        // Column j of R is the principal stretch vector stretched by lambda[j]
        auto &R = m_R[ti];
        auto &U = m_U[ti];
        auto lambda = m_lambda.row(ti);
        {
            Eigen::JacobiSVD<M2d> svd(JB, Eigen::ComputeFullU | Eigen::ComputeFullV);
            M2d tmp = svd.matrixU();
            lambda = svd.singularValues();
            // Note: we want to make sure both U *and* R are true rotations, not reflections.
            // If det(JB) < 0, a singular value needs to be flipped negative
            // (along with its column in tmp), which will guarantee a positive
            // determinant of U = tmp * V^T.
            // But R could still be a reflection; we negate its last column in
            // this case (which leaves the mapping U R Lambda R^T unchanged).
            if (JB.determinant() < 0) {
                tmp.col(1) *= -1;
                lambda[1]  *= -1;
            }
            U = tmp * svd.matrixV().transpose(); // positive determinant
            R = svd.matrixV();
            if (R.determinant() < 0) { R.col(1) *= -1; }
        }
        m_alpha[ti] = std::min(m_alphaMax, std::max(m_alphaMin, lambda[0]));
        Vector2D lambda_target(m_alpha[ti], 1.0);

        M2d M = U * (R * (lambda_target.asDiagonal() * R.transpose()));
        m_M_Bt[ti] = M * m_B[ti].transpose();
    };

    const size_t nt = m.numTris();
#if MESHFEM_WITH_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, nt), [&](const tbb::blocked_range<size_t> &b) { for (size_t ti = b.begin(); ti < b.end(); ++ti) process_tri(ti); });
#else
    for (size_t ti = 0; ti < nt; ++ti) process_tri(ti);
#endif

    // Update the energy
    {
        // Accumulate in temporary so other threads don't read intermediate values.
        Real energy = 0;
        for (size_t ti = 0; ti < nt; ++ti)
            energy += 0.5 * (m_J[ti] - m_M_Bt[ti]).squaredNorm() * m.element(ti)->volume();
        m_energy = energy;
    }

    BENCHMARK_STOP_TIMER_SECTION("Local step");
}

void LocalGlobalParametrizer::runIteration() {
    const auto &m = mesh();
    const size_t nv = m.numVertices();

    // Global step
    BENCHMARK_START_TIMER_SECTION("Global step");
    // Compute RHS vectors
    UVMap rhs_uv = UVMap::Zero(nv, 2);
    for (auto tri : m.elements()) {
        const size_t ti = tri.index();
        for (auto v : tri.vertices())
            rhs_uv.row(v.index()) += (m_M_Bt[ti] * tri->gradBarycentric().col(v.localIndex())) * tri->volume();
    }

    // Solve two Poisson equations
    UVMap uv_new(m_uv.rows(), 2);
    Eigen::VectorXd soln;
    L->solve(rhs_uv.col(0), soln);
    uv_new.col(0) = soln;
    L->solve(rhs_uv.col(1), soln);
    uv_new.col(1) = soln;
    BENCHMARK_STOP_TIMER_SECTION("Global step");

    setUV(uv_new); // Update Jacobians and run the next local step, allowing us to evaluate energy.
}

Real LocalGlobalParametrizer::leftStretchAngle(size_t i) const {
    Eigen::Rotation2D<Real> UR(getU(i) * getR(i));
    return UR.angle();
}

Real LocalGlobalParametrizer::rightStretchAngle(size_t i) const {
    return Eigen::Rotation2D<Real>(getR(i)).angle();
}

LocalGlobalParametrizer::~LocalGlobalParametrizer() { }

////////////////////////////////////////////////////////////////////////////////
// RegularizedParametrizer: Global nonlinear energy with auxiliary variables
////////////////////////////////////////////////////////////////////////////////
// Initialize from the local-global parametrizer
RegularizedParametrizer::RegularizedParametrizer(LocalGlobalParametrizer &lgparam)
    : Parametrizer(lgparam.meshPtr()),
      m_alphaMin(lgparam.alphaMin()),
      m_alphaMax(lgparam.alphaMax())
{
    const size_t nt = mesh().numTris();
    m_phi.resize(nt);
    m_psi.resize(nt);
    m_alpha.resize(nt);

    // Initialize the variable fields from the local-global parametrizer
    for (size_t i = 0; i < nt; ++i) {
        m_phi[i]   = lgparam. leftStretchAngle(i);
        m_psi[i]   = lgparam.rightStretchAngle(i);
        m_alpha[i] = lgparam.getAlpha(i);
    }

    setUV(lgparam.uv());

    // Cache the (constant) Laplacian block of the Hessian.
    m_laplacian = SuiteSparseMatrix(Laplacian::construct(mesh()));
}

void RegularizedParametrizer::m_evalIterate() {
    // Accumulate energy contributions in temporaries so that other threads don't read intermediate values.
    Real fittingEnergy = 0,
        alphaRegEnergy = 0,
          phiRegEnergy = 0;

    const size_t nt = mesh().numTris();
    m_M.resize(nt);
    m_U.resize(nt);
    m_V.resize(nt);
    m_dU_dphi.resize(nt);
    m_dV_dpsi.resize(nt);

    for (const auto &tri : mesh().elements()) {
        const size_t ti = tri.index();
        auto &M = m_M[ti];
        auto &U = m_U[ti];
        auto &V = m_V[ti];
        auto &dU_dphi = m_dU_dphi[ti];
        auto &dV_dpsi = m_dV_dpsi[ti];

        U = Eigen::Rotation2D<Real>(m_phi[ti]).matrix();
        V = Eigen::Rotation2D<Real>(m_psi[ti]).matrix();

        dU_dphi = Eigen::Rotation2D<Real>(m_phi[ti] + M_PI / 2).matrix();
        dV_dpsi = Eigen::Rotation2D<Real>(m_psi[ti] + M_PI / 2).matrix();

        M = U * Vector2D(m_alpha[ti], 1.0).asDiagonal() * V.transpose();

        const M2d JB = m_J[ti] * m_B[ti];
        fittingEnergy += (JB - M).squaredNorm() * tri->volume();
    }

    // Edge-based regularization terms
    for (const auto &he : mesh().halfEdges()) {
        if (he.isBoundary() || !he.isPrimary()) continue;
        const size_t ti = he.tri().index(),
                     tj = he.opposite().tri().index();

        // phiRegEnergy += std::pow(1 - cos(2 * (m_phi[ti] - m_phi[tj])), m_phi_reg_p * 0.5);
        phiRegEnergy += std::pow(std::abs(sin(m_phi[ti] - m_phi[tj])), m_phi_reg_p);

        if (!m_variableAlpha) continue;
        alphaRegEnergy += std::pow(std::abs(m_alpha[ti] - m_alpha[tj]), m_alpha_reg_p);
    }

    m_energy = (0.5 * fittingEnergy) + (m_alpha_reg_w / m_alpha_reg_p) * alphaRegEnergy + (m_phi_reg_w / m_phi_reg_p) * phiRegEnergy;
}

Eigen::VectorXd RegularizedParametrizer::gradient(EnergyType etype) const {
    Eigen::VectorXd result(numVars());
    result.setZero();
    Eigen::Map<UVMap> grad_uv(result.data(), m_uv.rows(), m_uv.cols());

    // Gradient of fitting energy
    if ((etype == EnergyType::Full) || (etype == EnergyType::Fitting)) {
        for (const auto &tri : mesh().elements()) {
            const size_t ti = tri.index();
            const M2d JB = m_J[ti] * m_B[ti];
            M2d scaled_dist = tri->volume() * (JB - m_M[ti]);

            // Gradient wrt parametrization:
            for (const auto &v : tri.vertices())
                grad_uv.row(v.index()) += (scaled_dist * m_B[ti].transpose()) * tri->gradBarycentric().col(v.localIndex());

            Vector2D tgt_sigma(m_alpha[ti], 1.0);
            // Gradient wrt rotations:
            result[phiOffset() + ti] -= (scaled_dist.transpose() * m_dU_dphi[ti] * tgt_sigma.asDiagonal() *       m_V[ti].transpose()).trace();
            result[psiOffset() + ti] -= (scaled_dist.transpose() *       m_U[ti] * tgt_sigma.asDiagonal() * m_dV_dpsi[ti].transpose()).trace();

            if (!m_variableAlpha) continue;

            result[alphaOffset() + ti] -= ((scaled_dist.transpose() * m_U[ti].col(0)) * m_V[ti].col(0).transpose()).trace();
        }
    }

    // Gradient of edge-based regularization terms
    for (const auto &he : mesh().halfEdges()) {
        if (he.isBoundary() || !he.isPrimary()) continue;
        const size_t ti = he.tri().index(),
                     tj = he.opposite().tri().index();

        // Phi regularization
        if ((etype == EnergyType::Full) || (etype == EnergyType::PhiRegularization)) {
            Real phi_diff = m_phi[ti] - m_phi[tj];
            Real s = sin(phi_diff),
                 c = cos(phi_diff);
            // Using std::copysign(1.0, s) doesn't work since it gives bad derivatives around phi_diff = 0.
            // We get better results explicitly setting the derivative equal to zero in this case.
            Real sign = 0.0;
            if (s > 0) sign =  1.0;
            if (s < 0) sign = -1.0;

            // Real val = m_phi_reg_w * std::pow(1 - cos(2 * phi_diff), m_phi_reg_p * 0.5 - 1.0) * sin(2 * phi_diff);

            Real val;
            if (m_phi_reg_p == 1.0) { val = m_phi_reg_w * c * sign; }
            else                    { val = m_phi_reg_w * std::pow(std::abs(s), m_phi_reg_p - 1.0) * c * sign; } // This is well-behaved for p > 0 (finite, non-nan value)

            result[phiOffset() + ti] += val;
            result[phiOffset() + tj] -= val;
        }

        // Alpha regularization
        if (m_variableAlpha && ((etype == EnergyType::Full) || (etype == EnergyType::AlphaRegularization))) {
            Real alpha_diff = m_alpha[ti] - m_alpha[tj];

            // Using std::copysign(1.0, alpha_diff) doesn't work since it gives bad derivatives around alpha_diff = 0.
            // We get better results explicitly setting the derivative equal to zero in this case.
            Real sign = 0.0;
            if (alpha_diff > 0) sign =  1.0;
            if (alpha_diff < 0) sign = -1.0;

            Real val;
            if (m_alpha_reg_p == 1.0) { val = m_alpha_reg_w * sign; }
            else                      { val = m_alpha_reg_w * std::pow(std::abs(alpha_diff), m_alpha_reg_p - 1.0) * sign; }
            result[alphaOffset() + ti] += val;
            result[alphaOffset() + tj] -= val;
        }
    }

    return result;
}

SuiteSparseMatrix RegularizedParametrizer::hessianSparsityPattern(Real val) const {
    SuiteSparseMatrix result(numVars(), numVars());
    result.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
    result.Ap.reserve(numVars() + 1);

    auto &Ap = result.Ap;
    auto &Ai = result.Ai;

    auto addIdx = [&](const size_t idx) { Ai.push_back(idx); };

    auto finalizeCol = [&]() {
        const size_t colStart = Ap.back();
        const size_t colEnd = Ai.size();
        Ap.push_back(colEnd);
        std::sort(Ai.begin() + colStart, Ai.begin() + colEnd);
    };

    // Build the sparsity pattern in compressed form one column (variable) at a time.
    result.Ap.push_back(0);

    // Laplacian blocks: each vertex value interacts with itself and its neighbors
    const auto &m = mesh();
    const size_t nv = m.numVertices();
    for (size_t uvo = 0; uvo < 2; ++uvo) { // 0: u variables, 1: v variables
        for (const auto &v : m.vertices()) {
            size_t vi = v.index() + uvo * nv;
            addIdx(vi);
            for (const auto &he : v.incidentHalfEdges()) {
                size_t ui = he.tail().index() + uvo * nv;
                if (ui < vi) addIdx(ui);
            }
            finalizeCol();
        }
    }

    // Tri field columns: interact with corner vertices, neighbors, and selves
    const size_t phio = phiOffset(), psio = psiOffset(), alphao = alphaOffset();
    const size_t numTriFields = variableAlpha() ? 3 : 2;
    for (size_t fieldOffset = 0; fieldOffset < numTriFields; ++fieldOffset) { // 0: phi variables, 1: psi variables, 2: alpha variables
        for (const auto &tri : m.elements()) {
            const size_t tj = tri.index();
            for (const auto &v : tri.vertices()) {
                addIdx(v.index());      // u variable
                addIdx(v.index() + nv); // v variable
            }
            addIdx(phio + tj); // phi-phi/phi-psi/phi-alpha interaction
            if (fieldOffset > 0) addIdx(psio   + tj); // psi-psi/psi-alpha interaction
            if (fieldOffset > 1) addIdx(alphao + tj); // alpha-alpha interaction

            // Laplacian-style regularization (upper triangle)
            for (const auto &tri_i : tri.neighbors()) {
                if (!tri_i || (size_t(tri_i.index()) > tj)) continue;
                const size_t ti = tri_i.index();
                if (fieldOffset == 0) { addIdx(phio   + ti); } //   phi regularization interaction
                if (fieldOffset == 2) { addIdx(alphao + ti); } // alpha regularization interaction
            }

            finalizeCol();
        }
    }

    result.nz = result.Ai.size();
    result.Ax.assign(result.nz, val);
    return result;
}

SuiteSparseMatrix RegularizedParametrizer::hessian(EnergyType etype) const {
    SuiteSparseMatrix H = hessianSparsityPattern();
    hessian(H, etype);
    return H;
}

void RegularizedParametrizer::hessian(SuiteSparseMatrix &H, EnergyType etype) const {
    const size_t uo = uOffset(),
                 vo = vOffset(),
                 phio   = phiOffset(),
                 psio   = psiOffset(),
                 alphao = alphaOffset();

    if ((etype == EnergyType::Full) || (etype == EnergyType::Fitting)) {
        // u-u, v-v (Laplacian)
        for (const auto &entry : m_laplacian) {
            H.addNZ(uo + entry.i, uo + entry.j, entry.v);
            H.addNZ(vo + entry.i, vo + entry.j, entry.v);
        }

        for (const auto &tri : mesh().elements()) {
            const size_t ti = tri.index();
            Real A = tri->volume();
            Vector2D tgt_sigma(m_alpha[ti], 1.0);

            // target_fields-u, target_fields-v
            M2d dM_dphi   = m_dU_dphi[ti] * tgt_sigma.asDiagonal() *       m_V[ti].transpose(),
                dM_dpsi   =       m_U[ti] * tgt_sigma.asDiagonal() * m_dV_dpsi[ti].transpose(),
                dM_dalpha = m_U[ti].col(0) * m_V[ti].col(0).transpose();
            for (const auto &v : tri.vertices()) {
                Vector2D dE_duv_dphi = -A * ((dM_dphi * m_B[ti].transpose()) * tri->gradBarycentric().col(v.localIndex()));
                H.addNZ(uo + v.index(), phio + tri.index(), dE_duv_dphi[0]);
                H.addNZ(vo + v.index(), phio + tri.index(), dE_duv_dphi[1]);

                Vector2D dE_duv_dpsi = -A * ((dM_dpsi * m_B[ti].transpose()) * tri->gradBarycentric().col(v.localIndex()));
                H.addNZ(uo + v.index(), psio + ti, dE_duv_dpsi[0]);
                H.addNZ(vo + v.index(), psio + ti, dE_duv_dpsi[1]);

                Vector2D dE_duv_dalpha = -A * ((dM_dalpha * m_B[ti].transpose()) * tri->gradBarycentric().col(v.localIndex()));
                H.addNZ(uo + v.index(), alphao + ti, dE_duv_dalpha[0]);
                H.addNZ(vo + v.index(), alphao + ti, dE_duv_dalpha[1]);
            }

            M2d dist = m_J[ti] * m_B[ti] - m_M[ti];
            M2d d2M_dphi_dpsi = m_dU_dphi[ti] * tgt_sigma.asDiagonal() * m_dV_dpsi[ti].transpose();

            M2d d2M_dphi_dalpha = m_dU_dphi[ti].col(0) * m_V[ti].col(0).transpose(),
                d2M_dpsi_dalpha = m_U[ti].col(0) * m_dV_dpsi[ti].col(0).transpose();

            // psi-phi
            H.addNZ(phio + ti, psio + ti, A * ((dM_dphi.transpose() * dM_dpsi).trace() - (dist.transpose() * d2M_dphi_dpsi).trace()));

            // phi-phi, psi-psi
            // Note: d^2U/dphi^2 = -U, so d^2M/dphi^2 = -M = d^2M/dpsi^2
            Real dist_contract_neg_d2M_dangle2 = (dist.transpose() * m_M[ti]).trace();
            H.addNZ(phio + ti, phio + ti, A * (dM_dphi.squaredNorm() + dist_contract_neg_d2M_dangle2));
            H.addNZ(psio + ti, psio + ti, A * (dM_dpsi.squaredNorm() + dist_contract_neg_d2M_dangle2));

            if (!variableAlpha()) continue;

            // phi-alpha, psi-alpha
            H.addNZ(phio + ti, alphao + ti, A * ((dM_dphi.transpose() * dM_dalpha).trace() - (dist.transpose() * d2M_dphi_dalpha).trace()));
            H.addNZ(psio + ti, alphao + ti, A * ((dM_dpsi.transpose() * dM_dalpha).trace() - (dist.transpose() * d2M_dpsi_dalpha).trace()));
            // alpha-alpha (note d2M_dalpha_dalpha = 0)
            H.addNZ(alphao + ti, alphao + ti, A * dM_dalpha.squaredNorm());
        }
    }

    // Regularization terms' Hessians (edge-based)
    for (const auto &he : mesh().halfEdges()) {
        if (he.isBoundary()) continue;
        const size_t ti = he.tri().index(),
                     tj = he.opposite().tri().index();
        if (ti > tj) continue; // work with upper triangle contributions only (also ensures each edge is visited only once)

        if ((etype == EnergyType::Full) || (etype == EnergyType::PhiRegularization)) {
            Real phi_diff = m_phi[ti] - m_phi[tj];
            // Real c = cos(2 * phi_diff),
            //      s = sin(2 * phi_diff);
            // Real val = std::pow(1 - c, m_phi_reg_p * 0.5 - 2.0) * s * s * (m_phi_reg_p * 0.5 - 1.0) +
            //            std::pow(1 - c, m_phi_reg_p * 0.5 - 1.0) * c;
            // val *= 2 * m_phi_reg_w;

            Real s = sin(phi_diff),
                 c = cos(phi_diff);
            // Using std::copysign(1.0, s) doesn't work since it gives bad derivatives around phi_diff = 0.
            // We get better results explicitly setting the derivative equal to zero in this case.
            Real sign = 0.0;
            if (s > 0) sign =  1.0;
            if (s < 0) sign = -1.0;

            // Note: second derivatives in both the "p = 1" and "p = 2" cases are well behaved,
            // but they blow up around phi_diff = 0 when "1 < p < 2". We discard Hessian
            // contributions near this blowup.
            Real val = 0.0;
            const Real p = m_phi_reg_p;
            if      (m_phi_reg_p == 1.0) { val = -std::abs(s); }
            else if (m_phi_reg_p == 2.0) { val = c * c - s * s; } // equivalently: cos(2 * phi_diff)
            else if (std::abs(s) > 1e-4) { val = (p - 1) * std::pow(std::abs(s), p - 2.0) * c * c
                                                         - std::pow(std::abs(s), p - 1.0) * s * sign; }
            else                         { val = 0.0; } // discard Hessian contributions in cases that blow up

            val *= m_phi_reg_w;

            if (val != 0.0) {
                H.addNZ(phio + ti, phio + tj, -val);
                H.addNZ(phio + tj, phio + tj,  val);
                H.addNZ(phio + ti, phio + ti,  val);
            }
        }

        const Real alpha_diff = m_alpha[ti] - m_alpha[tj];
        if (!((etype == EnergyType::Full) || (etype == EnergyType::AlphaRegularization))) continue;
        if (!m_variableAlpha || (m_alpha_reg_p == 1.0) || ((m_alpha_reg_p < 2.0) && std::abs(alpha_diff) < 1e-14)) continue;

        Real val = m_alpha_reg_w * (m_alpha_reg_p - 1.0) * std::pow(std::abs(alpha_diff), m_alpha_reg_p - 2.0);
        H.addNZ(alphao + ti, alphao + tj, -val);
        H.addNZ(alphao + tj, alphao + tj,  val);

        // H.addNZ(alphao + tj, alphao + ti, -val); (lower triangle)
        H.addNZ(alphao + ti, alphao + ti,  val);
    }

}

////////////////////////////////////////////////////////////////////////////////
// RegularizedParametrizerSVD: Global nonlinear energy with uv variables only
////////////////////////////////////////////////////////////////////////////////
RegularizedParametrizerSVD::RegularizedParametrizerSVD(const std::shared_ptr<Mesh> &inMesh, const UVMap &uvInit, Real amin, Real amax, bool transformForRigidMotionConstraint)
    : Parametrizer(inMesh), dualLaplacianStencil(*inMesh), m_alphaMin(amin), m_alphaMax(amax)
{
    Eigen::RowVector2d c = uvInit.colwise().mean();
    // Usually when we are disabling the transformForRigidMotionConstraint
    // setting it's because are initializing from a solution that already satisfies
    // rigid motion pin constraints with a vertex perfectly at the origin;
    // don't translate in this case.
    if (!transformForRigidMotionConstraint) c.setZero();
    int centralIdx;
    (uvInit.rowwise() - c).rowwise().squaredNorm().minCoeff(&centralIdx);
    UVMap uvTransformed = uvInit;
    uvTransformed.rowwise() -= uvInit.row(centralIdx);

    // If we are allowed to transform the UV initialization for better
    // enforcing rigid motion constraints (if transformForRigidMotionConstraint
    // is true), find the rotation that brings the furthest vertex to the u axis,
    // where a v = 0 constraint can be used to pin down rotation.
    int rotationPinVar = 0;
    if (transformForRigidMotionConstraint) {
        int furthestIdx;
        uvTransformed.rowwise().squaredNorm().maxCoeff(&furthestIdx);
        Eigen::RowVector2d p = uvTransformed.row(furthestIdx);
        Real angle = std::atan2(p[1], p[0]);
        // rotate by -angle to bring point p onto the u axis. This is the same as applying the
        // matrix R(-angle)^T = R(angle) to uvTransformed on the right.
        uvTransformed = (uvTransformed * Eigen::Rotation2D<Real>(angle).toRotationMatrix()).eval();
        rotationPinVar = furthestIdx;
        setUV(uvTransformed); // Necesssary side-effect: enables the use of vOffset()
    }
    else {
        // Find a distant vertex that still closely satisfies v = 0 so that no
        // rotation is needed to apply rigid motion constraints.
        Eigen::VectorXd vDist = uvTransformed.col(1).cwiseAbs();
        Real threshold = 1e-8 * vDist.maxCoeff();
        Eigen::VectorXd admissibleUDist = (vDist.array() < threshold).select(uvTransformed.col(0).cwiseAbs(), Eigen::VectorXd::Zero(uvTransformed.rows()));
        int furthestIdx;
        Real val = admissibleUDist.maxCoeff(&furthestIdx);
        if (val == 0.0) { throw std::runtime_error("No admissible rotation pin vertex"); }
        rotationPinVar = furthestIdx;
        setUV(uvInit); // Necesssary side-effect: enables the use of vOffset()
    }

    m_rigidMotionPinVars[0] = uOffset() + centralIdx;
    m_rigidMotionPinVars[1] = vOffset() + centralIdx;
    m_rigidMotionPinVars[2] = vOffset() + rotationPinVar;

    // Construct per-triangle averaged shape operator
    {
        const auto &m = mesh();
        CurvatureInfo cinfo(mesh());

        m_shapeOperators.reserve(m.numTris());
        m_shapeOperators.clear();

        for (auto tri : m.tris()) {
            const size_t ti = tri.index();
            M2d S(M2d::Zero());
            for (auto corner : tri.vertices()) {
                M2d d;
                V2d k(cinfo.kappa_1[corner.index()],
                      cinfo.kappa_2[corner.index()]);
                for (size_t i = 0; i < 2; ++i) {
                    // Project curvature direction onto triangle tangent plane and re-normalize.
                    d.col(i) = (cinfo.d(i).row(corner.index()) * m_B[ti]).normalized().eval();
                }
                S += d * k.asDiagonal() * d.transpose();
            }
            S /= 3.0;
            m_shapeOperators.push_back(S);
        }
    }
}

void RegularizedParametrizerSVD::m_evalIterate() {
    const size_t nt = mesh().numTris();

    m_svds.resize(nt);
    m_phi.resize(nt);
    m_alpha.resize(nt);

    BENCHMARK_START_TIMER_SECTION("Update SVD Sensitivities");

    auto processTri = [&](size_t ti) {
        m_svds[ti].setMatrix(m_J[ti] * m_B[ti]); // compute SVD of each Jacobian and its sensitivity information
        const auto &svd = m_svds[ti];

        const Eigen::Vector2d u0 = svd.u(0);
        m_phi[ti]   = std::atan2(u0[1], u0[0]);
        m_alpha[ti] = svd.sigma(0);
    };

#if MESHFEM_WITH_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, nt), [&](const tbb::blocked_range<size_t> &b) { for (size_t ti = b.begin(); ti < b.end(); ++ti) processTri(ti); });
#else
 for (size_t ti = 0; ti < nt; ++ti) processTri(ti);
#endif
    BENCHMARK_STOP_TIMER_SECTION("Update SVD Sensitivities");
}

Real RegularizedParametrizerSVD::energy(EnergyType etype) const {
    Real fittingEnergy = 0.0,
        alphaRegEnergy = 0.0,
          phiRegEnergy = 0.0,
         bendRegEnergy = 0.0;
    const auto &m = mesh();
    const Real &surfaceArea = m.volume();

    for (const auto &tri : m.tris()) {
        const size_t ti = tri.index();
        const auto &svd = m_svds[ti];
        Real stretchDeviationAbs = 0.0;
        if (m_alpha[ti] < m_alphaMin) stretchDeviationAbs = m_alphaMin - m_alpha[ti];
        if (m_alpha[ti] > m_alphaMax) stretchDeviationAbs = m_alpha[ti] - m_alphaMax;
        // stretchDeviation = 0; // for disabling the less-smooth alpha fitting term

        fittingEnergy += tri->volume() * (0.5 * std::pow((svd.sigma(1) - 1.0), 2)
                                      +  (1.0 / m_stretch_deviation_p) * std::pow(stretchDeviationAbs, m_stretch_deviation_p));

        Real kappa = svd.v(1).dot(m_shapeOperators[ti] * svd.v(1));
        bendRegEnergy += 0.25 * kappa * kappa * tri->volume();
    };
    if (scaleInvariantFittingEnergy) fittingEnergy /= surfaceArea;

    // Dual Laplacian-based regularization terms
    dualLaplacianStencil.visit_edges([this, &phiRegEnergy, &alphaRegEnergy](size_t i, size_t j, Real w_ij) {
        const Eigen::Vector2d ui = m_svds[i].u(0);
        const Eigen::Vector2d uj = m_svds[j].u(0);

        // |sin(phi_i - phi_j)|^p expressed in terms of first singular vector components using angle difference formula:
        //      sin(a - b) = sin(a) cos(b) - cos(a) sin(b)
        phiRegEnergy   += w_ij * std::pow(std::abs(ui[1] * uj[0] - ui[0] * uj[1]), m_phi_reg_p);
        alphaRegEnergy += w_ij * std::pow(std::abs(m_alpha[i] - m_alpha[j]),     m_alpha_reg_p);
    });

    if (etype != EnergyType::Full) {
        if (etype != EnergyType::Fitting              )  fittingEnergy = 0.0;
        if (etype != EnergyType::AlphaRegularization  ) alphaRegEnergy = 0.0;
        if (etype != EnergyType::PhiRegularization    )   phiRegEnergy = 0.0;
        if (etype != EnergyType::BendingRegularization)  bendRegEnergy = 0.0;
    }
    return fittingEnergy + m_bend_reg_w * bendRegEnergy + (m_alpha_reg_w / m_alpha_reg_p) * alphaRegEnergy + (m_phi_reg_w / m_phi_reg_p) * phiRegEnergy;
}

Eigen::VectorXd RegularizedParametrizerSVD::gradient(EnergyType etype) const {
    BENCHMARK_START_TIMER_SECTION("RegularizedParametrizerSVD gradient");
    Eigen::VectorXd result(numVars());
    result.setZero();
    Eigen::Map<UVMap> grad_uv(result.data(), m_uv.rows(), m_uv.cols());
    const Real invSurfaceArea = 1.0 / mesh().volume();

    for (const auto &tri : mesh().elements()) {
        const size_t ti = tri.index();
        const auto &svd_i = m_svds[ti];
        Real A = tri->volume();

        M2d ET_prime(M2d::Zero()); // derivative of objective wrt JB

        if ((etype == EnergyType::Full) || (etype == EnergyType::Fitting)) {
            ET_prime = (A * (svd_i.sigma(1) - 1.0)) * svd_i.dsigma(1);

            Real stretchDeviationAbs = 0.0, sign = 1.0;
            if (m_alpha[ti] < m_alphaMin) { stretchDeviationAbs = m_alphaMin - m_alpha[ti]; sign = -1.0; }
            if (m_alpha[ti] > m_alphaMax) { stretchDeviationAbs = m_alpha[ti] - m_alphaMax; sign =  1.0; }
            ET_prime += (A * sign * std::pow(stretchDeviationAbs, m_stretch_deviation_p - 1)) * svd_i.dsigma(0);

            if (scaleInvariantFittingEnergy) ET_prime *= invSurfaceArea;
        }

        if ((etype == EnergyType::Full) || (etype == EnergyType::BendingRegularization)) {
            V2d Sv_2 = m_shapeOperators[ti] * svd_i.v(1);
            Real kappa = svd_i.v(1).dot(Sv_2);
            ET_prime += (A * m_bend_reg_w * kappa) * (Sv_2[0] * svd_i.dv1(0)
                                                    + Sv_2[1] * svd_i.dv1(1));
        }

        // Dual Laplacian regularization terms
        dualLaplacianStencil.visit(ti, [this, ti, etype, &ET_prime, &svd_i](size_t /* ti */, size_t tj, Real w_ij) {
            const auto &svd_j = m_svds[tj];

            if ((etype == EnergyType::Full) || (etype == EnergyType::AlphaRegularization)) {
                Real alpha_diff = m_alpha[ti] - m_alpha[tj];

                // Using std::copysign(1.0, alpha_diff) doesn't work since it gives bad derivatives around alpha_diff = 0.
                // We get better results explicitly setting the derivative equal to zero in this case.
                Real sign = 0.0;
                if (alpha_diff > 0) sign =  1.0;
                if (alpha_diff < 0) sign = -1.0;

                Real d_alpha_reg_d_sigma0_i;
                if (m_alpha_reg_p == 1.0) { d_alpha_reg_d_sigma0_i = m_alpha_reg_w * sign; }
                else                      { d_alpha_reg_d_sigma0_i = m_alpha_reg_w * std::pow(std::abs(alpha_diff), m_alpha_reg_p - 1.0) * sign; }
                ET_prime += (w_ij * d_alpha_reg_d_sigma0_i) * svd_i.dsigma(0); // could be combined with alpha fitting coefficient for small speedup...
            }

            if ((etype == EnergyType::Full) || (etype == EnergyType::PhiRegularization)) {
                const Eigen::Vector2d ui = svd_i.u(0);
                const Eigen::Vector2d uj = svd_j.u(0);
                Real s = ui[1] * uj[0] - ui[0] * uj[1];

                // Using std::copysign(1.0, s) doesn't work since it gives bad derivatives around phi_diff = 0.
                // We get better results explicitly setting the derivative equal to zero in this case.
                Real sign = 0.0;
                if (s > 0) sign =  1.0;
                if (s < 0) sign = -1.0;

                V2d d_phi_reg_d_u0_i(-uj[1], uj[0]);
                if (m_phi_reg_p == 1.0) { d_phi_reg_d_u0_i *= m_phi_reg_w * sign; }
                else                    { d_phi_reg_d_u0_i *= m_phi_reg_w * std::pow(std::abs(s), m_phi_reg_p - 1.0) * sign; } // This is well-behaved for p > 0 (finite, non-nan value)

                d_phi_reg_d_u0_i *= w_ij;
                ET_prime += d_phi_reg_d_u0_i[0] * svd_i.du0(0)
                         +  d_phi_reg_d_u0_i[1] * svd_i.du0(1);
            }
        });

        M23d grad_barycentric_local = m_B[ti].transpose() * tri->gradBarycentric();
        for (const auto &v : tri.vertices()) {
            result[v.index() + uOffset()] += ET_prime.row(0).dot(grad_barycentric_local.col(v.localIndex()));
            result[v.index() + vOffset()] += ET_prime.row(1).dot(grad_barycentric_local.col(v.localIndex()));
        }
    }

    BENCHMARK_STOP_TIMER_SECTION("RegularizedParametrizerSVD gradient");
    return result;
}

SuiteSparseMatrix RegularizedParametrizerSVD::hessianSparsityPattern(Real val) const {
    SuiteSparseMatrix result(numVars(), numVars());
    result.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
    result.Ap.reserve(numVars() + 1);

    auto &Ap = result.Ap;
    auto &Ai = result.Ai;

    // Build the sparsity pattern in compressed form one column (variable) at a time.
    result.Ap.push_back(0);
    // Vertices interact with themselves, with all vertices appearing in each
    // incident triangle T, and with all vertices of triangles appearing in the
    // dual laplacian energy stencil for T.
    const auto &m = mesh();
    const size_t nv = m.numVertices();
    for (size_t uvo = 0; uvo < 2; ++uvo) { // 0: u variables, 1: v variables
        for (const auto &v : m.vertices()) {
            // Collect all vertices interacting with v in sorted order.
            std::set<size_t> stencilVertices;
            // For each incident triangle, add all vertices in the energy stencil.
            for (const auto &he : v.incidentHalfEdges()) {
                const auto tri_i = he.tri();
                if (!tri_i) continue;
                // Add all vertices of the incident triangle, tri_i
                for (const auto &v_j : tri_i.vertices()) stencilVertices.insert(v_j.index());
                // Add all vertices of triangles in stencil of tri_i
                dualLaplacianStencil.visit(tri_i.index(), [&stencilVertices, &m](size_t /* i */, size_t j, Real /* w_ij */) {
                        for (const auto &v_j : m.tri(j).vertices())
                            stencilVertices.insert(v_j.index());
                    });
            }
            // std::cout << "Num stencil vertices: " << stencilVertices.size() << std::endl;

            size_t var_i = v.index() + uvo * nv;

            // Add variables var_j interacting with var_i to var_i's column in sorted order:
            for (size_t uvo_j = 0; uvo_j < 2; ++uvo_j) {
                for (size_t v_j : stencilVertices) {
                    size_t var_j = v_j + uvo_j * nv;
                    if (var_j <= var_i) Ai.push_back(var_j);
                }
            }

            // Finalize the column
            Ap.push_back(Ai.size());
        }
    }

    result.nz = result.Ai.size();
    result.Ax.assign(result.nz, val);
    return result;
}

// Note: the following parallel_for + thread local data implementation is
// significantly faster than parallel_reduce. This is because parallel_reduce
// spawns way more tasks than worker threads, which incurs high split/join
// overhead due to the sparse matrix copy/accumulate.
SuiteSparseMatrix RegularizedParametrizerSVD::hessian(EnergyType etype, bool projectionMask) const {
    SuiteSparseMatrix H = hessianSparsityPattern();
    hessian(H, etype, projectionMask);

    return H;
}

// Potential speedup: exploit (pi/2 rotation transformation) relationship between "d2 sigma_0 / (dA dB)"
// and "d2 sigma_1 / (dA dB)."
void RegularizedParametrizerSVD::hessian(SuiteSparseMatrix &H, EnergyType etype, bool projectionMask) const {
    const size_t uo = uOffset(), vo = vOffset();

    using VSFJ = VectorizedShapeFunctionJacobian<2, V2d>;

    const bool fittingActive = (etype == EnergyType::Full) || (etype == EnergyType::Fitting),
               bendingActive = (etype == EnergyType::Full) || (etype == EnergyType::BendingRegularization),
              alphaRegActive = (etype == EnergyType::Full) || (etype == EnergyType::AlphaRegularization),
                phiRegActive = (etype == EnergyType::Full) || (etype == EnergyType::PhiRegularization);

    const Real invSurfaceArea = 1.0 / mesh().volume();

    using PerElemHessian = Eigen::Matrix<Real, 6, 6>;
    auto assemblePerTriHessian = [&](const size_t ti, SuiteSparseMatrix &Hout) {
        const auto &tri = mesh().element(ti);
        const auto &svd_i = m_svds[ti];
        M23d grad_barycentric_local = m_B[ti].transpose() * tri->gradBarycentric();

        if (fittingActive || bendingActive) {
            const Real A = tri->volume();
            Real stretchDeviationAbs = 0.0, sign = 1.0;
            if (m_alpha[ti] < m_alphaMin) { stretchDeviationAbs = m_alphaMin - m_alpha[ti]; sign = -1.0; }
            if (m_alpha[ti] > m_alphaMax) { stretchDeviationAbs = m_alpha[ti] - m_alphaMax; sign =  1.0; }
            Real sigma1Deviation = svd_i.sigma(1) - 1.0;
            // stretchDeviation = 0; // for disabling the less-smooth alpha fitting term

            const M2d dsigma0 = svd_i.dsigma(0),
                      dsigma1 = svd_i.dsigma(1);
            auto contrib = [&](size_t u_or_v_i, size_t u_or_v_j, size_t li, size_t lj) {
                VSFJ dJB_i(u_or_v_i, grad_barycentric_local.col(li)),
                     dJB_j(u_or_v_j, grad_barycentric_local.col(lj));

                // M2d dJB_i(M2d::Zero()), dJB_j(M2d::Zero());
                // dJB_i.row(u_or_v_i) = grad_barycentric_local.col(li).transpose();
                // dJB_j.row(u_or_v_j) = grad_barycentric_local.col(lj).transpose();
                Real result = 0;
                if (fittingActive) {
                    V2d d2Sigma = svd_i.d2Sigma(dJB_i, dJB_j);

                    // Derivative with respect to <u or v j> of (sigma1 - 1.0) dsigma1 / d_<uv_i>
                    result += doubleContract(dsigma1, dJB_i) * doubleContract(dsigma1, dJB_j)
                           +  sigma1Deviation * d2Sigma[1];
                    // Derivative with respect to <u or v j> of stretchDeviation dsigma0 / d_<uv_i>
                    if (stretchDeviationAbs != 0) {
                        // result += dsigma0.row(u_or_v_i).dot(grad_barycentric_local.col(li)) * dsigma0.row(u_or_v_j).dot(grad_barycentric_local.col(lj))
                        //        +  stretchDeviation * d2Sigma[0];
                        result += ((m_stretch_deviation_p - 1) * std::pow(stretchDeviationAbs, m_stretch_deviation_p - 2)) * doubleContract(dsigma0, dJB_i) * doubleContract(dsigma0, dJB_j)
                               +  (sign * std::pow(stretchDeviationAbs, m_stretch_deviation_p - 1)) * d2Sigma[0];
                    }

                    if (scaleInvariantFittingEnergy) result *= invSurfaceArea;
                }
                if (bendingActive) {
                    V2d Sv_2 = m_shapeOperators[ti] * svd_i.v(1);
                    Real kappa = svd_i.v(1).dot(Sv_2);
                    V2d delta_v1_i = svd_i.dv1(dJB_i),
                        delta_v1_j = svd_i.dv1(dJB_j);

                    result += m_bend_reg_w * (kappa * (Sv_2.dot(svd_i.d2v1(dJB_i, dJB_j)) + (m_shapeOperators[ti] * delta_v1_i).dot(delta_v1_j))
                                              + 2.0 * Sv_2.dot(delta_v1_i)
                                                    * Sv_2.dot(delta_v1_j));
                }
                result *= A;

                return result;
            };

            // Evaluate **lower** triangle of per-element Hessian (since this
            // is what Eigen's SelfAdjointEigenSolver uses)
            PerElemHessian H_elem;
            for (size_t vj = 0; vj < 3; ++vj) {
                for (size_t vi = 0; vi < 3; ++vi) {
                    if (vi >= vj) {
                        H_elem(    vi,     vj) = contrib(0, 0, vi, vj);
                        H_elem(3 + vi, 3 + vj) = contrib(1, 1, vi, vj);
                    }
                    H_elem(3 + vi, vj) = contrib(1, 0, vi, vj);
                }
            }

            if (projectionMask) {
                using ESolver  = Eigen::SelfAdjointEigenSolver<PerElemHessian>;
                ESolver Hes(H_elem);
                H_elem = Hes.eigenvectors() * Hes.eigenvalues().cwiseMax(0.0).asDiagonal() * Hes.eigenvectors().transpose();
            }
            else {
                H_elem.template triangularView<Eigen::Upper>() = H_elem.template triangularView<Eigen::Lower>().transpose();
            }

            // Accumulate to upper triangle of Hessian.
            for (const auto &vi : tri.vertices()) {
                for (const auto &vj : tri.vertices()) {
                    if (vi.index() <= vj.index()) {
                        Hout.addNZ(vi.index() + uo, vj.index() + uo, H_elem(    vi.localIndex(),     vj.localIndex()));
                        Hout.addNZ(vi.index() + vo, vj.index() + vo, H_elem(3 + vi.localIndex(), 3 + vj.localIndex()));
                    }
                    Hout.addNZ(vi.index() + uo, vj.index() + vo, H_elem(3 + vj.localIndex(), vi.localIndex()));
                }
            }
        }

        // Dual Laplacian regularization terms
        dualLaplacianStencil.visit(ti, [this, ti, &tri_i = tri, &Hout, &svd_i, alphaRegActive, phiRegActive, &grad_barycentric_local, uo, vo](const size_t /* ti */, const size_t tj, const Real w_ij) {
            if (ti > tj) return; // Visit each stencil edge (unordered triangle pair) exactly once
            const auto &tri_j = mesh().element(tj);

            const auto &svd_j = m_svds[tj];
            M23d grad_barycentric_local_j = m_B[tj].transpose() * tri_j->gradBarycentric();

            Real d_alpha_reg_dsigma0_i = 0.0, d2_alpha_reg_dalpha_i_dalpha_j = 0.0; // alpha reg
            V4d  d_phi_reg_d_u0     (V4d::Zero());
            M4d d2_phi_reg_d_u0_d_u0(M4d::Zero());

            if (alphaRegActive) {
                Real alpha_diff = m_alpha[ti] - m_alpha[tj];
                Real sign = 0.0;
                if (alpha_diff > 0) sign =  1.0;
                if (alpha_diff < 0) sign = -1.0;

                if (m_alpha_reg_p == 1.0) { d_alpha_reg_dsigma0_i = m_alpha_reg_w * sign; }
                else                      { d_alpha_reg_dsigma0_i = m_alpha_reg_w * std::pow(std::abs(alpha_diff), m_alpha_reg_p - 1.0) * sign; }

                if ((m_alpha_reg_p != 1.0) && ((m_alpha_reg_p >= 2.0) || (std::abs(alpha_diff) > 1e-14)))
                    d2_alpha_reg_dalpha_i_dalpha_j = m_alpha_reg_w * (m_alpha_reg_p - 1.0) * std::pow(std::abs(alpha_diff), m_alpha_reg_p - 2.0);
            }

            if (phiRegActive) {
                const Eigen::Vector2d ui = svd_i.u(0);
                const Eigen::Vector2d uj = svd_j.u(0);

                Real s = ui[1] * uj[0] - ui[0] * uj[1];

                Real sign_phi = 0.0;
                if (s > 0) sign_phi =  1.0;
                if (s < 0) sign_phi = -1.0;

                d_phi_reg_d_u0 << -uj[1],  uj[0], // d phi_reg / d u0_i
                                   ui[1], -ui[0]; // d phi_reg / d u0_j

                const Real p = m_phi_reg_p;
                Real scale;
                if (p == 1.0) { scale = m_phi_reg_w * sign_phi; }
                else          { scale = m_phi_reg_w * std::pow(std::abs(s), p - 1.0) * sign_phi; } // This is well-behaved for p > 0 (finite, non-nan value)

                //                /-uj[1]\.
                // derivative of |  uj[0] |  * m_phi_reg_w * |ui[1] * uj[0] - ui[0] * uj[1]|^(p - 1) * sign_phi
                //               |  ui[1] |   \________________________________________________________________/
                //                \-ui[0]/                                  scale
                d2_phi_reg_d_u0_d_u0 << 0, 0, 0, -1,
                                        0, 0, 1,  0,
                                        0, 1, 0,  0,
                                       -1, 0, 0,  0;
                d2_phi_reg_d_u0_d_u0 *= scale;
                if (p != 1.0) { // d scale / d_u0 contribution
                    if (p == 2.0)                               { d2_phi_reg_d_u0_d_u0 += (m_phi_reg_w * d_phi_reg_d_u0) * d_phi_reg_d_u0.transpose(); }
                    else if ((std::abs(s) > 1e-5) || (p > 2.0)) { d2_phi_reg_d_u0_d_u0 += (m_phi_reg_w * (p - 1) * std::pow(std::abs(s), p - 2.0) * d_phi_reg_d_u0) * d_phi_reg_d_u0.transpose(); }
                    // else: discard Hessian terms that blow up...
                }

                d_phi_reg_d_u0 *= scale;
            }

            // Derivative of 2x2 Jacobian of one of the two triangles in the stencil, with respect to the u or v coordinate
            // of one of the vertices in the triangle.
            auto dJB = [&](size_t local_tri_index, size_t local_vertex_index, size_t u_or_v) {
                if (local_tri_index == 0) return VSFJ(u_or_v, grad_barycentric_local  .col(local_vertex_index));
                else                      return VSFJ(u_or_v, grad_barycentric_local_j.col(local_vertex_index));
            };

            // local triangle indices ltk, ltl: 0 --> tri_i, 1 --> tri_j
            auto entry = [&](size_t ltk, size_t ltl, size_t lvk, size_t lvl, size_t u_or_v_k, size_t u_or_v_l) {
                Real result = 0;
                const auto &svd_k = (ltk == 0) ? svd_i : svd_j;
                const auto &svd_l = (ltl == 0) ? svd_i : svd_j;
                auto dJB_k = dJB(ltk, lvk, u_or_v_k);
                auto dJB_l = dJB(ltl, lvl, u_or_v_l);

                {
                    // alpha regularization
                    const double alpha_sign_k = ((ltk == 0) ? 1.0 : -1.0),
                                 alpha_sign_l = ((ltl == 0) ? 1.0 : -1.0);
                    if (ltk == ltl) { result += alpha_sign_k * d_alpha_reg_dsigma0_i * svd_k.d2sigma(0, dJB_k, dJB_l); }
                    result += alpha_sign_k * alpha_sign_l * d2_alpha_reg_dalpha_i_dalpha_j * svd_k.dsigma(0, dJB_k) * svd_l.dsigma(0, dJB_l);
                }

                {
                    // phi regularization
                    if (ltk == ltl) { result += d_phi_reg_d_u0.segment<2>(2 * ltk).dot(svd_k.d2u0(dJB_k, dJB_l)); }
                    result += svd_k.du0(dJB_k).dot(d2_phi_reg_d_u0_d_u0.block<2, 2>(2 * ltk, 2 * ltl) * svd_l.du0(dJB_l));
                }

                return result;
            };

            // Contributions from all ordered pairs of triangles in the stencil
            for (size_t ltk = 0; ltk < 2; ++ltk) {
                const auto &tri_k = (ltk == 0) ? tri_i : tri_j;
                for (size_t ltl = 0; ltl < 2; ++ltl) {
                    const auto &tri_l = (ltl == 0) ? tri_i : tri_j;
                    // Contributions from all pairings of vertices from one triangle with vertices from the other
                    for (const auto &vk : tri_k.vertices()) {
                        for (const auto &vl : tri_l.vertices()) {
                            Hout.addNZ(vk.index() + uo, vl.index() + vo, w_ij * entry(ltk, ltl, vk.localIndex(), vl.localIndex(), 0, 1));
                            if (vk.index() > vl.index()) continue; // lower tri
                            Hout.addNZ(vk.index() + uo, vl.index() + uo, w_ij * entry(ltk, ltl, vk.localIndex(), vl.localIndex(), 0, 0));
                            Hout.addNZ(vk.index() + vo, vl.index() + vo, w_ij * entry(ltk, ltl, vk.localIndex(), vl.localIndex(), 1, 1));
                        }
                    }
                }
            }
        });
    };

    assemble_parallel(assemblePerTriHessian, H, mesh().numTris());
}

}

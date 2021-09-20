#include "MetricFitter.hh"
#include <cassert>
#include <MeshFEM/ParallelAssembly.hh>

void MetricFitter::setImmersion(Eigen::Matrix3Xd P) {
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

    m_rigidMotionPinVars[0] = 3 * c_idx + 0;
    m_rigidMotionPinVars[1] = 3 * c_idx + 1;
    m_rigidMotionPinVars[2] = 3 * c_idx + 2;

    m_rigidMotionPinVars[3] = 3 * p_idx + 1;
    m_rigidMotionPinVars[4] = 3 * p_idx + 2;

    m_rigidMotionPinVars[5] = 3 * q_idx + 2;

    setVars(Eigen::Map<const VXd>(P.data(), 3 * P.cols()));
}

void MetricFitter::setVars(const Eigen::Ref<const VXd> &vars) {
    if (size_t(vars.size()) != numVars()) throw std::runtime_error("Invalid variable size");
    m_currVars = vars;
    const auto &m = mesh();
    m_J.resize(m.numElements());

    for (const auto &e : m.elements()) {
        const size_t ei = e.index();
        auto &J = m_J[ei];
        J = getDeformedTriCornerPositions(ei)
          * e->gradBarycentric().transpose();
        M2d C = J.transpose() * J;
        m_fitEnergy[ei].setMatrix(C);
        m_collapsePrevention[ei].setMatrix(C);
    }

    for (size_t hi = 0; hi < m_halfedgeForHinge.size(); ++hi) {
        auto he = m.halfEdge(m_halfedgeForHinge[hi]);
        auto stencil = bendingHingeStencil(he);
        m_edgeHinges[hi].setDeformedConfiguration(deformedVtxPos(stencil[0]),
                                                  deformedVtxPos(stencil[1]),
                                                  deformedVtxPos(stencil[2]),
                                                  deformedVtxPos(stencil[3]));
    }
}

void MetricFitter::setBendingReferenceImmersion(const Eigen::Matrix3Xd &P) {
    m_edgeHinges.clear();
    m_edgeHinges.reserve(m_halfedgeForHinge.size());
    const auto &m = mesh();

    for (const int hei : m_halfedgeForHinge) {
        auto he = m.halfEdge(hei);
        auto stencil = bendingHingeStencil(he);
        m_edgeHinges.emplace_back(P.col(stencil[0]),
                                  P.col(stencil[1]),
                                  P.col(stencil[2]),
                                  P.col(stencil[3]));
    }
}

double MetricFitter::energy(EnergyType  etype) const {
    double result = 0;
    const auto &m = mesh();
    // Metric fitting energy
    if ((etype == EnergyType::Full) || (etype == EnergyType::MetricFitting)) {
        for (const auto &e : m.elements())
            result += m_fitEnergy[e.index()].energy() * e->volume();
    }

    // Collapse prevention energy
    if ((etype == EnergyType::Full) || (etype == EnergyType::CollapsePrevention)) {
        for (const auto &e : m.elements())
            result += collapsePreventionWeight * m_collapsePrevention[e.index()].energy() * e->volume();
    }

    // Bending energy
    if ((etype == EnergyType::Full) || (etype == EnergyType::Bending)) {
        for (const auto &hinge : m_edgeHinges)
            result += bendingStiffness * hinge.energy();
    }

    // Gravitational potential energy
    if ((etype == EnergyType::Full) || (etype == EnergyType::Gravitational)) {
        // Assumes uniform (unit) mass density over the elements:
        // the gravitational potential energy is
        //      -sum_e int_e rho g.x dA 
        //      = -sum_e g.xavg * vol
        for (const auto &e : m.elements()) {
            result -= e->volume() / 3 * gravityVector.dot(
                    deformedVtxPos(e.node(0).index()) +
                    deformedVtxPos(e.node(1).index()) +
                    deformedVtxPos(e.node(2).index()));
        }
    }

    return result;
}

MetricFitter::VXd MetricFitter::gradient(EnergyType etype) const {
    VXd g = VXd::Zero(numVars());

    const auto &m = mesh();

    // Metric fitting + collapse prevention energy
    if ((etype == EnergyType::Full) || (etype == EnergyType::MetricFitting) || (etype == EnergyType::CollapsePrevention)) {
        for (const auto &e : m.elements()) {
            const size_t ei = e.index();
            M2d denergy(M2d::Zero());
            if ((etype == EnergyType::Full) || (etype == EnergyType::     MetricFitting)) denergy += m_fitEnergy[ei].denergy();
            if ((etype == EnergyType::Full) || (etype == EnergyType::CollapsePrevention)) denergy += collapsePreventionWeight * m_collapsePrevention[ei].denergy().cast<double>();
            M3d contrib = (2 * e->volume()) * m_J.at(ei) * denergy * e->gradBarycentric();
            for (const auto &n : e.nodes())
                g.segment<3>(3 * n.index()) += contrib.col(n.localIndex());
        }
    }

    // Bending energy
    if ((etype == EnergyType::Full) || (etype == EnergyType::Bending)) {
        for (size_t hi = 0; hi < m_halfedgeForHinge.size(); ++hi) {
            auto he = m.halfEdge(m_halfedgeForHinge[hi]);
            auto stencil = bendingHingeStencil(he);
            auto gradHingeEnergy = m_edgeHinges[hi].gradient();
            for (size_t svi = 0; svi < 4; ++svi)
                g.segment<3>(3 * stencil[svi]) += bendingStiffness * gradHingeEnergy.col(svi);
        }
    }

    // Gravitational potential energy
    if ((etype == EnergyType::Full) || (etype == EnergyType::Gravitational)) {
        // Assumes uniform (unit) mass density over the elements:
        // the gravitational potential energy is
        //      sum_e int_e rho g.x dA
        //      = sum_e g.xavg * vol
        for (const auto &e : m.elements()) {
            for (const auto &n : e.nodes())
                g.segment<3>(3 * n.index()) -= (e->volume() / 3) * gravityVector;
        }
    }

    return g;
}

void MetricFitter::hessian(SuiteSparseMatrix &H, EnergyType etype) const {
    const auto &m = mesh();
    HALocalData<double> haLocalData;

    const bool fittingActive = (etype == EnergyType::Full) || (etype == EnergyType::MetricFitting);
    const bool collapsePreventionActive = (etype == EnergyType::Full) || (etype == EnergyType::CollapsePrevention);

    // Metric fitting + collapse prevention Hessian energy has a per-triangle contribution
    auto perTriContrib = [&](const size_t ei, SuiteSparseMatrix &Hout) {
        const auto &e = m.element(ei);
        M2d dpsi(M2d::Zero());
        if ((etype == EnergyType::Full) || (etype == EnergyType::     MetricFitting)) dpsi += m_fitEnergy[ei].denergy();
        if ((etype == EnergyType::Full) || (etype == EnergyType::CollapsePrevention)) dpsi += collapsePreventionWeight * m_collapsePrevention[ei].denergy().cast<double>();
        const auto &gradLambda = e->gradBarycentric();
        const auto &J = m_J.at(ei);

        for (const auto &n_j : e.nodes()) {
            M32d grad_phi_j(M32d::Zero());
            for (size_t c_j = 0; c_j < 3; ++c_j) {
                grad_phi_j.row(c_j) = gradLambda.col(n_j.localIndex()).transpose();
                M2d symm_term = grad_phi_j.transpose() * J;
                symmetrize(symm_term);
                symm_term *= 2;
                M2d delta_denergy(M2d::Zero());
                if (           fittingActive) delta_denergy += m_fitEnergy[ei].delta_denergy(symm_term);
                if (collapsePreventionActive) delta_denergy += collapsePreventionWeight * m_collapsePrevention[ei].delta_denergy(symm_term).cast<double>();
                M3d contrib = (2 * e->volume()) * ((grad_phi_j * dpsi + J * delta_denergy) * gradLambda);
                grad_phi_j.row(c_j).setZero();

                size_t hint = 0;
                for (const auto &n_i : e.nodes()) {
                    for (size_t c_i = 0; c_i < 3; ++c_i) {
                        size_t var_i = 3 * n_i.index() + c_i,
                               var_j = 3 * n_j.index() + c_j;
                        if (var_i > var_j) continue;
                        hint = Hout.addNZ(var_i, var_j, contrib(c_i, n_i.localIndex()), hint);
                    }
                }
            }
        }
    };

    // Bending energy has a per-hinge contribution
    auto perHingeContrib = [&](const size_t hi, SuiteSparseMatrix &Hout) {
        auto he = m.halfEdge(m_halfedgeForHinge[hi]);
        auto stencil = bendingHingeStencil(he);
        auto hessHingeEnergy = m_edgeHinges[hi].hessian();

        for (size_t svj = 0; svj < 4; ++svj) {
            for (size_t c_j = 0; c_j < 3; ++c_j) {
                size_t hint = 0;
                for (size_t svi = 0; svi < 4; ++svi) {
                    for (size_t c_i = 0; c_i < 3; ++c_i) {
                        size_t var_i = 3 * stencil[svi] + c_i,
                               var_j = 3 * stencil[svj] + c_j;
                        if (var_i > var_j) continue;
                        hint = Hout.addNZ(var_i, var_j, bendingStiffness * hessHingeEnergy(3 * svi + c_i, 3 * svj + c_j), hint);
                    }
                }
            }
        }
    };

    assemble_parallel(perTriContrib, (fittingActive || collapsePreventionActive) ? m.numTris() : 0,
                      perHingeContrib, ((etype == EnergyType::Full) || (etype == EnergyType::Bending)) ? m_halfedgeForHinge.size() : 0,
                      H, "Metric fitting Hessian", "Bending energy Hessian");
}

SuiteSparseMatrix MetricFitter::hessianSparsityPattern(Real val) const {
    TripletMatrix<> Hsp(numVars(), numVars());
    Hsp.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;

    const auto &m = mesh();

    // Sparsity pattern contributions from metric fitting energy
    for (const auto &e : m.elements()) {
        for (const auto &n_i : e.nodes()) {
            for (const auto &n_j : e.nodes()) {
                for (size_t c_i = 0; c_i < 3; ++c_i) {
                    for (size_t c_j = 0; c_j < 3; ++c_j) {
                        size_t var_i = 3 * n_i.index() + c_i,
                               var_j = 3 * n_j.index() + c_j;
                        if (var_i > var_j) continue;
                        Hsp.addNZ(var_i, var_j, 1.0);
                    }
                }
            }
        }
    }

    // Sparsity pattern contributions from bending energy
    for (const auto &he : m.halfEdges()) {
        if (!he.isPrimary() || he.isBoundary()) continue;
        // Add (upper tri) interactions between vertices across a common edge.
        int vtx_i = he           .next().tip().index(),
            vtx_j = he.opposite().next().tip().index();
        if (vtx_i > vtx_j) std::swap(vtx_i, vtx_j);
        for (size_t c_i = 0; c_i < 3; ++c_i)
            for (size_t c_j = 0; c_j < 3; ++c_j)
                Hsp.addNZ(3 * vtx_i + c_i, 3 * vtx_j + c_j, 1.0);
    }

    SuiteSparseMatrix Hsp_csc(std::move(Hsp));
    Hsp_csc.fill(val);
    return Hsp_csc;
}

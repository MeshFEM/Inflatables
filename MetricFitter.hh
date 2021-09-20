////////////////////////////////////////////////////////////////////////////////
// MetricFitter.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Compute an immersion of a triangle mesh with a requested metric by fitting
//  the first fundamental form at each triangle:
//      min_f 1/2 int ||J^T J - g||_F^2 dA
//  where f: M -> R^2 is the unknown immersion and J = grad f.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  05/23/2019 20:51:01
////////////////////////////////////////////////////////////////////////////////
#ifndef METRICFITTER_HH
#define METRICFITTER_HH
#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/SparseMatrices.hh>

#include "BendingEnergy.hh"
#include "MetricFittingEnergy.hh"
#include "CollapsePreventionEnergy.hh"

#include <array>

struct MetricFitter {
    enum class EnergyType { Full, MetricFitting, Bending, Gravitational, CollapsePrevention };

    using Mesh = FEMMesh<2, 1, Vector2D>; // Piecewise linear triangle mesh embedded in R^2

    using  V2d = Eigen::Vector2d;
    using  V3d = Eigen::Vector3d;
    using  V4d = Eigen::Vector4d;
    using  VXd = Eigen::VectorXd;
    using  M2d = Eigen::Matrix2d;
    using  M3d = Eigen::Matrix3d;
    using M23d = Eigen::Matrix<Real, 2, 3>;
    using M32d = Eigen::Matrix<Real, 3, 2>;
    using MX3d = Eigen::Matrix<Real, Eigen::Dynamic, 3>;
    using  M4d = Eigen::Matrix4d;

    MetricFitter(const std::shared_ptr<Mesh> &mesh)
        : m_mesh(mesh), m_fitEnergy(mesh->numElements()), m_collapsePrevention(mesh->numElements()) {
        m_constructHinges();
        setBendingReferenceImmersion(getIdentityImmersion());
        setIdentityTargetMetric();
        setIdentityImmersion();
    }

    // Let the collapse prevention kick in when the element is compressed to
    // 1/4 the area requested by the metric
    void setTargetMetric(const std::vector<M2d> &metric, double relativeCollapsePreventionThreshold = 0.25) {
        const size_t ne = mesh().numElements();
        if (metric.size() != ne) throw std::runtime_error("Incorrect metric tensor field size");

        for (size_t ei = 0; ei < ne; ++ei)
            m_fitEnergy[ei].targetMetric = metric[ei];

        // (Cauchy deformation gradient det relativeCollapsePreventionThreshold^2 times the target metric determinant).
        double relDetThreshold = relativeCollapsePreventionThreshold * relativeCollapsePreventionThreshold;
        for (size_t ei = 0; ei < ne; ++ei)
            m_collapsePrevention[ei].setActivationThreshold(relDetThreshold * metric[ei].determinant());
    }

    void setCurrentMetricAsTarget(double relativeCollapsePreventionThreshold = 0.25) {
        const size_t ne = mesh().numElements();
        std::vector<M2d> metric(ne);
        for (size_t ei = 0; ei < ne; ++ei)
            metric[ei] = m_fitEnergy[ei].currMetric();
        setTargetMetric(metric, relativeCollapsePreventionThreshold);
    }

    size_t numVars() const { return 3 * mesh().numVertices(); }
    const VXd &getVars() const { return m_currVars; }
    void setVars(const Eigen::Ref<const VXd> &vars);

    // Initialize the immersed vertex positions to some rigid transformation of "P"
    void setImmersion(Eigen::Matrix3Xd P); // Copy of P modified inside.

    Eigen::Matrix3Xd getImmersion() const {
        const size_t nv = mesh().numVertices();
        Eigen::Matrix3Xd result(3, nv);
        for (size_t vi = 0; vi < nv; ++vi)
            result.col(vi) = m_currVars.segment<3>(3 * vi);

        return result;
    }

    Eigen::Matrix3Xd getIdentityImmersion() {
        Eigen::Matrix3Xd P(3, mesh().numVertices());
        for (const auto &v : mesh().vertices())
            P.col(v.index()) = padTo3D(v.node()->p);
        return P;
    }
    void setIdentityImmersion() { setImmersion(getIdentityImmersion()); }

    void setIdentityTargetMetric() {
        setTargetMetric(std::vector<M2d>(mesh().numElements(), M2d::Identity()));
    }

    // Set the current immersed surface as the reference configuration for the
    // bending energy.
    void setBendingReferenceImmersion(const Eigen::Matrix3Xd &P);

    const std::array<size_t, 6> &rigidMotionPinVars() const { return m_rigidMotionPinVars; }

    double energy(EnergyType etype = EnergyType::Full) const;
    VXd  gradient(EnergyType etype = EnergyType::Full) const;

    SuiteSparseMatrix hessianSparsityPattern(Real val = 0.0) const;

    void              hessian(SuiteSparseMatrix &H, EnergyType etype = EnergyType::Full) const;
    SuiteSparseMatrix hessian(                      EnergyType etype = EnergyType::Full) const {
        SuiteSparseMatrix H = hessianSparsityPattern();
        hessian(H, etype);
        return H;
    }

          Mesh &mesh()       { return *m_mesh; }
    const Mesh &mesh() const { return *m_mesh; }

    size_t numHinges() const { return m_edgeHinges.size(); }
    const HingeEnergy<double> &hinge(size_t hingeIdx) const {
        return m_edgeHinges.at(hingeIdx);
    }

    const CollapsePreventionEnergyDet &collapsePreventer(size_t idx) const {
        return m_collapsePrevention.at(idx);
    }

    auto deformedVtxPos(size_t vi) const {
        return m_currVars.segment<3>(3 * vi);
    }
    M3d getDeformedTriCornerPositions(size_t ti) const {
        M3d out;
        const auto &tri = mesh().element(ti);
        for (const auto &v : tri.vertices())
            out.col(v.localIndex()) = deformedVtxPos(v.index());
        return out;
    }

    VXd metricDistSq() const {
        const auto &m = mesh();
        VXd result(m.numElements());
        for (const auto &e : m.elements())
            result[e.index()] = 2.0 * m_fitEnergy[e.index()].energy();
        return result;
    }

    // Access the mesh shared pointer from this instance
    std::shared_ptr<Mesh> meshPtr() { return m_mesh; }

    double bendingStiffness = 0.0;
    double collapsePreventionWeight = 0.0;

    Vector3D gravityVector = Vector3D::Zero(); // Gravity direction and magnitude

    // (adjacent vertex, hinge index) for each hinge edge incident vertex "vi"
    // Could be made O(1) if we had a table holding the hinge for each half-edge.
    std::vector<std::pair<size_t, size_t>> getIncidentHinges(size_t vi) const {
        std::vector<std::pair<size_t, size_t>> result;
        const size_t nh = numHinges();
        for (size_t hi = 0; hi < nh; ++hi) {
            auto he = mesh().halfEdge(m_halfedgeForHinge[hi]);
            if ((size_t(he.tip().index()) == vi))
                result.push_back({he.tail().index(), hi});
            if ((size_t(he.tail().index()) == vi))
                result.push_back({he.tip().index(), hi});
        }
        return result;
    }

private:
    std::shared_ptr<Mesh> m_mesh;

    void m_constructHinges() {
        m_halfedgeForHinge.clear();
        for (const auto &he : mesh().halfEdges()) {
            if (!he.isPrimary() || he.isBoundary()) continue;
            m_halfedgeForHinge.push_back(he.index());
        }
    }

    VXd m_currVars;

    ////////////////////////////////////////////////////////////////////////////
    // Quantities computed from the current deformation
    ////////////////////////////////////////////////////////////////////////////
    // Jacobian for each triangle (mapping from the triangle's 2D tangent space to 3D)
    // in the top sheet (first) and bottom sheet (after)
    std::vector<M32d> m_J;
    std::vector<M2d>  m_targetMetric;
    std::vector<MetricFittingEnergy> m_fitEnergy;
    std::vector<CollapsePreventionEnergyDet> m_collapsePrevention;

    // Data for hinges at each interior edge
    std::vector<int> m_halfedgeForHinge;
    std::vector<HingeEnergy<double>> m_edgeHinges;

    std::array<size_t, 6> m_rigidMotionPinVars;
};

#endif /* end of include guard: METRICFITTER_HH */

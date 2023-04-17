////////////////////////////////////////////////////////////////////////////////
// CollapseBarrier.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Provide energy, gradient, and Hessian for the integral of the
//  CollapsePreventionEnergy density over a mesh.
//  This class assumes the mapping analyzed maps into R^2.
//  We apply the collapse barrier to the Jacobian (deformation gradient)
//  instead of the Cauchy deformation gradient so that inverted elements are
//  assigned infinite energy.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  07/23/2019 13:27:30
////////////////////////////////////////////////////////////////////////////////
#ifndef COLLAPSEBARRIER_HH
#define COLLAPSEBARRIER_HH

#include "CollapsePreventionEnergy.hh"
#include <MeshFEM/Parallelism.hh>
#include <MeshFEM/ParallelAssembly.hh>

template<class CollapsePreventionEnergy = CollapsePreventionEnergyDet>
struct CollapseBarrier {
    using Mesh = InflatableSheet::Mesh;
    using Real = InflatableSheet::Real;

    using  M2d = InflatableSheet::M2d;
    using  M3d = InflatableSheet::M3d;
    using MX2d = Eigen::Matrix<Real, Eigen::Dynamic, 2>;
    using M23d = InflatableSheet::M23d;

    using CPE = CollapsePreventionEnergy;

    // Let the collapse prevention kick in when the element is compressed to
    // activationThreshold * its area in the original rest configuration.
    // Note: this class stores a *copy* of the mesh so that the mapping is always
    // from the original rest configuration (not the current rest configuration, which
    // has been modified by the optimization).
    CollapseBarrier(const Mesh &mesh, Real detActivationThreshold)
        : m_mesh(mesh)
    {
        const size_t ne = mesh.numElements();
        m_cpe.resize(ne);
        for (size_t ei = 0; ei < ne; ++ei)
            m_cpe[ei].setActivationThreshold(detActivationThreshold);

        m_J.resize(ne);
    }

    const Mesh &mesh() const { return m_mesh; }

    void setPositions(Eigen::Ref<const MX2d> P) {
        const auto &m = mesh();
        parallel_for_range(m.numElements(), [&](size_t ei) {
            auto &J = m_J[ei];
            const auto e = m.element(ei);
            M23d triCornerPositions;
            for (const auto &v : e.vertices())
                triCornerPositions.col(v.localIndex()) = P.row(v.index()).transpose();
            J = triCornerPositions * e->gradBarycentric().transpose().template leftCols<2>().template cast<Real>(); // map from z=0 tangent plane to 2D
            if (e->gradBarycentric().row(2).squaredNorm() > 1e-16) throw std::runtime_error("Only planar meshes parallel to the xy plane are supported.");
            m_cpe[ei].setMatrix(J);
        });
    }

    Real energy() const {
        return summation_parallel([this](size_t ei) {
                return m_cpe[ei].energy() * mesh().element(ei)->volume(); }, mesh().numElements());
    }

    const CPE &collapsePreventionEnergy(size_t ti) const {
        return m_cpe.at(ti);
    }

    template<typename Derived>
    void accumulateGradient(Eigen::MatrixBase<Derived> &gradRestPositions) const {
        static_assert(Derived::ColsAtCompileTime == 2, "Expected an X by 2 matrix.");
        for (const auto &tri : mesh().elements()) {
            const auto &gradLambdas = tri->gradBarycentric().template topRows<2>();
            // denergy : (delta_v otimes gradLambda[v])
            // = delta_v^T denergy gradLambda[v]
            M2d denergy = m_cpe[tri.index()].denergy();
            M23d grad_vtxs = (tri->volume() * denergy) * gradLambdas.template cast<Real>();
            for (const auto &v : tri.vertices())
                gradRestPositions.row(v.index()) += grad_vtxs.col(v.localIndex()).transpose();
        }
    }

    // varIdx: (vertex index, component index) => global var index
    void accumulateHessian(SuiteSparseMatrix &H, const std::function<size_t(size_t, size_t)> &varIdx) const {
        for (const auto &tri : mesh().elements()) {
            const auto &cpe         = m_cpe[tri.index()];
            const auto gradLambdas = tri->gradBarycentric().template topRows<2>().template cast<Real>().eval();

            // grad_va = (tri->volume() * denergy) * gradLambdas
            // delta grad_va = tri->volume() * delta_denergy * gradLambdas
            for (size_t comp_b = 0; comp_b < 2; ++comp_b) {
                for (const auto &vb : tri.vertices()) {
                    size_t b = varIdx(vb.index(), comp_b);
                    M2d dJ_b = M2d::Zero();
                    dJ_b.row(comp_b) = gradLambdas.col(vb.localIndex()).transpose();
                    M23d delta_grad_va = tri->volume() * cpe.delta_denergy(dJ_b) * gradLambdas;

                    size_t hint = std::numeric_limits<int>::max(); // not size_t max which would be come -1 on cast to int!
                    for (size_t comp_a = 0; comp_a < 2; ++comp_a) {
                        for (const auto &va : tri.vertices()) {
                            size_t a = varIdx(va.index(), comp_a);
                            if (a > b) continue;
                            hint = H.addNZ(a, b, delta_grad_va(comp_a, va.localIndex()), hint);
                        }
                    }
                }
            }
        }
    }

    std::vector<Real> getActivationThresholds      () const { return m_collectCPEProperties([](const CollapsePreventionEnergy &cpe) { return cpe.activationThreshold();    }); }
    std::vector<bool> getApplyStretchBarriers      () const { return m_collectCPEProperties([](const CollapsePreventionEnergy &cpe) { return cpe.applyStretchBarrier;      }); }
    std::vector<Real> getStretchBarrierActiviations() const { return m_collectCPEProperties([](const CollapsePreventionEnergy &cpe) { return cpe.stretchBarrierActivation; }); }
    std::vector<Real> getStretchBarrierLimits      () const { return m_collectCPEProperties([](const CollapsePreventionEnergy &cpe) { return cpe.stretchBarrierLimit;      }); }

    void setActivationThresholds      (const std::vector<Real> &vals) { m_setCPEProperties([](CollapsePreventionEnergy &cpe, Real val) { cpe.setActivationThreshold(val);    }, vals); }
    void setApplyStretchBarriers      (const std::vector<bool> &vals) { m_setCPEProperties([](CollapsePreventionEnergy &cpe, bool val) { cpe.applyStretchBarrier      = val; }, vals); }
    void setStretchBarrierActiviations(const std::vector<Real> &vals) { m_setCPEProperties([](CollapsePreventionEnergy &cpe, Real val) { cpe.stretchBarrierActivation = val; }, vals); }
    void setStretchBarrierLimits      (const std::vector<Real> &vals) { m_setCPEProperties([](CollapsePreventionEnergy &cpe, Real val) { cpe.stretchBarrierLimit      = val; }, vals); }

    ////////////////////////////////////////////////////////////////////////////
    // Serialization support for pickling
    ////////////////////////////////////////////////////////////////////////////
    using StateBackwardsCompat = std::tuple<Mesh, std::vector<Real>, aligned_std_vector<M2d>>; // before stretch barrier was added
    using State                = std::tuple<Mesh, std::vector<Real>, std::vector<bool>, std::vector<Real>, std::vector<Real>, aligned_std_vector<M2d>>; // before stretch barrier was added

    static State serialize(const CollapseBarrier &cb) {
        return std::make_tuple(cb.mesh(), cb.getActivationThresholds(), cb.getApplyStretchBarriers(), cb.getStretchBarrierActiviations(), cb.getStretchBarrierLimits(), cb.m_J);
    }

    static std::shared_ptr<CollapseBarrier> deserialize(const State &state) {
        auto cb = std::make_shared<CollapseBarrier>(std::get<0>(state), 1.0);
        cb->setActivationThresholds(std::get<1>(state));
        if (std::get<2>(state).size()) cb->setApplyStretchBarriers      (std::get<2>(state));
        if (std::get<3>(state).size()) cb->setStretchBarrierActiviations(std::get<3>(state));
        if (std::get<4>(state).size()) cb->setStretchBarrierLimits      (std::get<4>(state));
        cb->m_J = std::get<5>(state);

        auto &cpe = cb->m_cpe;
        const auto &J = cb->m_J;
        for (size_t i = 0; i < cpe.size(); ++i)
            cpe[i].setMatrix(J[i]);

        return cb;
    }

    static std::shared_ptr<CollapseBarrier> deserialize(const StateBackwardsCompat &state) {
        return deserialize(std::make_tuple(std::get<0>(state), std::get<1>(state), std::vector<bool>(), std::vector<Real>(), std::vector<Real>(), std::get<2>(state)));
    }

    std::shared_ptr<CollapseBarrier> clone() const { return deserialize(serialize(*this)); }

private:
    aligned_std_vector<CollapsePreventionEnergy> m_cpe;
    aligned_std_vector<M2d>                      m_J;
    Mesh                                         m_mesh; // COPY of the mesh, since we need to consider mappings from the initial rest mesh.

    template<class F>
    auto m_collectCPEProperties(const F &f) const {
        const size_t ne = mesh().numElements();
        std::vector<return_type<F>> result;
        result.reserve(ne);
        for (const auto &cpe : m_cpe)
            result.push_back(f(cpe));
        return result;
    }

    template<class F, class Values>
    auto m_setCPEProperties(const F &property_setter, const Values &vals) {
        if (vals.size() != m_cpe.size()) throw std::runtime_error("Size mismatch");
        for (size_t i = 0; i < m_cpe.size(); ++i)
            property_setter(m_cpe[i], vals[i]);
    }

};

#endif /* end of include guard: COLLAPSEBARRIER_HH */

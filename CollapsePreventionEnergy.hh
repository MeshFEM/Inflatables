////////////////////////////////////////////////////////////////////////////////
// CollapsePreventionEnergy.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Sheet material energy density that prevents elements from collapsing into
//  degenerate configurations (which will break the bending energy...) with
//  an infinite energy barrier:
//      (-log((det(C) - activationThreshold) / activationThreshold + 1))^2
//  for det(C) < activationThreshold, 0 otherwise
//
//  This energy term is C1. We could make it C2 (to avoid the single point
//  where the Hessian is undefined) by raising the power from 2 to 3--at the
//  expense of a faster ramp-up (greater nonlinearity).
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  05/30/2019 17:23:18
////////////////////////////////////////////////////////////////////////////////
#ifndef COLLAPSEPREVENTIONENERGY_HH
#define COLLAPSEPREVENTIONENERGY_HH
#include <cmath>

#include <Eigen/Dense>
#include <MeshFEM/EnergyDensities/Tensor.hh>
#include "SVDSensitivity.hh"
#include "InflatableSheet.hh"

struct BarrierFuncLogSq {
    using Real = InflatableSheet::Real;

    constexpr static Real inf = std::numeric_limits<double>::infinity();

    static Real   b(Real x) { if (x <= 0) return inf; if (x >= 1.0) return 0.0; return 0.5 * std::pow(log(x), 2); }
    static Real  db(Real x) { if (x <= 0) return inf; if (x >= 1.0) return 0.0; return log(x) / x; }
    static Real d2b(Real x) { if (x <= 0) return inf; if (x >= 1.0) return 0.0; return (1 - log(x)) / (x * x); }
};

template<class BarrierFunc>
struct NormalizedBarrierFunction {
    using Real = typename BarrierFunc::Real;
    using BF = BarrierFunc;

    void setActivationThreshold(Real val) { m_a = val; }
    Real activationThreshold() const { return m_a; }

    Real   b(Real x) const { return BF::  b(x / m_a); }
    Real  db(Real x) const { return BF:: db(x / m_a) / m_a; }
    Real d2b(Real x) const { return BF::d2b(x / m_a) / (m_a * m_a); }

protected:
    Real m_a = 1.0;
};

template<class BarrierFunc>
struct CollapsePreventionDet : public NormalizedBarrierFunction<BarrierFunc> {
    using BF = NormalizedBarrierFunction<BarrierFunc>;
    using M2d  = InflatableSheet::M2d;
    using Real = InflatableSheet::Real;

    template<typename Derived>
    void setMatrix(const Eigen::MatrixBase<Derived> &C) {
        static_assert((Derived::RowsAtCompileTime == 2) && (Derived::ColsAtCompileTime == 2), "Only 2x2 supported for now");
        m_det = C.determinant();
        m_grad_det <<  C(1, 1), -C(1, 0),
                      -C(0, 1),  C(0, 0);
    }

    Real energy() const { return BF::b(m_det); }
    M2d denergy() const { return BF::db(m_det) * m_grad_det; }

    template<typename Derived>
    M2d delta_denergy(const Eigen::MatrixBase<Derived> &dC) const {
        static_assert((Derived::RowsAtCompileTime == 2) && (Derived::ColsAtCompileTime == 2), "Only 2x2 supported for now");

        M2d delta_grad_det;
        delta_grad_det <<  dC(1, 1), -dC(1, 0),
                          -dC(0, 1),  dC(0, 0);

        return ((BF::d2b(m_det) * doubleContract(m_grad_det, dC.template cast<Real>()))) * m_grad_det
               + BF:: db(m_det) * delta_grad_det;
    }

    // For debugging scalar function of det + its derivatives
    void setDet(Real det) { m_det = det; }
    Real det() const { return m_det; }
    Real normalizedDet()  const { return m_det / BF::m_a; }
    Real denergy_ddet()   const { return BF::db(m_det);  }
    Real d2energy_d2det() const { return BF::d2b(m_det); }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    Real m_det;
    M2d m_grad_det;
};

template<class BarrierFunc>
struct CollapsePreventionSingularValues : public NormalizedBarrierFunction<BarrierFunc> {
    using BFRaw = BarrierFunc;
    using BF    = NormalizedBarrierFunction<BarrierFunc>;
    using V2d   = InflatableSheet::V2d;
    using M2d   = InflatableSheet::M2d;
    using Real  = InflatableSheet::Real;

    // The activation threshold for the singular value barriers should be the
    // square root of the area barrier activation threshold.
    void setActivationThreshold(Real val) { BF::setActivationThreshold(std::sqrt(val)); }
    Real activationThreshold() const { return std::pow(BF::activationThreshold(), 2); }

    template<typename Derived>
    void setMatrix(const Eigen::MatrixBase<Derived> &F) { m_svd.setMatrix(F); m_det = F.determinant(); }

    Real energy() const {
        if (m_det < 0.0) return std::numeric_limits<double>::infinity();
        Real result = BF::b(m_svd.sigma(0)) + BF::b(m_svd.sigma(1));
        if (applyStretchBarrier) {
            Real scale = 1.0 / (stretchBarrierLimit - stretchBarrierActivation);
            result += BFRaw::b(scale * (stretchBarrierLimit - m_svd.sigma(0)))
                   +  BFRaw::b(scale * (stretchBarrierLimit - m_svd.sigma(1)));
        }
        return result;
    }

    M2d denergy() const {
        if (m_det < 0.0) {
            M2d result;
            result.setConstant(std::numeric_limits<double>::infinity());
        }
        V2d dE_dsigma(BF::db(m_svd.sigma(0)),
                      BF::db(m_svd.sigma(1)));
        if (applyStretchBarrier) {
            Real scale = 1.0 / (stretchBarrierLimit - stretchBarrierActivation);
            dE_dsigma -= scale * V2d(BFRaw::db(scale * (stretchBarrierLimit - m_svd.sigma(0))),
                                     BFRaw::db(scale * (stretchBarrierLimit - m_svd.sigma(1))));
        }

        return m_svd.U() * (dE_dsigma.asDiagonal() * m_svd.V().transpose());
    }

    const SVDSensitivity &svd() const { return m_svd; }
    Real det() const { return m_det; }

    // Second derivatives blow up when sigma_0 == sigma_1!!!
    template<typename Derived>
    M2d delta_denergy(const Eigen::MatrixBase<Derived> &/* dF */) const {
        static_assert((Derived::RowsAtCompileTime == 2) && (Derived::ColsAtCompileTime == 2), "Only 2x2 supported for now");
        throw std::runtime_error("Second derivative of SVD collapse prevention unsupported; will blow up at sigma_0 == sigma_2.");
    #if 0
        // SVDSensitivity doesn't yet implement delta_dsigma...
        return (BF::d2b(m_svd.sigma(0)) * m_svd.dsigma(0, dF)) * m_svd.dsigma(0) +
               (BF::d2b(m_svd.sigma(1)) * m_svd.dsigma(1, dF)) * m_svd.dsigma(1) +
                BF:: db(m_svd.sigma(0)) * m_svd.delta_dsigma(0, dF) +
                BF:: db(m_svd.sigma(1)) * m_svd.delta_dsigma(1, dF);
    #endif
    }

    bool applyStretchBarrier = false;
    Real stretchBarrierActivation = 1.75; // threshold below which barrier term is smoothly deactivated
    Real stretchBarrierLimit      = 2.25; // placement of the infinite barrier

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    SVDSensitivity m_svd;
    Real m_det;
};

using CollapsePreventionEnergyDet = CollapsePreventionDet<BarrierFuncLogSq>;
using CollapsePreventionEnergySV  = CollapsePreventionSingularValues<BarrierFuncLogSq>;

#endif /* end of include guard: COLLAPSEPREVENTIONENERGY_HH */

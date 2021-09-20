////////////////////////////////////////////////////////////////////////////////
// IncompressibleBalloonEnergy.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implementation of the incompressible neo-Hookean-based strain energy
//  density used in Skouras 2014: Designing Inflatable Structures (before
//  homogenizing away the wrinkles using a relaxed energy density).
//
//  This energy is implemented as a function of the right Green-Green
//  deformation tensor.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  04/04/2019 18:19:10
////////////////////////////////////////////////////////////////////////////////
#ifndef INCOMPRESSIBLEBALLOONENERGY_HH
#define INCOMPRESSIBLEBALLOONENERGY_HH

#include <Eigen/Dense>
#include <array>
#include <MeshFEM/EnergyDensities/Tensor.hh>

template<typename Real>
struct IncompressibleBalloonEnergy {
    using V2d  = Eigen::Matrix<Real, 2, 1>;
    using M2d  = Eigen::Matrix<Real, 2, 2>;

    IncompressibleBalloonEnergy() { }

    template<typename Derived>
    IncompressibleBalloonEnergy(const Eigen::MatrixBase<Derived> &C) { setMatrix(C); }

    template<typename Derived>
    void setMatrix(const Eigen::MatrixBase<Derived> &C) {
        static_assert((Derived::RowsAtCompileTime == 2) && (Derived::ColsAtCompileTime == 2), "Only 2x2 supported for now");

        Real a = C(0, 0),
             b = C(0, 1),
             c = C(1, 1);
        if (std::abs(b - C(1, 0)) > 1e-15) throw std::runtime_error("Asymmetric matrix");

        m_C = C;
        m_trace_C = C.trace();
        m_det_C = a * c - b * b;
        m_grad_det_C <<  c, -b,
                        -b,  a;
    }

    Real energy() const {
        return stiffness * (m_trace_C + 1.0 / m_det_C - 3.0);
    }

    Real denergy(const M2d &dC) const {
        return stiffness * (dC.trace() - (1.0 / (m_det_C * m_det_C)) * doubleContract(m_grad_det_C, dC));
    }

    M2d denergy() const {
        return stiffness * (M2d::Identity() - (1.0 / (m_det_C * m_det_C)) * m_grad_det_C);
    }

    Real d2energy(const M2d &dC_a, const M2d &dC_b) const {
        return stiffness * ((2.0 / (m_det_C * m_det_C * m_det_C)) * doubleContract(m_grad_det_C, dC_a) * doubleContract(m_grad_det_C, dC_b)
                          - (1.0 / (m_det_C * m_det_C)) * (dC_a(1, 1) * dC_b(0, 0) + dC_a(0, 0) * dC_b(1, 1) - 2 * dC_a(0, 1) * dC_b(0, 1)));
    }

    M2d delta_denergy(const M2d &dC) const {
        M2d adj_dC;
        adj_dC << dC(1, 1), -dC(0, 1),
                 -dC(1, 0),  dC(0, 0);
        return (((2.0 * stiffness / (m_det_C * m_det_C * m_det_C)) * doubleContract(m_grad_det_C, dC)) * m_grad_det_C
                     - (stiffness / (m_det_C * m_det_C))                                               * adj_dC);
    }

    // Second derivatives evaluated at the reference configuration
    M2d delta_denergy_undeformed(const M2d &dC) const {
        return stiffness * (dC.trace() * M2d::Identity() + dC);
    }

    Real stiffness = 1.0;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    Real m_trace_C, m_det_C;
    M2d m_grad_det_C;
    M2d m_C;
};

#endif /* end of include guard: INCOMPRESSIBLEBALLOONENERGY_HH */

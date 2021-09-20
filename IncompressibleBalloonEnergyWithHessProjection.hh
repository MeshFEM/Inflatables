////////////////////////////////////////////////////////////////////////////////
// IncompressibleBalloonEnergyWithHessProjection.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implementation of the incompressible neo-Hookean-based strain energy
//  density used in Skouras 2014: Designing Inflatable Structures, expressed
//  in terms of the 3x2 deformation gradient F instead of the 2x2 Cauchy-Green
//  deformation tensor F^T F. We also provide analytical expressions for the
//  eigenvalues and eigenvectors of the Hessian (wrt F), enabling Hessian
//  projection to resolve indefiniteness.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  07/09/2019 10:56:53
////////////////////////////////////////////////////////////////////////////////
#ifndef INCOMPRESSIBLEBALLOONENERGYWITHHESSPROJECTION_HH
#define INCOMPRESSIBLEBALLOONENERGYWITHHESSPROJECTION_HH

#include <Eigen/Dense>
#include <array>

template<typename Real>
struct IncompressibleBalloonEnergyWithHessProjection {
    using V2d  = Eigen::Matrix<Real, 2, 1>;
    using M2d  = Eigen::Matrix<Real, 2, 2>;
    using M3d  = Eigen::Matrix<Real, 3, 3>;
    using M32d = Eigen::Matrix<Real, 3, 2>;

    IncompressibleBalloonEnergyWithHessProjection() { }

    template<typename Derived>
    IncompressibleBalloonEnergyWithHessProjection(const Eigen::MatrixBase<Derived> &F) { setF(F); }

    template<typename Derived>
    void setF(const Eigen::MatrixBase<Derived> &F) {
        static_assert((Derived::RowsAtCompileTime == 3) && (Derived::ColsAtCompileTime == 2), "F must be 3x2");
        m_F = F.template cast<Real>();

        Eigen::JacobiSVD<M32d> svd(m_F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        m_sigma = svd.singularValues();
        m_V = svd.matrixV();
        m_U = svd.matrixU();

        m_det32_F = m_sigma.prod();
        m_det_C = m_det32_F * m_det32_F;

        m_d_det32_F = (m_U.col(0) * m_sigma[1] * m_V.col(0).transpose() +
                       m_U.col(1) * m_sigma[0] * m_V.col(1).transpose());

        m_trace_C = m_F.squaredNorm();

        m_d_psi_ddet = -2.0 / (m_det_C * m_det32_F);

        // Compute eigendecomposition of det32_F's Hessian
        /* static constexpr */ const Real inv_sqrt2 = 1.0 / std::sqrt(2.0); // std::sqrt is not constexpr!

        // T = U [0 -1; 1 0; 0 0] V^T / sqrt(2)
        m_eigmat[2] = (-inv_sqrt2 * m_U.col(0)) * m_V.col(1).transpose() +
                       (inv_sqrt2 * m_U.col(1)) * m_V.col(0).transpose();
        m_eigval[2] = 2.0 + m_d_psi_ddet;
        // L = U [0 1; 1 0; 0 0] V^T / sqrt(2)
        m_eigmat[3] = (inv_sqrt2 * m_U.col(0)) * m_V.col(1).transpose() +
                      (inv_sqrt2 * m_U.col(1)) * m_V.col(0).transpose();
        m_eigval[3] = 2.0 - m_d_psi_ddet;
        // NX = U [0 0; 0 0; 1 0] V^T
        m_eigmat[4] = m_U.col(2) * m_V.col(0).transpose();
        m_eigval[4] = 2.0 + m_d_psi_ddet * m_sigma[1] / m_sigma[0];
        // NY = U [0 0; 0 0; 0 1] V^T
        m_eigmat[5] = m_U.col(2) * m_V.col(1).transpose();
        m_eigval[5] = 2.0 + m_d_psi_ddet * m_sigma[0] / m_sigma[1];

        // The "R" and "P" modes [1 0; 0 1; 0 0] and [1 0; 0 -1; 0 0]
        // are not orthogonal to d_det32_F and thus will not
        // be eigenmatrices of the energy density Hessian (after addition of the
        // rank 1 term proportional to (d_det32_F otimes d_det32_F)).
        // We need to compute the compute the eigendecomposition of the Hessian
        // in this subspace by solving a 2x2 matrix eigenvalue problem.
        // We first probe the Hessian with an orthonormal basis for this space:
        //      d0 := U [1, 0; 0, 0; 0, 0] V^T and d1 := U [0, 1; 0, 0; 0, 0] V^T
        // to obtain the reduced Hessian:
        //      A_ij = di : d2psi / dF2 : dj
        //  A = 2 I + 1 / det(c)^2 [6 sigma_2^2  4 det32_F  ]
        //                         [4 det32_F    6 sigma_1^2]
        // Whose eigendecomposition is:
        Real e = 3 * (m_sigma[1] * m_sigma[1] - m_sigma[0] * m_sigma[0]),
             x = std::sqrt(16 * m_det_C + e * e);
        Eigen::Vector2d v0(e - x, 4 * m_det32_F),
                        v1(e + x, 4 * m_det32_F);
        v0.normalize();
        v1.normalize();
        m_eigmat[0] = (v0[0] * m_U.col(0)) * m_V.col(0).transpose() +
                      (v0[1] * m_U.col(1)) * m_V.col(1).transpose();
        m_eigmat[1] = (v1[0] * m_U.col(0)) * m_V.col(0).transpose() +
                      (v1[1] * m_U.col(1)) * m_V.col(1).transpose();
        m_eigval[0] = 2 + (3 * m_trace_C - x) / (m_det_C * m_det_C);
        m_eigval[1] = 2 + (3 * m_trace_C + x) / (m_det_C * m_det_C);
    }

    Real energy() const {
        return stiffness * (m_trace_C + 1.0 / m_det_C - 3.0);
    }

    M32d denergy() const {
        return 2 * m_F + m_d_psi_ddet * m_d_det32_F;
    }

    template<class DeltaF>
    Real denergy(const DeltaF &dF) const {
        const Real d_det32_F = m_sigma[1] * m_U.col(0).dot(dF * m_V.col(0)) +
                               m_sigma[0] * m_U.col(1).dot(dF * m_V.col(1));

        return 2.0 * doubleContract(m_F, dF) + m_d_psi_ddet * d_det32_F;
    }

    template<class DeltaF>
    M32d delta_denergy(const DeltaF &dF) const {
        M32d result(M32d::Zero());
        for (size_t i = 0; i < 6; ++i) {
            if (applyHessianProjection && (m_eigval[i] <= 0)) continue;
            result += (m_eigval[i] * doubleContract(m_eigmat[i], dF)) * m_eigmat[i];
        }

        return result;
    }

    template<class DeltaF>
    Real d2energy(const DeltaF &dF_a, const DeltaF &dF_b) const {
        return doubleContract(delta_denergy(dF_a), dF_b);
    }

    Real stiffness = 1.0;

    // Whether to project d2energy/denergy onto the space of positive semi-semidefinite tensors.
    bool applyHessianProjection = false;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    // We need the singular value decomposition for our gradient/Hessian formulas.
    M32d m_F, m_d_det32_F;

    // Eigenmatrices of the determinant: "R", "P", "T", "L", "NX", and "NY"
    std::array<M32d, 6> m_eigmat;
    std::array<Real, 6> m_eigval;

    M2d  m_V;
    M3d  m_U;
    V2d  m_sigma;
    Real m_trace_C, m_det32_F, m_det_C, m_d_psi_ddet;
};

#endif /* end of include guard: INCOMPRESSIBLEBALLOONENERGYWITHHESSPROJECTION_HH */

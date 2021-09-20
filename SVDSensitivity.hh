#ifndef SVDSENSITIVITY_HH
#define SVDSENSITIVITY_HH
#include <Eigen/Dense>
#include <array>
#include <MeshFEM/EnergyDensities/Tensor.hh>

// 2x2 case only for now, also can be sped up.
struct SVDSensitivity {
    using M2d = Eigen::Matrix2d;
    using V2d = Eigen::Vector2d;

    SVDSensitivity() { }

    template<typename Derived>
    SVDSensitivity(const Eigen::MatrixBase<Derived> &A) { setMatrix(A); }

    template<typename Derived>
    void setMatrix(const Eigen::MatrixBase<Derived> &A) {
        static_assert((Derived::RowsAtCompileTime == 2) && (Derived::ColsAtCompileTime == 2), "Only 2x2 supported for now");
        Eigen::JacobiSVD<M2d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        m_U     = svd.matrixU();
        m_V     = svd.matrixV();
        m_Sigma = svd.singularValues();

        // Cache first derivatives (needed for computing second derivatives)
        m_dSigma[0] = m_U.col(0) * m_V.col(0).transpose();
        m_dSigma[1] = m_U.col(1) * m_V.col(1).transpose();

        const double sigmaSqDiff = m_Sigma[0] * m_Sigma[0] - m_Sigma[1] * m_Sigma[1];
        if (std::abs(sigmaSqDiff) < 1e-15) { m_degenerate =  true; m_invSigmaSqDiff = 0.0; }
        else                               { m_degenerate = false; m_invSigmaSqDiff = 1.0 / sigmaSqDiff; }

        m_y = m_invSigmaSqDiff * (m_Sigma[1] * m_U.col(0) * m_V.col(1).transpose()
                                + m_Sigma[0] * m_U.col(1) * m_V.col(0).transpose());
        m_du0[0] =  m_y * m_U(0, 1);
        m_du0[1] =  m_y * m_U(1, 1);
        m_du1[0] = -m_y * m_U(0, 0);
        m_du1[1] = -m_y * m_U(1, 0);

        m_z = m_invSigmaSqDiff * (m_Sigma[0] * m_U.col(0) * m_V.col(1).transpose()
                                + m_Sigma[1] * m_U.col(1) * m_V.col(0).transpose());
        m_dv0[0] =  m_z * m_V(0, 1);
        m_dv0[1] =  m_z * m_V(1, 1);
        m_dv1[0] = -m_z * m_V(0, 0);
        m_dv1[1] = -m_z * m_V(1, 0);
    }

    // Access SVD
    const M2d &    U() const { return m_U; }
    const M2d &    V() const { return m_V; }
    const V2d &Sigma() const { return m_Sigma; }

    auto       u(size_t i) const { return m_U.col(i); }
    auto       v(size_t i) const { return m_V.col(i); }
    double sigma(size_t i) const { return m_Sigma[i]; }

    ////////////////////////////////////////////////////////////////////////////
    // First derivative expressions
    ////////////////////////////////////////////////////////////////////////////
    M2d dsigma(size_t i) const { return m_dSigma.at(i); }
    M2d du0   (size_t i) const { return m_du0   .at(i); }
    M2d du1   (size_t i) const { return m_du1   .at(i); }
    M2d dv0   (size_t i) const { return m_dv0   .at(i); }
    M2d dv1   (size_t i) const { return m_dv1   .at(i); }

    template<typename M2d_, EnableIfMatrixOfSize<M2d_, 2, 2, int> = 0> V2d dSigma(const M2d_ &dA) const { return V2d(doubleContract(m_dSigma[0], dA), doubleContract(m_dSigma[1], dA)); }
    template<typename M2d_, EnableIfMatrixOfSize<M2d_, 2, 2, int> = 0> V2d du0   (const M2d_ &dA) const { return V2d(doubleContract(m_du0   [0], dA), doubleContract(m_du0   [1], dA)); }
    template<typename M2d_, EnableIfMatrixOfSize<M2d_, 2, 2, int> = 0> V2d du1   (const M2d_ &dA) const { return V2d(doubleContract(m_du1   [0], dA), doubleContract(m_du1   [1], dA)); }
    template<typename M2d_, EnableIfMatrixOfSize<M2d_, 2, 2, int> = 0> V2d dv0   (const M2d_ &dA) const { return V2d(doubleContract(m_dv0   [0], dA), doubleContract(m_dv0   [1], dA)); }
    template<typename M2d_, EnableIfMatrixOfSize<M2d_, 2, 2, int> = 0> V2d dv1   (const M2d_ &dA) const { return V2d(doubleContract(m_dv1   [0], dA), doubleContract(m_dv1   [1], dA)); }

    template<typename M2d_, EnableIfMatrixOfSize<M2d_, 2, 2, int> = 0>
    double dsigma(size_t i, const M2d_ &dA) const { return doubleContract(m_dSigma[i], dA); }

    ////////////////////////////////////////////////////////////////////////////
    // Second derivative expressions
    ////////////////////////////////////////////////////////////////////////////
    // Note, to avoid using high order tensors, we only provide the contraction of the
    // singular value/vector Hessians with perturbation matrices.

    // Second derivative of singular values with respect to variables inducing
    // perturbations dA_1 and dA_2, respectively.
    template<typename M2d_, EnableIfMatrixOfSize<M2d_, 2, 2, int> = 0>
    V2d d2Sigma(const M2d_ &dA_1, const M2d_ &dA_2) const {
        return V2d(du0(dA_2).dot((dA_1 * m_V.col(0)).matrix()) + m_U.col(0).dot((dA_1 * dv0(dA_2)).matrix()),
                   du1(dA_2).dot((dA_1 * m_V.col(1)).matrix()) + m_U.col(1).dot((dA_1 * dv1(dA_2)).matrix()));
    }
    template<typename M2d_, EnableIfMatrixOfSize<M2d_, 2, 2, int> = 0>
    double d2sigma(size_t i, const M2d_ &dA_1, const M2d_ &dA_2) const {
        if (i == 0) return du0(dA_2).dot((dA_1 * m_V.col(0)).matrix()) + m_U.col(0).dot((dA_1 * dv0(dA_2)).matrix());
        if (i == 1) return du1(dA_2).dot((dA_1 * m_V.col(1)).matrix()) + m_U.col(1).dot((dA_1 * dv1(dA_2)).matrix());
        throw std::runtime_error("Index out of bounds");
    }

    // Second derivative of first left singular vector with respect to variables inducing
    // perturbations dA_1 and dA_2, respectively.
    template<typename M2d_, EnableIfMatrixOfSize<M2d_, 2, 2, int> = 0>
    V2d d2u0(const M2d_ &dA_1, const M2d_ &dA_2) const {
        M2d Ut_dA1_V = m_U.transpose() * (dA_1 * m_V);

        V2d d_sigma_d2 = dSigma(dA_2);
        const double y1 = doubleContract(m_y, dA_1);
        const double y2 = doubleContract(m_y, dA_2);
        const double z2 = doubleContract(m_z, dA_2);

        const double dy1_d2 = m_invSigmaSqDiff * (d_sigma_d2[1] * (2 * m_Sigma[1] * y1 + Ut_dA1_V(0, 1)) + d_sigma_d2[0] * (Ut_dA1_V(1, 0) - 2 * m_Sigma[0] * y1)
                                                  + Ut_dA1_V(1, 1) * (m_Sigma[1] * y2 + m_Sigma[0] * z2)
                                                  - Ut_dA1_V(0, 0) * (m_Sigma[0] * y2 + m_Sigma[1] * z2));

        return dy1_d2 * m_U.col(1) - y1 * y2 * m_U.col(0);
    }

    // Second derivative of second right singular vector with respect to variables inducing
    // perturbations dA_1 and dA_2, respectively.
    //      -d z_1 / d2 v0 - z_1 z_2 v1
    // Note: "z" is the same as "y" with sigma_0 and sigma_1 swapped (apart from the m_invSigmaSqDiff factor).
    template<typename M2d_, EnableIfMatrixOfSize<M2d_, 2, 2, int> = 0>
    V2d d2v1(const M2d_ &dA_1, const M2d_ &dA_2) const {
        M2d Ut_dA1_V = m_U.transpose() * (dA_1 * m_V);

        V2d d_sigma_d2 = dSigma(dA_2);
        const double z1 = doubleContract(m_z, dA_1);
        const double z2 = doubleContract(m_z, dA_2);
        const double y2 = doubleContract(m_y, dA_2);

        const double dz1_d2 = m_invSigmaSqDiff * (d_sigma_d2[0] * (Ut_dA1_V(0, 1) - 2 * m_Sigma[0] * z1) + d_sigma_d2[1] * (Ut_dA1_V(1, 0) + 2 * m_Sigma[1] * z1)
                                                  + Ut_dA1_V(1, 1) * (m_Sigma[0] * y2 + m_Sigma[1] * z2)
                                                  - Ut_dA1_V(0, 0) * (m_Sigma[1] * y2 + m_Sigma[0] * z2));

        return -dz1_d2 * m_V.col(0) - z1 * z2 * m_V.col(1);
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    M2d m_U, m_V, m_y, m_z;
    V2d m_Sigma;

    bool m_degenerate;
    double m_invSigmaSqDiff;

    std::array<M2d, 2> m_dSigma;
    std::array<M2d, 2> m_du0, m_du1,
                       m_dv0, m_dv1;
};

#endif /* end of include guard: SVDSENSITIVITY_HH */

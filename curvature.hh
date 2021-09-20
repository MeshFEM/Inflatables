#ifndef CURVATURE_HH
#define CURVATURE_HH

#include <Eigen/Dense>
#include <MeshFEM/Utilities/MeshConversion.hh>

struct CurvatureInfo {
    // Per vertex curvature quantities
    Eigen::MatrixX3d d_1, d_2;
    Eigen::VectorXd kappa_1, kappa_2;

    Eigen::VectorXd     meanCurvature() const { return 0.5 * (kappa_1 + kappa_2); }
    Eigen::VectorXd gaussianCurvature() const { return kappa_1.array() * kappa_2.array(); }

    const Eigen::MatrixX3d &d(size_t i) const {
        if (i == 0) return d_1;
        if (i == 1) return d_2;
        throw std::runtime_error("Index i out of bounds");
    }

    CurvatureInfo(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);

    template<class Mesh>
    CurvatureInfo(const Mesh &m) : CurvatureInfo(getV(m), getF(m)) { }
};

#endif /* end of include guard: CURVATURE_HH */

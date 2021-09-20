#include <igl/principal_curvature.h> // must come first so igl::Triplet doesn't get confused with ::Triplet from MeshFEM

#include "curvature.hh"

CurvatureInfo::CurvatureInfo(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) {
    // Compute curvature directions via quadric fitting
    // Note: the comments in libigl are incorrect; reading the function
    // and example 203 makes it clear that the (algebraically) *minimal*
    // curvature quantities are output first, maximal second.
    igl::principal_curvature(V, F, d_2, d_1, kappa_2, kappa_1);
}

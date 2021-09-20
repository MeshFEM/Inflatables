#include "DualLaplacianStencil.hh"

#include <igl/intrinsic_delaunay_cotmatrix.h>

void igl_intrinsic_delaunay_cotmatrix(const Eigen::MatrixX3d &V,
                                      const Eigen::MatrixXi &F,
                                      Eigen::SparseMatrix<Real> &L) {
    igl::intrinsic_delaunay_cotmatrix(V, F, L);
}

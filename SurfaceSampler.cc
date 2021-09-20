#include "SurfaceSampler.hh"

#include <igl/point_simplex_squared_distance.h>
#include <igl/AABB.h>

struct SurfaceAABB : public igl::AABB<Eigen::MatrixXd, 3> {
    using Base = igl::AABB<Eigen::MatrixXd, 3>;
    using Base::Base;
};

SurfaceSampler::SurfaceSampler(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
    : m_V(V), m_F(F)
{
    m_surfaceAABB = std::make_unique<SurfaceAABB>();
    m_surfaceAABB->init(m_V, m_F);
}

void SurfaceSampler::closestTriAndBaryCoords(const Eigen::MatrixXd &P, Eigen::VectorXi &I, Eigen::MatrixX3d &B) const {
    if (P.cols() != 3) throw std::runtime_error("P must be an X by 3 matrix");
    Eigen::VectorXd dists;
    Eigen::MatrixXd C; // closest points in 3D
    m_surfaceAABB->squared_distance(m_V, m_F, P, dists, I, C);

    const size_t np = P.rows();
    B.resize(np, 3);

    for (size_t i = 0; i < np; ++i) {
        Eigen::RowVector3d pt, baryCoords;
        double dist;
        igl::point_simplex_squared_distance<3>(C.row(i), m_V, m_F, I[i], dist, pt, baryCoords);
        B.row(i) = baryCoords;
    }
}

// Needed because m_surfaceAABB is a smart pointer to an incomplete type.
SurfaceSampler::~SurfaceSampler() { }

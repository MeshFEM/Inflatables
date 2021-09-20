////////////////////////////////////////////////////////////////////////////////
// SurfaceSampler.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Sample from piecewise linear fields on a triangulated surface by
//  evaluating the field at the closest point to the query.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  05/01/2019 23:55:30
////////////////////////////////////////////////////////////////////////////////
#ifndef SURFACE_SAMPLER_HH
#define SURFACE_SAMPLER_HH
#include <memory>
#include <stdexcept>
#include <Eigen/Dense>

// Forward declare AABB data structure
struct SurfaceAABB;

struct SurfaceSampler {
    SurfaceSampler(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);

    // P: (#points x dim) matrix of stacked query point row vectors
    // fieldValues (X x fieldDim) matrix of stacked per-vertex or per-triangle field values
    template<class PLField>
    PLField sample(const Eigen::MatrixXd &P, PLField &fieldValues) const {
        bool isPerTri = fieldValues.rows() == m_F.rows();
        if (!isPerTri && (fieldValues.rows() != m_V.rows())) throw std::runtime_error("Invalid piecewise linear/constant field size");

        Eigen::VectorXi  I;
        Eigen::MatrixX3d B;
        closestTriAndBaryCoords(P, I, B);

        const size_t np = P.rows();
        PLField outSamples(np, fieldValues.cols());
        for (size_t i = 0; i < np; ++i) {
            int tri = I[i];
            if (isPerTri)
                outSamples.row(i) = fieldValues.row(tri);
            else {
                outSamples.row(i) = B(i, 0) * fieldValues.row(m_F(tri, 0))
                                  + B(i, 1) * fieldValues.row(m_F(tri, 1))
                                  + B(i, 2) * fieldValues.row(m_F(tri, 2));
            }
        }

        return outSamples;
    }

    // P: (#points x dim) matrix of stacked query point row vectors
    // I: index of closest triangle for each query point
    // B: (#points x 3) matrix of stacked barycentric coordinate vectors for each query point
    void closestTriAndBaryCoords(const Eigen::MatrixXd &P, Eigen::VectorXi &I, Eigen::MatrixX3d &B) const;

    ~SurfaceSampler(); // Needed because m_surfaceAABB is a smart pointer to an incomplete type.
private:
    std::unique_ptr<SurfaceAABB> m_surfaceAABB;
    Eigen::MatrixXd m_V;
    Eigen::MatrixXi m_F;
};


#endif /* end of include guard: SURFACE_SAMPLER_HH */

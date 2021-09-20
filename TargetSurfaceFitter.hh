////////////////////////////////////////////////////////////////////////////////
// TargetSurfaceFitter.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implementation of a target surface to which points are fit using the
//  distance to their closest point projections.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  01/02/2019 11:51:11
////////////////////////////////////////////////////////////////////////////////
#ifndef TARGETSURFACEFITTER_HH
#define TARGETSURFACEFITTER_HH

#include <MeshFEM/FEMMesh.hh>
#include "InflatableSheet.hh"

// Forward declare AABB data structure
struct TargetSurfaceAABB;
using TargetSurfaceMesh = InflatableSheet::Mesh; // Piecewise linear triangle mesh embedded in R^3

struct BoundaryEdgeFitter {
    using V3d  = InflatableSheet::V3d;
    using Real = InflatableSheet::Real;
    struct BoundaryEdge {
        BoundaryEdge(Eigen::Ref<const V3d> p0, Eigen::Ref<const V3d> p1) {
            e_div_len = p1 - p0;
            Real sqLen = e_div_len.squaredNorm();
            e = e_div_len / std::sqrt(sqLen);

            e_div_len /= sqLen;
            p0_dot_e_div_len = p0.dot(e_div_len);
            endpoints[0] = p0;
            endpoints[1] = p1;
        }

        Real            barycoords(Eigen::Ref<const V3d> q) const { return q.dot(e_div_len) - p0_dot_e_div_len; }
        Real closestEdgeBarycoords(Eigen::Ref<const V3d> q) const { return std::max<Real>(std::min<Real>(barycoords(q), 1.0), 0.0); }

        void closestBarycoordsAndPt(Eigen::Ref<const V3d> q, Real &lambda, Eigen::Ref<V3d> p) {
            lambda = closestEdgeBarycoords(q);
            p = (1.0 - lambda) * endpoints[0] + lambda * endpoints[1];
        }

        V3d e_div_len, e;
        Real p0_dot_e_div_len;
        std::array<V3d, 2> endpoints;
    };

    BoundaryEdgeFitter() { }

    BoundaryEdgeFitter(const TargetSurfaceMesh &mesh) {
        boundaryEdges.reserve(mesh.numBoundaryEdges());
        for (const auto &be : mesh.boundaryElements()) {
            boundaryEdges.emplace_back(be.node(0).volumeNode()->p,
                                       be.node(1).volumeNode()->p);
        }
    }

    void closestBarycoordsAndPt(Eigen::Ref<const V3d> q, Real &lambda, Eigen::Ref<V3d> p, size_t &closestEdge) {
        Real sqDist = std::numeric_limits<double>::max();
        const size_t nbe = boundaryEdges.size();

        for (size_t i = 0; i < nbe; ++i) {
            V3d pp;
            Real l;
            boundaryEdges[i].closestBarycoordsAndPt(q, l, pp);
            Real candidate_sqDist = (q - pp).squaredNorm();
            if (candidate_sqDist < sqDist) {
                sqDist      = candidate_sqDist;
                closestEdge = i;
                p           = pp;
                lambda      = l;
            }
        }
        if (sqDist == std::numeric_limits<double>::max()) throw std::runtime_error("No closest edge found");
    }

    const BoundaryEdge &edge(size_t i) const { return boundaryEdges.at(i); }

    std::vector<BoundaryEdge> boundaryEdges;
};

struct TargetSurfaceFitter {
    using Real   = InflatableSheet::Real;
    using V3d    = InflatableSheet::V3d;
    using M3d    = InflatableSheet::M3d;
    using VXd    = InflatableSheet::VXd;
    using MX3d   = InflatableSheet::MX3d;
    using MXd    = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
    using MXi    = Eigen::MatrixXi;
    using RowV3d = Eigen::Matrix<Real, 1, 3>;

    TargetSurfaceFitter(const TargetSurfaceMesh &targetMesh);

    MX3d             closestSurfPts, queryPoints;
    std::vector<M3d> closestSurfPtSensitivities; // dp_dx
    std::vector<int> closestSurfItems; // for debugging: index of the closest triangle to each internal query point;
                                       //                index of the closest edge     to each boundary query point

    void updateClosestPoints(const MX3d &pts, const std::vector<bool> &isBoundary);

    bool holdClosestPointsFixed = false;

    Real energy()   const {
        auto diff = queryPoints - closestSurfPts;
        return 0.5 * (diff.transpose() * (m_queryPtWeights.asDiagonal() * diff)).trace();
    }
    // Gradient with respect to the query points
    MX3d gradient()            const { return m_queryPtWeights.asDiagonal() * (queryPoints - closestSurfPts); }
    // Hessian with respect to query point "vi"
    M3d vtx_hessian(size_t vi) const {
        if (holdClosestPointsFixed) return m_queryPtWeights[vi] * M3d::Identity();
#if 0
        // The following eigenvalue shift "sigma" is to keep the Hessian
        // contribution strictly positive definite to help mitigate issues with
        // rank-deficient Hessians in the equilibrium solve due to compressed
        // regions (where stiffness vanishes).
        // It results in the eigenvalue modification:
        //      [(1, 0, 0), (1, 1, 0), (1, 1, 1)] ==> [(1, s, s), (1, 1, s), (1, 1, 1)]
        // for the case where the closest point lies in a tri/edge/vtx of the target surface, respectively.
        Real sigma = 1e-3;
        return m_queryPtWeights[vi] * (M3d::Identity() - (1 - sigma) * closestSurfPtSensitivities[vi]);
#else
        return m_queryPtWeights[vi] * (M3d::Identity() - closestSurfPtSensitivities[vi]);
#endif
    }

    void setTargetSurface(const TargetSurfaceMesh &targetMesh);
    void setQueryPtWeights(const VXd &weights) { m_queryPtWeights = weights; }
    const MXd &getTargetSurfaceV() const { return m_tgt_surf_V; }
    const MXi &getTargetSurfaceF() const { return m_tgt_surf_F; }

    ~TargetSurfaceFitter(); // Needed because m_tgt_surf_aabb_tree is a smart pointer to an incomplete type.

    ////////////////////////////////////////////////////////////////////////////
    // Serialization + cloning support (for pickling)
    ////////////////////////////////////////////////////////////////////////////
    using State = std::tuple<MX3d, Eigen::MatrixXi, bool,               // V, F, holdClosestPointsFixed
                             MX3d, std::vector<bool>, VXd,              // Query points, queryPtIsBoundary, query point weights
                             MX3d, std::vector<M3d>, std::vector<int>>; // Closest point information (needed if holdClosestPointsFixed is true!)
    static State serialize(const TargetSurfaceFitter &tsf) {
        return std::make_tuple(tsf.m_tgt_surf_V, tsf.m_tgt_surf_F, tsf.holdClosestPointsFixed,
                               tsf.queryPoints, tsf.m_queryPtIsBoundary, tsf.m_queryPtWeights,
                               tsf.closestSurfPts, tsf.closestSurfPtSensitivities, tsf.closestSurfItems);
    }
    static std::shared_ptr<TargetSurfaceFitter> deserialize(const State &state) {
        const auto F = std::get<1>(state);
        auto tsf = std::make_shared<TargetSurfaceFitter>(TargetSurfaceMesh(std::get<1>(state), std::get<0>(state)));
        tsf->holdClosestPointsFixed     = std::get<2>(state);
        tsf->queryPoints                = std::get<3>(state);
        tsf->m_queryPtIsBoundary        = std::get<4>(state);
        tsf->m_queryPtWeights           = std::get<5>(state);
        tsf->closestSurfPts             = std::get<6>(state);
        tsf->closestSurfPtSensitivities = std::get<7>(state);
        tsf->closestSurfItems           = std::get<8>(state);
        return tsf;
    }

    std::shared_ptr<TargetSurfaceFitter> clone() { return deserialize(serialize(*this)); }

private:
    // Target surface to which the deployed joints are fit.
    std::unique_ptr<TargetSurfaceAABB> m_tgt_surf_aabb_tree;
    VXd m_queryPtWeights;
    MXd m_tgt_surf_V, m_tgt_surf_N;
    MXi m_tgt_surf_F;

    std::vector<bool> m_queryPtIsBoundary;
    BoundaryEdgeFitter m_bdryEdgeFitter;
};

#endif /* end of include guard: TARGETSURFACEFITTER_HH */

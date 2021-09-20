#include "TargetSurfaceFitter.hh"
#include <MeshFEM/TriMesh.hh>
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/Utilities/MeshConversion.hh>
#include <igl/per_face_normals.h>
#include <igl/point_simplex_squared_distance.h>
#include <igl/AABB.h>

struct TargetSurfaceAABB : public igl::AABB<Eigen::Matrix<InflatableSheet::Real, Eigen::Dynamic, Eigen::Dynamic>, 3> {
    using Base = igl::AABB<Eigen::Matrix<InflatableSheet::Real, Eigen::Dynamic, Eigen::Dynamic>, 3>;
    using Base::Base;
};

TargetSurfaceFitter::TargetSurfaceFitter(const TargetSurfaceMesh &targetMesh) {
    setTargetSurface(targetMesh);
}

void TargetSurfaceFitter::setTargetSurface(const TargetSurfaceMesh &targetMesh) {
    m_tgt_surf_V = getV(targetMesh);
    m_tgt_surf_F = getF(targetMesh);
    igl::per_face_normals(m_tgt_surf_V, m_tgt_surf_F, m_tgt_surf_N);

    m_bdryEdgeFitter = BoundaryEdgeFitter(targetMesh);

    m_tgt_surf_aabb_tree = std::make_unique<TargetSurfaceAABB>();
    m_tgt_surf_aabb_tree->init(m_tgt_surf_V, m_tgt_surf_F);
    updateClosestPoints(queryPoints, m_queryPtIsBoundary);
}

void TargetSurfaceFitter::updateClosestPoints(const MX3d &pts, const std::vector<bool> &isBoundary) {
    if (pts.rows() != m_queryPtWeights.rows())
        throw std::runtime_error("Number of query points does not match query point weight size");
    queryPoints = pts;
    m_queryPtIsBoundary = isBoundary;
    if (isBoundary.size() != size_t(pts.rows())) throw std::runtime_error("Invalid isBoundary array");
    const size_t npts = pts.rows();

    if (holdClosestPointsFixed) return;

    BENCHMARK_SCOPED_TIMER_SECTION timer("Update closest points");
    closestSurfPts.resize(npts, 3);
    closestSurfPtSensitivities.resize(npts);
    closestSurfItems.resize(npts);

    for (size_t pi = 0; pi < npts; ++pi) {
        // Boundary vertex: find the closest point on the target boundary.
        if (isBoundary[pi]) {
            Real lambda = 0.0;
            size_t closestEdge = 0;
            V3d p;
            m_bdryEdgeFitter.closestBarycoordsAndPt(queryPoints.row(pi), lambda, p, closestEdge);
            closestSurfPts.row(pi) = p.transpose();

            if ((lambda == 0.0) || (lambda == 1.0))
                closestSurfPtSensitivities[pi].setZero();
            else {
                const auto &e = m_bdryEdgeFitter.edge(closestEdge).e;
                closestSurfPtSensitivities[pi] = e * e.transpose();
            }
            closestSurfItems[pi] = closestEdge;

            continue;
        }

        // Interior vertex: find the closest point on the target surface.
        // Could be parallelized (libigl does this internally for multi-point queries)
        RowV3d p, query;
        query = queryPoints.row(pi);
        int closest_idx;

        Real sqdist = m_tgt_surf_aabb_tree->squared_distance(m_tgt_surf_V, m_tgt_surf_F, query, closest_idx, p);
        closestSurfPts.row(pi) = p;
        closestSurfItems  [pi] = closest_idx;

        // Compute the sensitivity of the closest point projection with respect to the query point (dp_dx).
        // There are three cases depending on whether the closest point lies in the target surface's
        // interior, on one of its boundary edges, or on a boundary vertex.
        RowV3d barycoords;
        igl::point_simplex_squared_distance<3>(query, m_tgt_surf_V, m_tgt_surf_F, closest_idx, sqdist, p, barycoords);

        std::array<int, 3> nonzeroLoc;
        int numNonzero = 0;
        for (int i = 0; i < 3; ++i) {
            if (barycoords[i] == 0.0) continue;
            // It is extremely unlikely a vertex will be closest to a point/edge if this is not a stable association.
            // Therefore we assume even for smoothish surfaces that points are constrained to lie on their closest
            // simplex.
            nonzeroLoc[numNonzero++] = i;
        }
        assert(numNonzero >= 1);

        if (numNonzero == 3) {
            // If the closest point lies in the interior, the sensitivity is (I - n n^T) (the query point perturbation is projected onto the tangent plane).
            closestSurfPtSensitivities[pi] = M3d::Identity() - m_tgt_surf_N.row(closest_idx).transpose() * m_tgt_surf_N.row(closest_idx);
        }
        else if (numNonzero == 2) {
            // If the closest point lies on a boundary edge, we assume it can only slide along this edge (i.e., the constraint is active)
            // (The edge orientation doesn't matter.)
            RowV3d e = m_tgt_surf_V.row(m_tgt_surf_F(closest_idx, nonzeroLoc[0])) -
                       m_tgt_surf_V.row(m_tgt_surf_F(closest_idx, nonzeroLoc[1]));
            e.normalize();
            closestSurfPtSensitivities[pi] = e.transpose() * e;
        }
        else if (numNonzero == 1) {
            // If the closest point coincides with a boundary vertex, we assume it is "stuck" there (i.e., the constraint is active)
            closestSurfPtSensitivities[pi].setZero();
        }
        else {
            assert(false);
        }
    }
}

TargetSurfaceFitter::~TargetSurfaceFitter() = default;

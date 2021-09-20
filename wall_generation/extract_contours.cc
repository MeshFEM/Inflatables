#include "extract_contours.hh"

#include <MeshFEM/TriMesh.hh>
#include <MeshFEM/MSHFieldWriter.hh>

#include <list>
#include <map>

void extract_contours(const std::vector<MeshIO::IOVertex > &vertices,
                      const std::vector<MeshIO::IOElement> &triangles,
                      const std::vector<double> &sdf,
                      const double targetEdgeSpacing,
                      const double minContourLen, // length below which contours are removed
                      Eigen::MatrixX3d                       &outVertices,
                      std::vector<std::pair<size_t, size_t>> &outEdges)
{
    constexpr size_t NONE = std::numeric_limits<size_t>::max();
    std::vector<MeshIO::IOVertex > contourVertices;
    std::vector<MeshIO::IOElement> contourEdges;

    // The *primary* half-edge for each edge stores the point splitting the edge (if any)
    struct HalfEdgeData     { size_t contourPt = NONE; };
    struct BoundaryEdgeData { bool   visited   = false; };

    using Mesh = TriMesh<TMEmptyData, HalfEdgeData, TMEmptyData, TMEmptyData, BoundaryEdgeData>;
    Mesh mesh(triangles, vertices.size());

    std::vector<int> halfedgeForContourVertex; // index of the halfedge from which a vtx in contourVertices originated
    // Extract contour edges one triangle at a time. ("Marching triangles")
    {
        // index into contourVertices of the contour point for each half-edge
        // of a triangle (if any)
        std::vector<size_t> contourPtIdx; 
        for (const auto tri : mesh.tris()) {
            contourPtIdx.clear();
            for (const auto he : tri.halfEdges()) {
                size_t i = he.tail().index(),
                       j = he. tip().index();
                double vi = sdf.at(i),
                       vj = sdf.at(j);
                if (vi * vj >= 0) continue; // no contour on this edge

                size_t idx = he.primary()->contourPt;
                if (idx == NONE) { // If this edge's contour point hasn't been created yet, do it now.
                    // (1 - alpha) * vi + alpha * vj == 0
                    // ==> alpha = vi / (vi - vj)
                    double alpha = vi / (vi - vj);
                    Eigen::Vector3d pi = vertices[i].point,
                                    pj = vertices[j].point;
                    auto p = ((1 - alpha) * pi + alpha * pj).eval();

                    // Construct a new contour vertex.
                    idx = contourVertices.size();
                    contourVertices.emplace_back(p);
                    halfedgeForContourVertex.push_back(he.primary().index());
                    he.primary()->contourPt = idx;
                }

                contourPtIdx.push_back(idx);
            }
            const size_t n = contourPtIdx.size();
            if ((n == 1) || (n == 3)) throw std::runtime_error("Invalid field encountered");
            if (n == 2) contourEdges.emplace_back(contourPtIdx[0], contourPtIdx[1]);
        }
    }
    std::cout << "Extracted " << contourVertices.size() << " contour points" << std::endl;
    // MeshIO::save("initial_contours.obj", contourVertices, contourEdges);

    std::list<std::vector<Eigen::Vector3d>> polylines;
    // There are often small contours incident the boundary due to noise; we must ensure
    // that when these contours are filtered out, the attached contour points on the boundaries
    // are cleared--otherwise the boundary resampling will retain arbitrarily close vertices.
    std::list<std::array<size_t, 2>> incidentBdryHEsOfPolyline;

    // Decompose the extracted contours into polylines that start and end at boundary intersections
    // (which much be preserved as "feature points"). For stripes not intersecting the boundary,
    // the closed polygon is represented by a polyline whose start and endpoint is an arbitrary
    // vertex on the boundary).
    // Note: contour vertices are guaranteed to be valence 2 (interior contour point) or 1
    // (boundary contour point) since the boundary loops are not treated as contours.
    struct VTXAdjacencies {
        VTXAdjacencies() : adjIdx{{NONE, NONE}} { }
        void add(size_t idx) {
            if (adjIdx[0] == NONE) { adjIdx[0] = idx; return; }
            if (adjIdx[1] == NONE) { adjIdx[1] = idx; return; }
            throw std::runtime_error("Non-manifold contour vertex");
        }
        size_t valence() const { return (adjIdx[0] != NONE) + (adjIdx[1] != NONE); }
        size_t next(size_t prevIdx) const {
            if (adjIdx[0] == prevIdx) return adjIdx[1];
            if (adjIdx[1] == prevIdx) return adjIdx[0];
            throw std::runtime_error("Prev contour vtx passed is not actually adjacent.");
        }
        std::array<size_t, 2> adjIdx;
    };
    const size_t ncv = contourVertices.size();
    std::vector<VTXAdjacencies> adjacency(ncv);

    for (size_t ei = 0; ei < contourEdges.size(); ++ei) {
        const auto &e = contourEdges[ei];
        for (size_t i = 0; i < 2; ++i) {
            try {
                // std::cout << "Adding adjacency " << e[i] << ", " << e[(i + 1) % 2] << std::endl;
                adjacency[e[i]].add(e[(i + 1) % 2]);
            }
            catch (std::runtime_error &error) {
                size_t vi = e[i];
                std::cout << "Exception while processing vertex " << vi << " adjacency" << std::endl;
                std::cout << "For contour edge " << ei << ": " << e[0] << ", " << e[1] << std::endl;
                std::cout << "Existing adjacency: " << adjacency[vi].adjIdx[0] << ", " << adjacency[vi].adjIdx[1] << std::endl;
                throw error;
            }
        }
    }

    // Get polylines starting at the boundary. These polylines must also end at
    // the boundary, so we simply stop when we reach another valence 1 vertex.
    std::vector<bool> added(ncv, false);
    for (size_t i = 0; i < ncv; ++i) {
        if (added[i]) continue;

        size_t val = adjacency[i].valence();
        if (val == 0) throw std::runtime_error("Dangling vertex");
        if (val == 2) continue; // Skip interior vertices

        // Construct the polyline starting at "i"
        polylines.emplace_back();
        size_t heStart = halfedgeForContourVertex[i];
        if (!mesh.halfEdge(heStart).isBoundary()) {
            std::cout << contourVertices[i].point.transpose() << std::endl;
            std::cout << mesh.halfEdge(heStart).tail().index() << " --> " << mesh.halfEdge(heStart).tip().index() << std::endl;
            throw std::runtime_error("Halfedge for start boundary contour vertex is not boundary.");
        }

        incidentBdryHEsOfPolyline.emplace_back(std::array<size_t, 2>{{heStart, NONE}});
        auto &polyline = polylines.back();
        {
            size_t last;
            for (size_t curr = i, prev = NONE; curr != NONE; std::tie(curr, prev) = std::make_pair(adjacency[curr].next(prev), curr)) {
                polyline.push_back(contourVertices[curr].point);
                assert(!added[curr]);
                added[curr] = true;
                last = curr;
            }
            size_t heEnd = halfedgeForContourVertex[last];
            if (!mesh.halfEdge(heEnd).isBoundary()) throw std::runtime_error("Halfedge for end boundary contour vertex is not boundary.");
            incidentBdryHEsOfPolyline.back()[1] = heEnd;
        }
    }

    // Get closed polygons in the interior, starting at an arbitrary vertex.
    for (size_t i = 0; i < ncv; ++i) {
        if (added[i]) continue;
        polylines.emplace_back();
        incidentBdryHEsOfPolyline.emplace_back(std::array<size_t, 2>{{NONE, NONE}});
        auto &polyline = polylines.back();

        size_t curr, prev;
        for (curr = i, prev = adjacency[i].adjIdx[0]; // traverse in an arbitrary direction
             !added[curr];
             std::tie(curr, prev) = std::make_pair(adjacency[curr].next(prev), curr)) {
            polyline.push_back(contourVertices[curr].point);
            added[curr] = true;
        }
        assert(curr == i); // we better have returned to the starting point...
        polyline.push_back(contourVertices[i].point); // close the polyline by adding a second copy of the starting point.
    }

    {
        // Filter the small polylines
        std::list<std::vector<Eigen::Vector3d>> filteredPolylines;
        auto line_it = polylines.begin();
        auto bhe_it = incidentBdryHEsOfPolyline.begin();
        for (; line_it != polylines.end(); ++line_it, ++bhe_it) {
            if (bhe_it == incidentBdryHEsOfPolyline.end()) throw std::runtime_error("Zipper mismatch");
            const auto &line = *line_it;
            const auto &bhe  = *bhe_it;
            const size_t n = line.size();
            double len = 0.0;
            for (size_t i = 1; i < n; ++i) len += (line[i] - line[i - 1]).norm();
            if (len < minContourLen) {
                // Erase contour points on the boundary due to small polylines
                bool isBoundaryIncident = (bhe[0] != NONE);
                if (isBoundaryIncident != (bhe[1] != NONE)) throw std::runtime_error("Conflicting boundary incidence");
                if (isBoundaryIncident) {
                    mesh.halfEdge(bhe[0])->contourPt = NONE;
                    mesh.halfEdge(bhe[1])->contourPt = NONE;
                }
                // Discard small polyline
                continue;
            }
            filteredPolylines.emplace_back(std::move(*line_it));
        }
        if (bhe_it != incidentBdryHEsOfPolyline.end()) throw std::runtime_error("Zipper mismatch");
        polylines = std::move(filteredPolylines);
    }

    // Extract the mesh boundary loop curves as well, conforming to the channel walls.
    // Start each boundary component traversal on an edge split by a contour.
    // Each step of the boundary loop traversal adds the current boundary
    // edge's contour point (if any) and the tip vertex--the tail vertex is added by
    // the previous edge. The contour points divide the boundary loop into individual polylines.
    bool firstPolyline = true; // Is this the first polyline for the current boundary loop?
    auto processBoundaryContourPt = [&](size_t ptIdx) {
        if (ptIdx == NONE) return;
        const auto &pt = contourVertices.at(ptIdx).point;
        if (!firstPolyline) // contour point ends the old polyline...
            polylines.back().push_back(pt);
        firstPolyline = false;
        polylines.emplace_back(); // ... and starts a new one.
        polylines.back().push_back(pt);
    };

    // Contour point indices are stored on the halfedges:
    auto getBoundaryContourPt = [](decltype(mesh.boundaryEdge(0)) be) {
        return be.volumeHalfEdge()->contourPt;
    };

    // Iterate over boundary loops
    for (auto be : mesh.boundaryEdges()) {
        if (be->visited) continue;

        // Start at a contour point
        auto start = be;
        do {
            if (getBoundaryContourPt(start) != NONE) break;
        } while ((start = start.next()) != be);
        if (getBoundaryContourPt(start) == NONE) throw std::runtime_error("Couldn't find contour-boundary intersection");

        firstPolyline = true;
        auto curr = start;
        for (; !(curr->visited); curr = curr.next()) {
            processBoundaryContourPt(getBoundaryContourPt(curr));
            polylines.back().push_back(vertices[curr.tip().volumeVertex().index()].point);
            curr->visited = true;
        }
        assert(curr == start);
        // We have now returned to the first boundary edge of the loop whose tail
        // vertex and contour point are needed to finalize the current polyline.
        polylines.back().push_back(vertices[curr.tail().volumeVertex().index()].point);
        polylines.back().push_back(contourVertices.at(getBoundaryContourPt(curr)).point);
    }

    std::cout << "Extracted " << polylines.size() << " distinct contour/boundary polylines" << std::endl;

    // Resample each polyline at targetEdgeSpacing
    std::list<std::vector<Eigen::Vector3d>> processedPolylines;

    size_t polylineIdx = 0;
    for (std::vector<Eigen::Vector3d> &line : polylines) {
        ++polylineIdx;
        const size_t n = line.size();
        double len = 0.0;
        for (size_t i = 1; i < n; ++i) len += (line[i] - line[i - 1]).norm();

        if (targetEdgeSpacing < std::numeric_limits<double>::max()) {
            const bool isPolygon = (line.front() - line.back()).norm() < 1e-16;
            const size_t newEdges = std::max<size_t>(isPolygon ? 3 : 1,                    // Prevent polygons degenerating into lines
                                                     std::round(len / targetEdgeSpacing)); // Approximate target spacing
            const double spacing = len / newEdges;

            std::vector<Eigen::Vector3d> newPoints;
            newPoints.reserve(newEdges + 1);

            newPoints.push_back(line.front());
            double distToNextPoint = spacing;
            for (size_t i = 1; i < n; ++i) { // Traverse original curve
                double distAlongEdge = 0;    // Start at beginning of edge
                const auto &p0 = line[i - 1],
                           &p1 = line[i    ];
                const double elen = (p1 - p0).norm();
                while (distAlongEdge + distToNextPoint <= elen) {
                    distAlongEdge += distToNextPoint;
                    double alpha = distAlongEdge / elen;
                    if (newPoints.size() == newEdges) { // The last point will be inserted at the end...
                        // but verify we have actually reached the last point properly
                        assert(i == n - 1);
                        if (std::abs(elen - distAlongEdge) > 1e-10 * targetEdgeSpacing) throw std::runtime_error("Did not reach last point");
                        distToNextPoint = 0;
                        goto traversalFinished;
                    }
                    newPoints.push_back((1.0 - alpha) * p0 + alpha * p1);
                    distToNextPoint = spacing;
                }
                distToNextPoint -= (elen - distAlongEdge); // Move to the next edge
                assert(distToNextPoint > 0);
            }
            traversalFinished:
            // std::cout << "Finished line resampling at spacing " << spacing << " with final gap " << distToNextPoint << std::endl;
            assert(newPoints.size() == newEdges);      // We should have already inserted all the points except the last...
            // ... and we should have reached the last point's location (up to machine precision).
            if (std::abs(distToNextPoint) > 1e-10 * targetEdgeSpacing) throw std::runtime_error("Did not reach last point");
            newPoints.push_back(line.back());
            line = std::move(newPoints); // replace the old polyline with the newly resampled one.
        }
        processedPolylines.emplace_back(std::move(line));
    }

    // Generate output vertices and edges, removing duplicate vertices
    using Pt = std::tuple<double, double, double>;
    using PointGluingMap = std::map<Pt, size_t>;
    PointGluingMap indexForPoint;
    std::vector<Eigen::Vector3d> gluedVertices;
    auto addPoint = [&](Eigen::Vector3d pt) {
        auto key = std::make_tuple(pt[0], pt[1], pt[2]);
        auto it = indexForPoint.lower_bound(key);
        if ((it != indexForPoint.end()) && (it->first == key)) return it->second;

        size_t idx = gluedVertices.size();
        gluedVertices.emplace_back(pt);
        indexForPoint.emplace_hint(it, key, idx);
        return idx;
    };

    outEdges.clear();
    for (const auto &line : processedPolylines) {
        size_t prevIdx = NONE;
        for (const auto &pt : line) {
            size_t idx = addPoint(pt);
            if (prevIdx != NONE) outEdges.emplace_back(prevIdx, idx);
            prevIdx = idx;
        }
    }

    outVertices.resize(gluedVertices.size(), 3);
    for (size_t i = 0; i < gluedVertices.size(); ++i)
        outVertices.row(i) = gluedVertices[i];
}

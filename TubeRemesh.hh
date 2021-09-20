////////////////////////////////////////////////////////////////////////////////
// TubeRemesh.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Often the triangulation of the tube regions is such that edges or whole
//  triangles at the corners and thin regions will be spuriously fused. We
//  minimize these effects by remeshing the "tube mesh" (the mesh consisting of
//  only the triangulated tube region whose boundary is the fusing curve
//  network).
//
//  In particular, we split spuriously fused edges (non-fused edges
//  connecting fused vertices) and triangulate the resulting polygons. Since
//  glancing intersections between fusing curves at the boundary can create
//  very small angles and extremely short edges, we only split edges above a
//  specified minimum length threshold (which is relative to the mean edge
//  length).
//  Edges shorter than this threshold are assumed not to let air through
//  anyway (thin gaps do not inflate well in practice).
//
//  As the input mesh is assumed to have nice, nearly-equiliateral triangles,
//  this length threshold also prevents the creation of tiny angles.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  12/18/2020 15:35:56
////////////////////////////////////////////////////////////////////////////////
#ifndef TUBEREMESH_HH
#define TUBEREMESH_HH
#include <limits>
#include <MeshFEM/Utilities/MeshConversion.hh>

// Attempt to remesh a triangulated tube mesh so that edges are only fused
// (i.e., connect vertices in `fusedVertices`) if they coincide with one of the
// original `fusedSegments`. This will not always be possible without creating
// extremely tiny edges.
// The newly created vertices and half-edges are guaranteed to be unfused. Furthermore,
// new vertices are only appended at the end, while the indices of the original vertices
// remain unchanged. Therefore, the input `isFusedV` and `fusedSegments` arrays remain valid.
// The calling code must simply pad `isFusedV` with zeros up to the new vertex count
// to get the fused markers for the remeshed tubes.
template<class Mesh>
std::shared_ptr<Mesh> tubeRemesh(const Mesh &tubeMesh,
                                 const std::vector<bool> &isFusedV,
                                 const std::vector<std::array<int, 2>> &fusedSegments,
                                 Real minRelEdgeLen = 0.5) {
    static constexpr size_t NONE = std::numeric_limits<size_t>::max();

    const size_t nhe = tubeMesh.numHalfEdges(),
                 nv  = tubeMesh.numVertices();
    // Locate the mesh halfedges corresponding to fused segments
    std::vector<bool> isFusedHE(nhe, false);
    for (const auto &fs : fusedSegments) {
        auto he = tubeMesh.halfEdge(fs[0], fs[1]);
        if (he.index() == -1) {
            // When the tube mesh boundary is non-manifold, the
            // circulation-based halfege lookup may fail. For these cases we do
            // a brute-force lookup to confirm the segment is truly missing.
            // This should not be too expensive since there should be very few
            // non-manifold boundary vertices.
            //
            // (Non-manifold tube meshes can happen when remeshing an
            // inflatable sheet that had two nearly-intersecting walls which
            // ended up touching at a vertex due to the addition of a
            // spuriously fused triangle.)
            std::cout << "WARNING: running brute-force search" << std::endl;
            for (const auto he_search : tubeMesh.halfEdges()) {
                if (    ((he_search.tail().index() == fs[0]) && (he_search.tip().index() == fs[1]))
                     || ((he_search.tail().index() == fs[1]) && (he_search.tip().index() == fs[0]))) {
                    he = he_search;
                    break;
                }
            }
            std::cout << "Found he " << he.index() << ": " << he.tail().index() << " --> " << he.tip().index() << std::endl;
            std::cout << "Opposite he " << he.opposite().index() << ": " << he.opposite().tail().index() << " --> " << he.opposite().tip().index() << std::endl;
            std::cout << "Primary he " << he.primary().index() << ": " << he.primary().tail().index() << " --> " << he.primary().tip().index() << std::endl;
            std::cout << "Primary he is boundary " << he.primary().isBoundary() << std::endl;
            if (he.index() == -1)
                throw std::runtime_error("Fused segment does not exist in mesh: "
                                            + std::to_string(fs[0]) + ", "
                                            + std::to_string(fs[1]));
        }
        he = he.primary();
        isFusedHE[he.index()] = true;
        if (!he.isBoundary())
            isFusedHE[he.opposite().index()] = true;
    }

    for (auto he : tubeMesh.halfEdges()) {
        if (!he.isBoundary() || isFusedHE[he.index()]) continue;
        std::cout << "WANRING: boundary halfedge " << he.index() << ": " << he.tail().index() << " --> " << he.tip().index()
                  << "was not marked as a fused segment; adding it." << std::endl;
        isFusedHE[he.index()] = true;
    }

    if (isFusedV.size() != nv) throw std::runtime_error("Invalid isFusedV size");

    Eigen::VectorXd edgeLength(nhe);
    std::vector<size_t> spuriouslyFused(nhe);
    for (auto he : tubeMesh.halfEdges()) {
        edgeLength[he.index()] = (he.tip().node()->p - he.tail().node()->p).norm();
        spuriouslyFused[he.index()] = isFusedV [he.tip().index()]
                                  &&  isFusedV [he.tail().index()]
                                  && !isFusedHE[he.index()];
    }

    // Note: edgeLength.mean() effectively gives the interior edges twice the
    // weight of the boundary edges, but this shouldn't matter...
    const Real minEdgeLen = minRelEdgeLen * edgeLength.mean();

    // Determine which edges need to be split, tagging them
    // with the (global) index of the midpoint vertex that will be created to
    // split them (in `edgeVtx`).
    std::vector<size_t> edgeVtx(nhe, NONE);
    const size_t old_nv = tubeMesh.numVertices();
    size_t numSplitEdges = 0;

    for (auto he : tubeMesh.halfEdges()) {
        const size_t hei = he.index();
        if (!he.isPrimary() || !spuriouslyFused[hei]) continue;
        if (edgeLength[hei] < 2 * minEdgeLen) continue; // don't create edges below minEdgeLen

        assert(!he.isBoundary()); // At this point, all boundary edges should be marked as fused segments and cannot be spuriously fused

        edgeVtx[he.index()] = edgeVtx[he.opposite().index()] = old_nv + numSplitEdges;
        ++numSplitEdges;
    }

    auto isSplit = [&](auto he) { return edgeVtx[he.index()] != NONE; };

    // Actually create the midpoints, appending to the full output vertex set.
    auto outVertices = getMeshIOVertices(tubeMesh);
    outVertices.reserve(outVertices.size() + numSplitEdges);
    for (auto he : tubeMesh.halfEdges()) {
        if (he.isPrimary() && isSplit(he))
            outVertices.emplace_back((0.5 * (he.tip().node()->p + he.tail().node()->p)).eval());
    }

    // Retriangulate all triangles containing split edges
    std::vector<MeshIO::IOElement> outTris;
    outTris.reserve(tubeMesh.numTris() + 2 * numSplitEdges);

    for (auto t : tubeMesh.tris()) {
        auto he = t.halfEdge(0);
        auto he_start = he;
        // Find the first in ccw order of up to three split half-edges in this triangle
        while (!isSplit(he)) { he = he.next(); if (he == he_start) break; }
        if (isSplit(he.prev())) he = he.prev();

        // Get the split vertex indices
        size_t s1 = edgeVtx[he.index()],
               s2 = edgeVtx[he.next().index()],
               s3 = edgeVtx[he.prev().index()];

        size_t v1 = he.tail().index(),
               v2 = he.tip().index(),
               v3 = he.next().tip().index();

        if (s1 == NONE) {
            assert(s2 == NONE);
            outTris.emplace_back(v1, v2, v3);
        }
        else if (s2 == NONE) {
            // Split into two triangles
            //     3             3
            //    / \           /|\
            //   /   \   ==>   / | \
            //  /  s1 \       /  |  \
            // 1---*---2     1---*---2
            //     he            he
            outTris.emplace_back(v1, s1, v3);
            outTris.emplace_back(s1, v2, v3);
        }
        else if (s3 == NONE) {
            // Split into three triangles
            //     3              3               3
            //    / \            /|\             / \
            //   /   * s2 ==>   / | *    or     /  ,*
            //  /  s1 \        /  |/ \         /,-'/ \
            // 1---*---2      1---*---2       1---*---2
            //     he             he              he
            // There are two possible triangulations, and we pick the one
            // splitting the larger angle. By the law of sines, this means
            // splitting the angle opposite the longest edge.
            // In all cases, we create the triangle (s1, tip, s2):
            outTris.emplace_back(s1, v2, s2);
            if (edgeLength[he.index()] > edgeLength[he.next().index()]) {
                // Split angle opposite `he` (left case in diagram above)
                outTris.emplace_back(v1, s1, v3);
                outTris.emplace_back(s1, s2, v3);
            }
            else {
                // Split angle opposte `he.next()` (right case)
                outTris.emplace_back(v1, s1, s2);
                outTris.emplace_back(v1, s2, v3);
            }
        }
        else {
            // Split into four triangles
            //     3              3 
            //    / \            / \
            // s3*   * s2 ==>   *---* 
            //  /  s1 \        / \ / \
            // 1---*---2      1---*---2
            //     he             he
            outTris.emplace_back(v1, s1, s3);
            outTris.emplace_back(s1, v2, s2);
            outTris.emplace_back(s1, s2, s3);
            outTris.emplace_back(s3, s2, v3);
        }
    }

    return std::make_unique<Mesh>(outTris, outVertices);
}

#endif /* end of include guard: TUBEREMESH_HH */

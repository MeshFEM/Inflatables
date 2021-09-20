#ifndef DUAL_MESH_HH
#define DUAL_MESH_HH

#include <MeshFEM/MeshIO.hh>

template<class Mesh>
void barycentricDual(const Mesh &m,
                     std::vector<MeshIO::IOVertex > &dualVertices,
                     std::vector<MeshIO::IOElement> &dualPolygons) {
    const size_t ne = m.numElements();
    dualVertices.clear();
    dualVertices.reserve(ne);
    for (size_t ei = 0; ei < ne; ++ei)
        dualVertices.push_back(m.elementBarycenter(ei));

    const size_t nv = m.numVertices();
    dualPolygons.clear();
    dualPolygons.reserve(nv); // There will be fewer than nv polygons, since we we only have them for internal vertices.
    for (const auto v : m.vertices()) {
        if (v.isBoundary()) continue;
        dualPolygons.emplace_back();
        // Circulate counter-clockwise.
        for (const auto he : v.incidentHalfEdges()) {
            assert(!he.isBoundary());
            dualPolygons.back().push_back(he.tri().index());
        }
    }
}

// Assumes the dual region is convex...
template<class Mesh>
void triangulatedBarycentricDual(const Mesh &m,
                                 std::vector<MeshIO::IOVertex > &dualVertices,
                                 std::vector<MeshIO::IOElement> &dualTriangles,
                                 std::vector<size_t> &originatingPolygon) {
    std::vector<MeshIO::IOElement> dualPolygons;
    barycentricDual(m, dualVertices, dualPolygons);

    dualTriangles.clear();
    originatingPolygon.clear();

    bool toggle = false;
    for (size_t pi = 0; pi < dualPolygons.size(); ++pi) {
        const auto &p = dualPolygons[pi];

        size_t start = 0, end = p.size() - 1;
        while (end - start + 1 >= 3) {
            if (toggle) {
                dualTriangles.emplace_back(p[start], p[start + 1], p[end]);
                ++start;
            }
            else {
                dualTriangles.emplace_back(p[start], p[end - 1], p[end]);
                --end;
            }
            originatingPolygon.push_back(pi);
            toggle = !toggle;
        }
    }
}

#endif /* end of include guard: DUAL_MESH_HH */

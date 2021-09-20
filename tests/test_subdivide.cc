#include <MeshFEM/MeshIO.hh>
#include "../subdivide_triangle.hh"

int main(int argc, const char *argv[]) {
    if (argc != 4) {
        std::cerr << "usage: test_subdivide in_mesh.obj nsubdiv out_mesh" << std::endl;
        exit(-1);
    }
    const std::string inPath = argv[1],
                     outPath = argv[3];
    const int nsubdiv = std::stoi(argv[2]);

    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    MeshIO::load(inPath, vertices, elements);

    std::vector<MeshIO::IOVertex > outVertices;
    std::vector<MeshIO::IOElement> outElements;
    PointGluingMap indexForPoint;

    auto newPt = [&](const Point3D &p, Real /* lambda_0 */, Real /* lambda_1 */, Real /* lambda_2 */) {
        outVertices.emplace_back(p);
        return outVertices.size() - 1;
    };

    auto newTri = [&](size_t i0, size_t i1, size_t i2) { outElements.emplace_back(i0, i1, i2); };

    for (const auto &e : elements) {
        if (e.size() != 3) throw std::runtime_error("Only triangles are supported");
        subdivide_triangle(nsubdiv, vertices[e[0]].point, vertices[e[1]].point, vertices[e[2]].point, indexForPoint, newPt, newTri);
    }

    MeshIO::save(outPath, outVertices, outElements);

    return 0;
}

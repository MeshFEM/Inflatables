#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
namespace py = pybind11;

#include "../examples/paraboloid.hh"
#include "../SurfaceSampler.hh"
#include "../subdivide_triangle.hh"

#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/MSHFieldWriter.hh>
#include <MeshFEM/Utilities/MeshConversion.hh>

#include <igl/loop.h>
#include <igl/intrinsic_delaunay_triangulation.h>
#include <igl/edge_lengths.h>
#include <igl/intrinsic_delaunay_cotmatrix.h>

#include "../InflatableSheet.hh"
#include "../DualMesh.hh"
#include "../TubeRemesh.hh"

// using Mesh = FEMMesh<2, 1, Vector3D>; // Piecewise linear triangle mesh embedded in R^3
using Real_ = InflatableSheet::Real;
using Mesh  = InflatableSheet::Mesh;
using V3d   = InflatableSheet::V3d;
using MX3d  = InflatableSheet::MX3d;

PYBIND11_MODULE(mesh_utilities, mod) {
    py::module::import("MeshFEM");

    mod.def("subdivide", [](const Mesh &m, const size_t ns) {
                PointGluingMap indexForPoint;

                std::vector<V3d>               subVertices;
                std::vector<MeshIO::IOElement> subTris;

                for (const auto e : m.elements()) {
                    subdivide_triangle(ns, e.node(0)->p, e.node(1)->p, e.node(2)->p, indexForPoint,
                            [&](const V3d &p, Real_, Real_, Real_) { subVertices.emplace_back(p); return subVertices.size() - 1; },
                            [&](size_t i0, size_t i1, size_t i2) { subTris.emplace_back(i0, i1, i2); }
                    );
                }
                return std::make_shared<Mesh>(subTris, subVertices);
            }, py::arg("mesh"), py::arg("numSubdiv") = 1);
    mod.def("subdivide_loop", [](const Mesh &m, const size_t ns) {
                auto            VCage = getV(m);
                Eigen::MatrixXi FCage = getF(m), F;

                decltype(VCage) V;

                igl::loop(VCage, FCage, V, F, ns);
                auto mio = getMeshIO(V, F);
                return std::make_shared<Mesh>(mio.second, mio.first);
            }, py::arg("mesh"), py::arg("numSubdiv") = 1) ;

    mod.def("barycentricDual", [](const Mesh &m) {
                std::vector<MeshIO::IOVertex > dualVertices;
                std::vector<MeshIO::IOElement> dualPolygons;
                barycentricDual(m, dualVertices, dualPolygons);

                std::pair<Eigen::MatrixX3d, std::vector<std::vector<size_t>>> result;
                result.first = getV(dualVertices);

                auto &dualPolygonsStdVec = result.second;
                dualPolygonsStdVec.reserve(dualPolygons.size());
                for (const auto &p : dualPolygons)
                    dualPolygonsStdVec.emplace_back(p);

                return result;
            });

    mod.def("triangulatedBarycentricDual", [](const Mesh &m) {
                std::vector<MeshIO::IOVertex > dualVertices;
                std::vector<MeshIO::IOElement> dualTris;

                std::tuple<Eigen::MatrixX3d, Eigen::MatrixXi, std::vector<size_t>> result;
                triangulatedBarycentricDual(m, dualVertices, dualTris, std::get<2>(result));
                std::get<0>(result) = getV(dualVertices);
                std::get<1>(result) = getF(dualTris);
                return result;
            });

    mod.def("barycentricDualIDT", [](const Mesh &m) {
        std::vector<MeshIO::IOVertex > dualVertices;
        std::vector<MeshIO::IOElement> dualTris;
        std::vector<size_t> originatingPolygon;
        triangulatedBarycentricDual(m, dualVertices, dualTris, originatingPolygon);
        Eigen::MatrixX3d V = getV(dualVertices);
        Eigen::MatrixXi  F = getF(dualTris);

        Eigen::MatrixXd l;
        igl::edge_lengths(V, F, l);
        Eigen::MatrixXd l_intrinsic;
        Eigen::MatrixXi F_intrinsic;
        igl::intrinsic_delaunay_triangulation(l, F, l_intrinsic, F_intrinsic);
        return std::make_pair(l_intrinsic, F_intrinsic);
    });

    mod.def("barycentricDualIDTLaplacian", [](const Mesh &m) {
        std::vector<MeshIO::IOVertex > dualVertices;
        std::vector<MeshIO::IOElement> dualTris;
        std::vector<size_t> originatingPolygon;
        triangulatedBarycentricDual(m, dualVertices, dualTris, originatingPolygon);

        Eigen::SparseMatrix<double> L;
        igl::intrinsic_delaunay_cotmatrix(getV(dualVertices), getF(dualTris), L);
        return L;
    });

    mod.def("tubeRemesh", [](const Mesh &tubeMesh,
                             const std::vector<bool> &isFusedV,
                             const std::vector<std::array<int, 2>> &fusedSegments,
                             Real minRelEdgeLen) {
                return std::shared_ptr<Mesh>(tubeRemesh(tubeMesh, isFusedV, fusedSegments, minRelEdgeLen)); // must convert unique_ptr to the mesh holder type to avoid a memory bug!
            }, py::arg("tubeMesh"), py::arg("isFusedV"), py::arg("fusedSegments"), py::arg("minRelEdgeLen") = 0.5);

    ////////////////////////////////////////////////////////////////////////////////
    // SurfaceSampler
    ////////////////////////////////////////////////////////////////////////////////
    py::class_<SurfaceSampler>(mod, "SurfaceSampler")
        .def(py::init<const Eigen::MatrixXd &, const Eigen::MatrixXi &>(),
             py::arg("V"), py::arg("F"))
        .def("sample", &SurfaceSampler::sample<Eigen::VectorXd> , py::arg("P"), py::arg("fieldValues")) // Scalar field
        .def("sample", &SurfaceSampler::sample<Eigen::MatrixX3d>, py::arg("P"), py::arg("fieldValues")) // Vector field
        .def("closestTriAndBaryCoords", [](const SurfaceSampler &ss, const Eigen::MatrixXd &P) {
                using RType = std::tuple<Eigen::VectorXi,   // I
                                         Eigen::MatrixX3d>; // B
                RType result;
                ss.closestTriAndBaryCoords(P, std::get<0>(result), std::get<1>(result));
                return result;
            }, py::arg("P"))
        ;

    ////////////////////////////////////////////////////////////////////////////
    // Free-standing mesh generation functions.
    ////////////////////////////////////////////////////////////////////////////
    mod.def("paraboloid", [](double triArea, double k1, double k2, bool delaunay) {
                std::vector<MeshIO::IOVertex > vertices;
                std::vector<MeshIO::IOElement> elements;
                paraboloid(triArea, k1, k2, vertices, elements, delaunay);
                return std::make_shared<Mesh>(elements, vertices);
            }, py::arg("triArea") = 0.01, py::arg("k1") = 1.0, py::arg("k2") = -1.0, py::arg("delaunay") = true);
}

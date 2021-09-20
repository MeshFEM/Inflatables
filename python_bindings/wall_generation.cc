#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>

#include <evaluate_stripe_field.hh>
#include <extract_contours.hh>
#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/Triangulate.h>

#include "../InflatableSheet.hh"

#include <tuple>

namespace py = pybind11;

using Mesh = InflatableSheet::Mesh; // Piecewise linear triangle mesh embedded in R^3

PYBIND11_MODULE(wall_generation, m) {
    m.doc() = "Air channel wall generation";

    py::module::import("MeshFEM");
    py::module::import("mesh");

    ////////////////////////////////////////////////////////////////////////////////
    // Free-standing functions
    ////////////////////////////////////////////////////////////////////////////////
    m.def("evaluate_stripe_field",
            [](const Eigen::MatrixX3d &vertices,
               const Eigen::MatrixX3i &triangles,
               const std::vector<double> &stretchAngles,
               const std::vector<double> &wallWidths,
               const double frequency,
               const size_t nsubdiv, bool glue) {
                std::tuple<Eigen::MatrixX3d,
                           Eigen::MatrixX3i,
                           Eigen::VectorXd> result;
                std::vector<double> stripeField;
                evaluate_stripe_field(vertices, triangles, stretchAngles, wallWidths, frequency,
                                      std::get<0>(result), std::get<1>(result), stripeField,
                                      nsubdiv, glue);
                std::get<2>(result) = Eigen::Map<Eigen::VectorXd>(stripeField.data(), stripeField.size());
                return result;
            },
            py::arg("vertices"), py::arg("triangles"),
            py::arg("stretchAngles"), py::arg("wallWidths"), py::arg("frequency") = 130.0 /* default frequency from Keenan's code */,
            py::arg("nsubdiv") = 3, py::arg("glue") = true);

    m.def("extract_contours", [](const Eigen::MatrixX3d &vertices,
                                 const Eigen::MatrixX3i &triangles,
                                 const std::vector<double> &sdf,
                                 const double targetEdgeSpacing,
                                 const double minContourLen) {
                std::tuple<Eigen::MatrixX3d, std::vector<std::pair<size_t, size_t>>> result;
                std::vector<MeshIO::IOVertex > ioVertices(vertices.rows());
                std::vector<MeshIO::IOElement> ioElements(triangles.rows());
                for (size_t i = 0; i < ioVertices.size(); ++i) ioVertices[i].point = vertices.row(i);
                for (size_t i = 0; i < ioElements.size(); ++i) ioElements[i]       = MeshIO::IOElement(triangles(i, 0), triangles(i, 1), triangles(i, 2));
                extract_contours(ioVertices, ioElements, sdf, targetEdgeSpacing, minContourLen,
                                 std::get<0>(result), std::get<1>(result));
                return result;
            }, 
            py::arg("vertices"),
            py::arg("triangles"),
            py::arg("sdf"),
            py::arg("targetEdgeSpacing") = std::numeric_limits<double>::max(),
            py::arg("minContourLen") = 0);

    m.def("triangulate_channel_walls", [](const std::vector<Eigen::Vector2d> &pts,
                                          std::vector<std::pair<size_t, size_t>> &edges,
                                          double triArea,
                                          const std::string &flags,
                                          bool omitQualityFlag,
                                          const std::vector<Point2D> &holePoints) {
            std::vector<MeshIO::IOVertex > vertices;
            std::vector<MeshIO::IOElement> elements;
            std::vector<int> pointMarkers;
            std::vector<std::array<int, 2>> markedEdges;
            triangulatePSLG(pts, edges, holePoints,
                            vertices, elements, triArea, flags, &pointMarkers, &markedEdges, omitQualityFlag);
            return py::make_tuple(std::make_shared<Mesh>(elements, vertices), pointMarkers, markedEdges);
        }, py::arg("pts"), py::arg("edges"), py::arg("triArea") = 0.01, py::arg("flags") = "", py::arg("omitQualityFlag") = false, py::arg("holePoints") = std::vector<Point2D>());

    ////////////////////////////////////////////////////////////////////////////////
    // Enable output redirection from Python side
    ////////////////////////////////////////////////////////////////////////////////
    py::add_ostream_redirect(m, "ostream_redirect");
}

#include <iostream>
#include <iomanip>
#include <sstream>
#include <utility>
#include <memory>

#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/MeshIO.hh>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#include "../BendingEnergy.hh"
#include "../MetricFitter.hh"
#include "../CollapsePreventionEnergy.hh"
#include "../fit_metric_newton.hh"

PYBIND11_MODULE(metric_fit, m) {
    m.doc() = "Metric-fitting surface immersion optimization";

    py::module::import("py_newton_optimizer");

    ////////////////////////////////////////////////////////////////////////////////
    // Free-standing functions
    ////////////////////////////////////////////////////////////////////////////////
    m.def("fit_metric_newton",  &fit_metric_newton, py::arg("mfit"), py::arg("fixedVars"), py::arg("options") = NewtonOptimizerOptions(), py::arg("callback") = nullptr);

    ////////////////////////////////////////////////////////////////////////////////
    // Mesh construction (for mesh type used by metric fitting routines)
    ////////////////////////////////////////////////////////////////////////////////
    using Mesh = MetricFitter::Mesh;
    // WARNING: Mesh's holder type is a shared_ptr; returning a unique_ptr will lead to a dangling pointer in the current version of Pybind11
    m.def("Mesh", [](const std::string &path) { return std::shared_ptr<Mesh>(Mesh::load(path)); }, py::arg("path"));
    m.def("Mesh", [](const Eigen::MatrixX2d &V, const Eigen::MatrixX3i &F) { return std::make_shared<Mesh>(F, V); }, py::arg("V"), py::arg("F"));
    m.def("Mesh", [](const Eigen::MatrixX3d &V, const Eigen::MatrixX3i &F) { return std::make_shared<Mesh>(F, V); }, py::arg("V"), py::arg("F"));

    using HE = HingeEnergy<double>;
    py::class_<HE>(m, "HingeEnergy")
        .def(py::init<const Point3D &, const Point3D &, const Point3D &, const Point3D &>(), py::arg("p0"), py::arg("p1"), py::arg("p2"), py::arg("p3"))
        .def("setDeformedConfiguration", &HE::setDeformedConfiguration, py::arg("p0"), py::arg("p1"), py::arg("p2"), py::arg("p3"))
        .def("gradTheta",                &HE::gradTheta)
        .def("hessTheta",                &HE::hessTheta)
        .def_readonly("e_bar_len", &HE::e_bar_len)
        .def_readonly("h_bar"    , &HE::h_bar)
        .def_readonly("theta_bar", &HE::theta_bar)

        .def_readonly("e_len"    , &HE::e_len)
        .def_readonly("theta"    , &HE::theta)

        .def_readonly("deformed_pts", &HE::deformed_pts)

        .def("energy"  , &HE::energy)
        .def("gradient", &HE::gradient)
        .def("hessian" , &HE::hessian)
        ;

    using CPE = CollapsePreventionEnergyDet;
    py::class_<CPE>(m, "CollapsePreventionEnergyDet")
        .def(py::init<>())
        .def_property("activationThreshold", &CPE::activationThreshold, &CPE::setActivationThreshold)
        .def("setMatrix",      [](CPE &cpe, const Eigen::Matrix2d &C) { cpe.setMatrix(C); })
        .def("energy",         &CPE::energy)
        .def("denergy",        &CPE::denergy)
        .def("delta_denergy",  [](const CPE &cpe, const Eigen::Matrix2d &dC) { return cpe.delta_denergy(dC).cast<double>(); })

        // det-specific functions for debugging.
        .def("setDet",         &CPE::setDet)
        .def("det",            &CPE::det   )
        .def("normalizedDet",  &CPE::normalizedDet)
        .def("denergy_ddet",   &CPE::denergy_ddet)
        .def("d2energy_d2det", &CPE::d2energy_d2det)
        ;

    py::class_<MetricFitter> pyMetricFitter(m, "MetricFitter");

    using EnergyType = MetricFitter::EnergyType;
    py::enum_<EnergyType>(pyMetricFitter, "EnergyType")
        .value("Full"         ,      EnergyType::Full)
        .value("MetricFitting",      EnergyType::MetricFitting)
        .value("Bending"      ,      EnergyType::Bending)
        .value("Gravitational",      EnergyType::Gravitational)
        .value("CollapsePrevention", EnergyType::CollapsePrevention)
        ;

    pyMetricFitter
        .def(py::init<const std::shared_ptr<Mesh> &>(), py::arg("mesh"))

        .def("mesh", &MetricFitter::meshPtr)

        .def_property_readonly("rigidMotionPinVars", [](const MetricFitter &mf) { return mf.rigidMotionPinVars(); })

        .def("numVars", &MetricFitter::numVars)
        .def("getVars", &MetricFitter::getVars)
        .def("setVars", &MetricFitter::setVars)

        .def("setIdentityImmersion", &MetricFitter::setIdentityImmersion)
        .def("getIdentityImmersion", &MetricFitter::getIdentityImmersion)
        .def(        "setImmersion", &MetricFitter::        setImmersion)
        .def(        "getImmersion", &MetricFitter::        getImmersion)

        .def_property_readonly("rigidMotionPinVars", [](const MetricFitter &mf) { return mf.rigidMotionPinVars(); })

        .def("setTargetMetric", &MetricFitter::setTargetMetric, py::arg("targetMetric"), py::arg("relativeCollapsePreventionThreshold") = 0.25)

        .def("setCurrentMetricAsTarget", &MetricFitter::setCurrentMetricAsTarget, py::arg("relativeCollapsePreventionThreshold") = 0.25)

        .def_readwrite("bendingStiffness",         &MetricFitter::bendingStiffness)
        .def_readwrite("collapsePreventionWeight", &MetricFitter::collapsePreventionWeight)
        .def_readwrite("gravityVector",            &MetricFitter::gravityVector)

        .def("energy",   &MetricFitter::energy  , py::arg("energyType") = EnergyType::Full)
        .def("gradient", &MetricFitter::gradient, py::arg("energyType") = EnergyType::Full)
        .def("hessian", py::overload_cast<EnergyType>(&MetricFitter::hessian, py::const_), py::arg("energyType") = EnergyType::Full)

        .def("hinge",             &MetricFitter::hinge, py::arg("hingeIdx"))
        .def("numHinges",         &MetricFitter::numHinges)
        .def("getIncidentHinges", &MetricFitter::getIncidentHinges, py::arg("vtxIdx"))

        .def("metricDistSq",      &MetricFitter::metricDistSq)
        .def("collapsePreventer", &MetricFitter::collapsePreventer, py::arg("edgeIdx"))
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // Enable output redirection from Python side
    ////////////////////////////////////////////////////////////////////////////////
    py::add_ostream_redirect(m, "ostream_redirect");
}

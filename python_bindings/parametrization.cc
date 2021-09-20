#include <iostream>
#include <iomanip>
#include <sstream>
#include <utility>
#include <memory>
#include <functional>

#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/MeshIO.hh>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#include "../DualLaplacianStencil.hh" // Must be included before MeshFEM's Triplet is defined...

#include "../parametrization.hh"
#include "../parametrization_newton.hh"

#include "../circular_mean.hh"

template <typename T>
std::string to_string_with_precision(const T &val, const int n = 6) {
    std::ostringstream ss;
    ss << std::setprecision(n) << val;
    return ss.str();

}
// Conversion of std::tuple to and from a py::tuple, since pybind11 doesn't seem to provide this...
template<typename... Args, size_t... Idxs>
py::tuple to_pytuple_helper(const std::tuple<Args...> &args, std::index_sequence<Idxs...>) {
    return py::make_tuple(std::get<Idxs>(args)...);
}

template<typename... Args>
py::tuple to_pytuple(const std::tuple<Args...> &args) {
    return to_pytuple_helper(args, std::make_index_sequence<sizeof...(Args)>());
}

template<class OutType>
struct FromPytupleImpl;

template<typename... Args>
struct FromPytupleImpl<std::tuple<Args...>> {
    template<size_t... Idxs>
    static auto run_helper(const py::tuple &t, std::index_sequence<Idxs...>) {
        return std::make_tuple((t[Idxs].cast<Args>())...);
    }
    static auto run(const py::tuple &t) {
        if (t.size() != sizeof...(Args)) throw std::runtime_error("Mismatched tuple size for py::tuple to std::tuple conversion.");
        return run_helper(t, std::make_index_sequence<sizeof...(Args)>());
    }
};

template<class OutType>
OutType from_pytuple(const py::tuple &t) {
    return FromPytupleImpl<OutType>::run(t);
}

using namespace parametrization;

PYBIND11_MODULE(inflatables_parametrization, m) {
    m.doc() = "Shear metric parametrization";

    py::module::import("py_newton_optimizer");
    py::module::import("mesh_utilities");

    ////////////////////////////////////////////////////////////////////////////////
    // Mesh construction (for mesh type used by parametrization routines)
    ////////////////////////////////////////////////////////////////////////////////
    // WARNING: Mesh's holder type is a shared_ptr; returning a unique_ptr will lead to a dangling pointer in the current version of Pybind11
    m.def("Mesh", [](const std::string &path) { return std::shared_ptr<Mesh>(Mesh::load(path)); }, py::arg("path"));
    m.def("Mesh", [](const Eigen::MatrixX3d &V, const Eigen::MatrixX3i &F) { return std::make_shared<Mesh>(F, V); }, py::arg("V"), py::arg("F"));

    ////////////////////////////////////////////////////////////////////////////////
    // Free-standing functions
    ////////////////////////////////////////////////////////////////////////////////
    m.def("lscm",     &lscm,     py::arg("mesh"),                          "Compute least-squares conformal parametrization");
    m.def("harmonic", &harmonic, py::arg("mesh"), py::arg("boundaryData"), "Compute harmonic map with given dirichlet data");

    m.def("regularized_parametrization_newton", &regularized_parametrization_newton<RegularizedParametrizer   >, py::arg("rparam"), py::arg("fixedVars"), py::arg("options") = NewtonOptimizerOptions());
    m.def("regularized_parametrization_newton", &regularized_parametrization_newton<RegularizedParametrizerSVD>, py::arg("rparam"), py::arg("fixedVars"), py::arg("options") = NewtonOptimizerOptions());

    using AngleVec = std::vector<double>;
    m.def("circularDistance",       &circularDistance      <double          >, py::arg("alpha"), py::arg("beta"));
    m.def("sumSquaredCircularDist", &sumSquaredCircularDist<double, AngleVec>, py::arg("alpha"), py::arg("angles"));
    m.def("circularMean",           &circularMean          <        AngleVec>, py::arg("angles"));

    ////////////////////////////////////////////////////////////////////////////////
    // Parametrization base class
    ////////////////////////////////////////////////////////////////////////////////
    py::class_<Parametrizer>(m, "Parametrizer")
        .def("setUV",    &Parametrizer::setUV, py::arg("uv"))
        .def("uv",       &Parametrizer::uv)
        .def("mesh",     py::overload_cast<>(&Parametrizer::mesh), py::return_value_policy::reference)
        .def("jacobian", &Parametrizer::jacobian, py::arg("tri_idx"))
        .def("leftStretchAngles",                             &Parametrizer::leftStretchAngles)
        .def("perVertexLeftStretchAngles",                    &Parametrizer::perVertexLeftStretchAngles,                                        py::arg("agreementThreshold") = M_PI / 8)
        .def("perVertexAlphas",                               &Parametrizer::perVertexAlphas)
        .def("upsampledVertexLeftStretchAnglesAndMagnitudes", &Parametrizer::upsampledVertexLeftStretchAnglesAndMagnitudes, py::arg("nsubdiv"), py::arg("agreementThreshold") = M_PI / 8)
        .def("upsampledUV",                                   &Parametrizer::upsampledUV,                                   py::arg("nsubdiv"))
        .def("numFlips", &Parametrizer::numFlips)
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // Local-global solver
    ////////////////////////////////////////////////////////////////////////////////
    py::class_<LocalGlobalParametrizer, Parametrizer>(m, "LocalGlobalParametrizer")
        .def(py::init<const std::shared_ptr<Mesh> &, const UVMap &>())
        .def_property("alphaMin", [](const LocalGlobalParametrizer &lg) { return lg.alphaMin(); },
                                  [](      LocalGlobalParametrizer &lg, Real val) { return lg.setAlphaMin(val); })
        .def_property("alphaMax", [](const LocalGlobalParametrizer &lg) { return lg.alphaMax(); },
                                  [](      LocalGlobalParametrizer &lg, Real val) { return lg.setAlphaMax(val); })
        .def("energy", &LocalGlobalParametrizer::energy)
        .def("getAlphas", &LocalGlobalParametrizer::getAlphas)
        .def("runIteration", &LocalGlobalParametrizer::runIteration)
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // Laplacian regularization stencil
    ////////////////////////////////////////////////////////////////////////////////
    using DLS = DualLaplacianStencil<Mesh>;
    py::class_<DLS> pyDualLapStencil(m, "DualLaplacianStencil");

    py::enum_<DLS::Type>(pyDualLapStencil, "Type")
        .value("DualGraph",   DLS::Type::DualGraph)
        .value("DualMeshIDT", DLS::Type::DualMeshIDT)
        ;

    pyDualLapStencil
        .def(py::init<const Mesh &>(), py::arg("mesh"))
        .def_readwrite("type", &DLS::type)
        .def_readwrite("useUniformGraphWeights", &DLS::useUniformGraphWeights)
        .def("visit", [](const DLS &stencil, const size_t i, const std::function<void(size_t, size_t, Real)> &visitor) { stencil.visit(i, visitor); }, py::arg("i"), py::arg("visitor"))
        // .def("visit", &DLS::visit, py::arg("i"), py::arg("visitor"))
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // Regularized parametrization energy
    ////////////////////////////////////////////////////////////////////////////////
    py::class_<RegularizedParametrizer, Parametrizer> pyRegParam(m, "RegularizedParametrizer");

    py::enum_<RegularizedParametrizer::EnergyType>(pyRegParam, "EnergyType")
        .value("Full"               , RegularizedParametrizer::EnergyType::Full)
        .value("Fitting"            , RegularizedParametrizer::EnergyType::Fitting)
        .value("PhiRegularization"  , RegularizedParametrizer::EnergyType::PhiRegularization)
        .value("AlphaRegularization", RegularizedParametrizer::EnergyType::AlphaRegularization)
        ;

    pyRegParam
        .def(py::init<LocalGlobalParametrizer &>())

        .def(   "uvOffset", &RegularizedParametrizer::   uvOffset)
        .def(    "uOffset", &RegularizedParametrizer::    uOffset)
        .def(    "vOffset", &RegularizedParametrizer::    vOffset)
        .def(  "phiOffset", &RegularizedParametrizer::  phiOffset)
        .def(  "psiOffset", &RegularizedParametrizer::  psiOffset)
        .def("alphaOffset", &RegularizedParametrizer::alphaOffset)

        .def_property("variableAlpha", [](const RegularizedParametrizer &rsvd)           { return rsvd.   variableAlpha(); },
                                       [](      RegularizedParametrizer &rsvd, bool val) { return rsvd.setVariableAlpha(val); })
        .def_property("alphaMin",      [](const RegularizedParametrizer &rsvd)           { return rsvd.   alphaMin(); },
                                       [](      RegularizedParametrizer &rsvd, Real val) { return rsvd.setAlphaMin(val); })
        .def_property("alphaMax",      [](const RegularizedParametrizer &rsvd)           { return rsvd.   alphaMax(); },
                                       [](      RegularizedParametrizer &rsvd, Real val) { return rsvd.setAlphaMax(val); })

        .def("numVars", &RegularizedParametrizer::numVars)
        .def("getVars", &RegularizedParametrizer::getVars)
        .def("setVars", &RegularizedParametrizer::setVars)

        .def("getAlphas", &RegularizedParametrizer::getAlphas)

        .def("energy", &RegularizedParametrizer::energy)

        .def("gradient", &RegularizedParametrizer::gradient, py::arg("energyType") = RegularizedParametrizer::EnergyType::Full)

        .def("hessian", [](const RegularizedParametrizer &rparam, RegularizedParametrizer::EnergyType et) { return rparam.hessian(et); }, py::arg("energyType") = RegularizedParametrizer::EnergyType::Full)
        .def("hessianSparsityPattern", &RegularizedParametrizer::hessianSparsityPattern, py::arg("val"))

        .def_property("alphaRegW", [](const RegularizedParametrizer &rsvd)           { return rsvd.   alphaRegW(); },
                                   [](      RegularizedParametrizer &rsvd, Real val) { return rsvd.setAlphaRegW(val); })
        .def_property("alphaRegP", [](const RegularizedParametrizer &rsvd)           { return rsvd.   alphaRegP(); },
                                   [](      RegularizedParametrizer &rsvd, Real val) { return rsvd.setAlphaRegP(val); })
        .def_property(  "phiRegW", [](const RegularizedParametrizer &rsvd)           { return rsvd.     phiRegW(); },
                                   [](      RegularizedParametrizer &rsvd, Real val) { return rsvd.  setPhiRegW(val); })
        .def_property(  "phiRegP", [](const RegularizedParametrizer &rsvd)           { return rsvd.     phiRegP(); },
                                   [](      RegularizedParametrizer &rsvd, Real val) { return rsvd.  setPhiRegP(val); })
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // Regularized parametrization energy, SVD version
    ////////////////////////////////////////////////////////////////////////////////
    py::class_<RegularizedParametrizerSVD, Parametrizer> pyRegParamSVD(m, "RegularizedParametrizerSVD");

    py::enum_<RegularizedParametrizerSVD::EnergyType>(pyRegParamSVD, "EnergyType")
        .value("Full",                  RegularizedParametrizerSVD::EnergyType::Full)
        .value("Fitting",               RegularizedParametrizerSVD::EnergyType::Fitting)
        .value("PhiRegularization",     RegularizedParametrizerSVD::EnergyType::PhiRegularization)
        .value("AlphaRegularization",   RegularizedParametrizerSVD::EnergyType::AlphaRegularization)
        .value("BendingRegularization", RegularizedParametrizerSVD::EnergyType::BendingRegularization)
        ;

    pyRegParamSVD
        .def(py::init<const std::shared_ptr<Mesh> &, const UVMap &, Real, Real, bool>(), py::arg("mesh"), py::arg("uv"), py::arg("alphaMin") = 1.0, py::arg("alphaMax") = 1.0, py::arg("transformForRigidMotionConstraint") = true)
        .def(py::init<LocalGlobalParametrizer &>(), py::arg("lgparam"))

        .def(   "uvOffset", &RegularizedParametrizerSVD::uvOffset)
        .def(    "uOffset", &RegularizedParametrizerSVD:: uOffset)
        .def(    "vOffset", &RegularizedParametrizerSVD:: vOffset)

        .def_property_readonly("rigidMotionPinVars", [&](const RegularizedParametrizerSVD &rsvd) { return rsvd.rigidMotionPinVars(); })

        .def_property("alphaMin",      [](const RegularizedParametrizerSVD &rsvd)           { return rsvd.   alphaMin(); },
                                       [](      RegularizedParametrizerSVD &rsvd, Real val) { return rsvd.setAlphaMin(val); })
        .def_property("alphaMax",      [](const RegularizedParametrizerSVD &rsvd)           { return rsvd.   alphaMax(); },
                                       [](      RegularizedParametrizerSVD &rsvd, Real val) { return rsvd.setAlphaMax(val); })

        .def("numVars", &RegularizedParametrizerSVD::numVars)
        .def("getVars", &RegularizedParametrizerSVD::getVars)
        .def("setVars", &RegularizedParametrizerSVD::setVars)

        .def("getAlphas",            &RegularizedParametrizerSVD::getAlphas)
        .def("getMinSingularValues", &RegularizedParametrizerSVD::getMinSingularValues)

        .def("tubeDirections", &RegularizedParametrizerSVD::tubeDirections)

        .def("curvature3d", &RegularizedParametrizerSVD::curvature3d, py::arg("tri"), py::arg("i"))

        .def("energy", py::overload_cast<RegularizedParametrizerSVD::EnergyType>(&RegularizedParametrizerSVD::energy, py::const_), py::arg("energyType") = RegularizedParametrizerSVD::EnergyType::Full)

        .def("gradient", &RegularizedParametrizerSVD::gradient, py::arg("energyType") = RegularizedParametrizerSVD::EnergyType::Full)

        .def("hessian", [](const RegularizedParametrizerSVD &rparam, RegularizedParametrizerSVD::EnergyType et, bool projectionMask) { return rparam.hessian(et, projectionMask); }, py::arg("energyType") = RegularizedParametrizerSVD::EnergyType::Full, py::arg("projectionMask") = false)
        .def("hessianSparsityPattern", &RegularizedParametrizerSVD::hessianSparsityPattern, py::arg("val"))

        .def_property("alphaRegW", [](const RegularizedParametrizerSVD &rsvd)           { return rsvd.   alphaRegW(); },
                                   [](      RegularizedParametrizerSVD &rsvd, Real val) { return rsvd.setAlphaRegW(val); })
        .def_property("alphaRegP", [](const RegularizedParametrizerSVD &rsvd)           { return rsvd.   alphaRegP(); },
                                   [](      RegularizedParametrizerSVD &rsvd, Real val) { return rsvd.setAlphaRegP(val); })
        .def_property(  "phiRegW", [](const RegularizedParametrizerSVD &rsvd)           { return rsvd.     phiRegW(); },
                                   [](      RegularizedParametrizerSVD &rsvd, Real val) { return rsvd.  setPhiRegW(val); })
        .def_property(  "phiRegP", [](const RegularizedParametrizerSVD &rsvd)           { return rsvd.     phiRegP(); },
                                   [](      RegularizedParametrizerSVD &rsvd, Real val) { return rsvd.  setPhiRegP(val); })
        .def_property( "bendRegW", [](const RegularizedParametrizerSVD &rsvd)           { return rsvd.    bendRegW(); },
                                   [](      RegularizedParametrizerSVD &rsvd, Real val) { return rsvd. setBendRegW(val); })
        .def_property( "stretchDeviationP", [](const RegularizedParametrizerSVD &rsvd)           { return rsvd.    stretchDeviationP(); },
                                            [](      RegularizedParametrizerSVD &rsvd, Real val) { return rsvd. setStretchDeviationP(val); })
        .def_readonly("dualLaplacianStencil",         &RegularizedParametrizerSVD::dualLaplacianStencil)
        .def_readwrite("scaleInvariantFittingEnergy", &RegularizedParametrizerSVD::scaleInvariantFittingEnergy)

        .def("setAlphas", &RegularizedParametrizerSVD::setAlphas, py::arg("newAlphas")) // for debugging/analysis only!
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // Enable output redirection from Python side
    ////////////////////////////////////////////////////////////////////////////////
    py::add_ostream_redirect(m, "ostream_redirect");
}

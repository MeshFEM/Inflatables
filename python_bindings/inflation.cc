#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/MeshIO.hh>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#include <InflatableSheet.hh>
#include <SheetOptimizer.hh>
#include <TargetAttractedInflation.hh>
#include <inflation_newton.hh>
#include <InflatedSurfaceAnalysis.hh>
#include <ReducedSheetOptimizer.hh>
#include <FusingCurveSmoothness.hh>
#include <CompressionPenalty.hh>

#include <MeshFEM/../../python_bindings/MeshEntities.hh>
#include <MeshFEM/../../python_bindings/BindingUtils.hh>

#if INFLATABLES_LONG_DOUBLE
#include "extended_precision.hh"
#endif

// For CPModulation classes that provide their own serialization.
template<class CPModSubclass, size_t Enabler = sizeof(typename CPModSubclass::State)>
auto bindCPMod_impl(py::module &m, const std::string &name, int /* PREFFERRED */) {
    py::class_<CPModSubclass, CPModulation, std::shared_ptr<CPModSubclass>> pyC(m, name.c_str());
    pyC.def(py::init<>());
    addSerializationBindings<CPModSubclass>(pyC);
    return pyC;
}

// Use NOP/empty serialization for the classes that lack serializable state.
template<class CPModSubclass>
auto bindCPMod_impl(py::module &m, const std::string &name, unsigned long long /* NON-PREFFERRED */) {
    return py::class_<CPModSubclass, CPModulation, std::shared_ptr<CPModSubclass>>(m, name.c_str())
                .def(py::init<>())
                .def(py::pickle([](const CPModSubclass &cp) { return py::make_tuple(); },
                                [](py::tuple &t) { return std::make_shared<CPModSubclass>(); }))
                ;
}

template<class CPModSubclass>
auto bindCPMod(py::module &m, const std::string &name) {
    return bindCPMod_impl<CPModSubclass>(m, name,
                                         int(0) /* Hack to prefer custom serialization implementation when available */);
}

PYBIND11_MODULE(inflation, m) {
    m.doc() = "Inflation simulation";
    py::module detail_module = m.def_submodule("detail");

    py::module::import("MeshFEM");
    py::module::import("mesh");
    py::module::import("py_newton_optimizer");

    ////////////////////////////////////////////////////////////////////////////////
    // Mesh construction (for mesh type used by inflation routines)
    ////////////////////////////////////////////////////////////////////////////////
    using Mesh = InflatableSheet::Mesh;
    // WARNING: Mesh's holder type is a shared_ptr; returning a unique_ptr will lead to a dangling pointer in the current version of Pybind11
    m.def("Mesh", [](const std::string &path) { return std::shared_ptr<Mesh>(Mesh::load(path)); }, py::arg("path"));
    m.def("Mesh", [](const Eigen::MatrixX3d &V, const Eigen::MatrixX3i &F) { return std::make_shared<Mesh>(F, V); }, py::arg("V"), py::arg("F"));

    ////////////////////////////////////////////////////////////////////////////////
    // Free-standing functions
    ////////////////////////////////////////////////////////////////////////////////
    m.def("inflation_newton", &inflation_newton<         InflatableSheet>, py::arg("isheet"),               py::arg("fixedVars"), py::arg("options") = NewtonOptimizerOptions(), py::arg("callback") = nullptr);
    m.def("inflation_newton", &inflation_newton<TargetAttractedInflation>, py::arg("targetAttractedSheet"), py::arg("fixedVars"), py::arg("options") = NewtonOptimizerOptions(), py::arg("callback") = nullptr);
    m.def("get_inflation_optimizer", &get_inflation_optimizer<         InflatableSheet>, py::arg("isheet"),               py::arg("fixedVars"), py::arg("options") = NewtonOptimizerOptions(), py::arg("callback") = nullptr);
    m.def("get_inflation_optimizer", &get_inflation_optimizer<TargetAttractedInflation>, py::arg("targetAttractedSheet"), py::arg("fixedVars"), py::arg("options") = NewtonOptimizerOptions(), py::arg("callback") = nullptr);

    ////////////////////////////////////////////////////////////////////////////////
    // Inflatable Sheet
    ////////////////////////////////////////////////////////////////////////////////
    py::class_<InflatableSheet, std::shared_ptr<InflatableSheet>> pyInflatableSheet(m, "InflatableSheet");

    using IEType = InflatableSheet::EnergyType;
    py::enum_<InflatableSheet::EnergyType>(pyInflatableSheet, "EnergyType")
        .value("Full"    , IEType::Full)
        .value("Elastic" , IEType::Elastic)
        .value("Pressure", IEType::Pressure)
        ;

    // Result of vtxForVar
    py::class_<InflatableSheet::ISheetVtx>(m, "_ISheetVtx")
        .def_readonly("vi",    &InflatableSheet::ISheetVtx::vi)
        .def_readonly("sheet", &InflatableSheet::ISheetVtx::sheet)
        .def("__repr__", [](const InflatableSheet::ISheetVtx &isv) -> std::string {
                if ((isv.sheet < 1) || (isv.sheet > 3)) return "NOT FOUND";
                return ("Mesh vertex " + std::to_string(isv.vi) + " on ") +
                       ((isv.sheet == 1) ? "top" :
                       ((isv.sheet == 2) ? "bottom" : "both")) + " sheet(s)";
            })
        ;

    pyInflatableSheet
        .def(py::init<const std::shared_ptr<Mesh> &, const std::vector<bool> &>(), py::arg("mesh"), py::arg("fusedVtx") = std::vector<bool>())
        .def("mesh",    [](InflatableSheet &s) { return s.meshPtr(); })
        .def("numSheetTris", &InflatableSheet::numSheetTris)
        .def("numVars",      &InflatableSheet::numVars)
        .def("getVars",      &InflatableSheet::getVars)
        .def("setVars",      &InflatableSheet::setVars)
        .def("setIdentityDeformation", &InflatableSheet::setIdentityDeformation)
        .def("setUninflatedDeformation", &InflatableSheet::setUninflatedDeformation, py::arg("P"), py::arg("prepareRigidMotionPinConstraints") = true)
        .def("setRestVertexPositions", [](InflatableSheet &isheet, const Eigen::MatrixXd &P) {
                if (P.cols() == 2)      { isheet.setRestVertexPositions(InflatableSheet::MX2d(P.cast<InflatableSheet::Real>())); }
                else throw std::runtime_error("P must be Vx2.");
        })
        .def("getDeformedVtxPosition", &InflatableSheet::getDeformedVtxPosition, py::arg("vi"), py::arg("sheetIdx"))

        .def("setMaterial", &InflatableSheet::setMaterial, py::arg("psi"))

        .def("numWallVertices",   &InflatableSheet::numWallVertices)
        .def("wallVertices",      &InflatableSheet::wallVertices)
        .def("isWallTri",         &InflatableSheet::isWallTri)
        .def("isWallVtx",         &InflatableSheet::isWallVtx)
        .def("wallBoundaryEdges", &InflatableSheet::wallBoundaryEdges)
        .def("trueWallVertices",  &InflatableSheet::trueWallVertices)
        .def("airChannelIndices", &InflatableSheet::airChannelIndices)
        .def("fusedRegionBooleanIntersectSheetBoundary", &InflatableSheet::fusedRegionBooleanIntersectSheetBoundary)

        .def("vtxForVar", &InflatableSheet::vtxForVar)

        .def("varIdx", [](const InflatableSheet &s, size_t sheetIdx, size_t vtxIdx, size_t compIdx) {
                // The C++ version of varIdx trips assertions if the arguments are out of bounds, so we check here.
                if (sheetIdx > 1)                    throw std::runtime_error("sheetIdx out of bounds");
                if (vtxIdx > s.mesh().numVertices()) throw std::runtime_error("vtxIdx   out of bounds");
                if (compIdx  > 3)                    throw std::runtime_error("compIdx  out of bounds");
                return s.varIdx(sheetIdx, vtxIdx, compIdx);
        }, py::arg("sheetIdx"), py::arg("vtxIdx"), py::arg("compIdx") = 0)

        .def("setUseTensionFieldEnergy",     &InflatableSheet::setUseTensionFieldEnergy)
        .def("setUseHessianProjectedEnergy", &InflatableSheet::setUseHessianProjectedEnergy)
        .def("setUseTensionFieldEnergy",     &InflatableSheet::setUseTensionFieldEnergy)
        .def("usingTensionFieldEnergy",      &InflatableSheet::usingTensionFieldEnergy,     py::arg("sheetTriIdx"))
        .def("usingHessianProjectedEnergy",  [](const InflatableSheet &is, size_t i) { return is.usingHessianProjectedEnergy(i); }, py::arg("sheetTriIdx"))
        .def("usingHessianProjectedEnergy",  [](const InflatableSheet &is          ) { return is.usingHessianProjectedEnergy() ; })
        .def("setRelaxedStiffnessEpsilon",   &InflatableSheet::setRelaxedStiffnessEpsilon)

        .def("tensionStateHistogram",        &InflatableSheet::tensionStateHistogram)

        .def("disableFusedRegionTensionFieldTheory", &InflatableSheet::disableFusedRegionTensionFieldTheory, py::arg("useHessianProjection"))

        .def_property_readonly("rigidMotionPinVars", [&](const InflatableSheet &isheet) { return isheet.rigidMotionPinVars(); })

        .def_property("pressure", &InflatableSheet::getPressure, &InflatableSheet::setPressure)

        .def("volume",   &InflatableSheet::volume)
        .def_property("referenceVolume", &InflatableSheet::referenceVolume, &InflatableSheet::setReferenceVolume)

        .def_property("thickness", &InflatableSheet::getThickness, &InflatableSheet::setThickness)
        .def_property("youngModulus", &InflatableSheet::getYoungModulus, &InflatableSheet::setYoungModulus)

        .def("energy",   &InflatableSheet::energy  , py::arg("energyType") = IEType::Full)
        .def("gradient", &InflatableSheet::gradient, py::arg("energyType") = IEType::Full)

        .def("hessianSparsityPattern", &InflatableSheet::hessianSparsityPattern, py::arg("val") = 1.0)
        .def("hessian",  py::overload_cast<IEType>(&InflatableSheet::hessian, py::const_), py::arg("energyType") = IEType::Full)

        .def("deformationGradient3D", &InflatableSheet::deformationGradient3D, py::arg("sheet_tri_idx"))

        .def("cauchyGreenDeformationTensors", &InflatableSheet::cauchyGreenDeformationTensors)
        .def("triEnergyDensities",            &InflatableSheet::triEnergyDensities, py::return_value_policy::reference)
        .def(  "deformedAreas",               &InflatableSheet::  deformedAreas)
        .def("undeformedAreas",               &InflatableSheet::undeformedAreas)
        // .def("tangentElasticityTensors",      &InflatableSheet::tangentElasticityTensors)

        .def("writeDebugMesh",    &InflatableSheet::writeDebugMesh)

        .def("getFusingPolylines", &InflatableSheet::getFusingPolylines, "Get the boundary (first) and interior (second) fusing curves as closed polylines, represented as a sequence of vertex indices.")


        .def("wallVertexPositionsFromMesh", &InflatableSheet::wallVertexPositionsFromMesh, py::arg("mesh"))
        .def("restWallVertexPositions",     &InflatableSheet::restWallVertexPositions)
        .def("deformedWallVertexPositions", &InflatableSheet::deformedWallVertexPositions)

        .def("visualizationMesh", &InflatableSheet::visualizationMesh, py::arg("duplicateFusedTris") = false)
        .def("visualizationField", &InflatableSheet::visualizationField, py::arg("field"), py::arg("duplicateFusedTris") = false)
        // Interface for MeshFEM's viewer
        .def("visualizationGeometry", [](const InflatableSheet &is, double normalCreaseAngle) { return getVisualizationGeometry(*is.visualizationMesh(), normalCreaseAngle); }, py::arg("normalCreaseAngle") =  M_PI)

        // Serialization
        .def(py::pickle([](const InflatableSheet &is) {
                            return py::make_tuple(is.mesh(),
                                                  is.fusedVtx(),
                                                  is.getMaterialConfiguration(),
                                                  is.rigidMotionPinVars(),
                                                  is.getVars(),
                                                  is.getPressure(),
                                                  is.referenceVolume(),
                                                  is.getThickness(),
                                                  is.getYoungModulus());
                        },
                        [](const py::tuple &t) {
                            if (!((t.size() == 7) || (t.size() == 9)))  throw std::runtime_error("Invalid pickled state!");
                            auto mesh = t[0].cast<std::shared_ptr<Mesh>>();
                            auto iwv  = t[1].cast<std::vector<bool>>();
                            std::shared_ptr<InflatableSheet> is = std::make_shared<InflatableSheet>(mesh, iwv);
                            is->setRigidMotionPinVars(t[3].cast<std::array<size_t, 6>>());
                            is->setVars(t[4].cast<InflatableSheet::VXd>());
                            is->setPressure(t[5].cast<Real>());
                            is->setReferenceVolume(t[6].cast<Real>());
                            if (t.size() == 9) {
                                is->setThickness(t[7].cast<Real>());
                                is->setYoungModulus(t[8].cast<Real>());
                            }

                            // Must happen after setThickness/setYoungModulus in case the user directly modified the energy density stiffness parameters
                            is->applyMaterialConfiguration(t[2].cast<std::vector<InflatableSheet::MaterialConfiguration>>());
                            return is;
                        }));
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // Optimization
    ////////////////////////////////////////////////////////////////////////////////
    py::class_<SheetOptimizer> pySheetOptimizer(m, "SheetOptimizer");

    using SOEType = SheetOptimizer::EnergyType;
    py::enum_<SOEType>(pySheetOptimizer, "EnergyType")
        .value("Full",            SOEType::Full)
        .value("Simulation",      SOEType::Simulation)
        .value("Fitting",         SOEType::Fitting)
        .value("Smoothing",       SOEType::Smoothing)
        .value("CollapseBarrier", SOEType::CollapseBarrier)
        ;

    py::class_<TargetSurfaceFitter, std::shared_ptr<TargetSurfaceFitter>>(m, "TargetSurfaceFitter")
        .def(py::init<const TargetSurfaceMesh &>(), py::arg("targetMesh"))
        .def("energy",                               &TargetSurfaceFitter::energy)
        .def("gradient",                             &TargetSurfaceFitter::gradient)
        .def("updateClosestPoints",                  &TargetSurfaceFitter::updateClosestPoints, py::arg("queryPts"), py::arg("isBoundary"))
        .def("vtx_hessian",                          &TargetSurfaceFitter::vtx_hessian, py::arg("i"), "Hessian with respect to query point i")
        .def_readonly ("queryPoints",                &TargetSurfaceFitter::queryPoints)
        .def_readwrite("closestSurfPts",             &TargetSurfaceFitter::closestSurfPts) // Warning: manually setting this does not update the closest point sensivities/item indices!
        .def_readonly ("closestSurfPtSensitivities", &TargetSurfaceFitter::closestSurfPtSensitivities)
        .def_readonly ("closestSurfItems",           &TargetSurfaceFitter::closestSurfItems)
        .def_property_readonly ("targetSurfaceV",    &TargetSurfaceFitter::getTargetSurfaceV)
        .def_property_readonly ("targetSurfaceF",    &TargetSurfaceFitter::getTargetSurfaceF)
        .def_readwrite("holdClosestPointsFixed",     &TargetSurfaceFitter::holdClosestPointsFixed)
        .def("setQueryPtWeights",                    &TargetSurfaceFitter::setQueryPtWeights)
        .def(py::pickle(&TargetSurfaceFitter::serialize, &TargetSurfaceFitter::deserialize))
        ;

    using CB = ReducedSheetOptimizer::CB;
    using CPE = CB::CPE;
    py::class_<SVDSensitivity>(m, "SVDSensitivity")
        .def("Sigma", &SVDSensitivity::Sigma)
        .def("U", &SVDSensitivity::U)
        .def("V", &SVDSensitivity::V)
        ;

    py::class_<CPE>(m, "CollapsePreventionEnergy")
        .def("energy", &CPE::energy)
        .def("svd",    &CPE::svd,    py::return_value_policy::reference)
        .def("det",    &CPE::det,    py::return_value_policy::reference)
        .def_readwrite("applyStretchBarrier",      &CPE::applyStretchBarrier)
        .def_readwrite("stretchBarrierActivation", &CPE::stretchBarrierActivation)
        .def_readwrite("stretchBarrierLimit",      &CPE::stretchBarrierLimit)
        .def_property("activationThreshold", &CPE::activationThreshold, &CPE::setActivationThreshold)
        ;

    using PyCB = py::class_<CB, std::shared_ptr<CB>>;
    PyCB pyCB(m, "CollapseBarrier");
    pyCB.def("energy",                   &CB::energy)
        .def("mesh",                     &CB::mesh,                                        py::return_value_policy::reference)
        .def("collapsePreventionEnergy", &CB::collapsePreventionEnergy, py::arg("triIdx"), py::return_value_policy::reference)
        .def("setActivationThresholds",       &CB::setActivationThresholds,       py::arg("activationThresholds"))
        .def("setApplyStretchBarriers",       &CB::setApplyStretchBarriers,       py::arg("applyStretchBarriers"))
        .def("setStretchBarrierActiviations", &CB::setStretchBarrierActiviations, py::arg("activations"))
        .def("setStretchBarrierLimits",       &CB::setStretchBarrierLimits,       py::arg("limits"))
        ;
    addSerializationBindings<CB, PyCB, CB::StateBackwardsCompat>(pyCB);

    using WPI = WallPositionInterpolator<InflatableSheet::Real>;
    py::class_<WPI, std::shared_ptr<WPI>>(m, "WallPositionInterpolator")
        .def(py::init<const InflatableSheet &>(), py::arg("sheet"))
        .def("interpolate", [](const WPI &wpi, const WPI::MX2d &pos) {
                    WPI::MX2d result(wpi.numVertices(), 2);
                    wpi.interpolate(pos, result);
                    return result;
                }, py::arg("wallPositions"))
        .def("adjoint", [](const WPI &wpi, const WPI::MX2d &vtxDiffForm) {
                    WPI::MX2d result(wpi.numWallVertices(), 2);
                    wpi.adjoint(vtxDiffForm, result);
                    return result;
                }, py::arg("vtxDiffForm"))
        .def_property_readonly("initialRestLaplacian", &WPI::initialRestLaplacian)
        .def_property_readonly("isBoundary",           &WPI::isBoundary)
        .def(py::pickle(&WPI::serialize, &WPI::deserialize))
        ;

    pySheetOptimizer
        .def(py::init<std::shared_ptr<InflatableSheet>, const Mesh &>(), py::arg("sheet"), py::arg("targetSurface"))
        .def("mesh",                py::overload_cast<>(&SheetOptimizer::mesh),                py::return_value_policy::reference)
        .def("sheet",               py::overload_cast<>(&SheetOptimizer::sheet),               py::return_value_policy::reference)
        .def("targetSurfaceFitter", py::overload_cast<>(&SheetOptimizer::targetSurfaceFitter), py::return_value_policy::reference)

        .def("numVars",            &SheetOptimizer::numVars)
        .def("numEquilibriumVars", &SheetOptimizer::numEquilibriumVars)
        .def("numDesignVars",      &SheetOptimizer::numDesignVars)

        .def("getVars", &SheetOptimizer::getVars)
        .def("setVars", &SheetOptimizer::setVars)

        .def("energy",   &SheetOptimizer::energy  , py::arg("energyType") = SOEType::Full)
        .def("gradient", &SheetOptimizer::gradient, py::arg("energyType") = SOEType::Full)

        .def("hessianSparsityPattern", &SheetOptimizer::hessianSparsityPattern, py::arg("val") = 1.0)
        .def("hessian",  py::overload_cast<SOEType>(&SheetOptimizer::hessian, py::const_), py::arg("energyType") = SOEType::Full)
        ;

    py::class_<Nondimensionalization>(m, "Nondimensionalization")
        .def_readwrite("length",         &Nondimensionalization::length)
        .def_readwrite("sheetArea",      &Nondimensionalization::sheetArea)
        .def_readwrite("sheetThickness", &Nondimensionalization::sheetThickness)
        .def_readwrite("fusedArea",      &Nondimensionalization::fusedArea)
        .def_readwrite("wallWidth",      &Nondimensionalization::wallWidth)
        .def_readwrite("youngsModulus",  &Nondimensionalization::youngsModulus)
        .def("potentialEnergyScale",     &Nondimensionalization::potentialEnergyScale)
        .def("fittingEnergyScale",       &Nondimensionalization::fittingEnergyScale)
        .def("collapseBarrierScale",     &Nondimensionalization::collapseBarrierScale)
        .def("smoothingScale",           &Nondimensionalization::smoothingScale)
        .def("dirichletSmoothingScale",  &Nondimensionalization::dirichletSmoothingScale)
        .def("restVarScale",             &Nondimensionalization::restVarScale)
        .def("equilibriumVarScale",      &Nondimensionalization::equilibriumVarScale)
        .def(py::pickle(&Nondimensionalization::serialize, &Nondimensionalization::deserialize))
        ;

    using TAI = TargetAttractedInflation;
    using PyTAI = py::class_<TAI, std::shared_ptr<TAI>>;
    PyTAI pyTAI(m, "TargetAttractedInflation");

    using TAIEType = TAI::EnergyType;
    py::enum_<TAIEType>(pyTAI, "EnergyType")
        .value("Full",       TAIEType::Full)
        .value("Simulation", TAIEType::Simulation)
        .value("Fitting",    TAIEType::Fitting)
        ;

    pyTAI
        .def(py::init<std::shared_ptr<InflatableSheet>, const Mesh &>(), py::arg("sheet"), py::arg("targetSurface"))
        .def("mesh",                [](TAI &tai) { return tai.sheet().meshPtr(); })
        .def("sheet",               [](TAI &tai) { return tai.sheetPtr(); })
        .def("targetSurfaceFitter", [](TAI &tai) { return tai.targetSurfaceFitterPtr(); })

        .def_readwrite("fittingWeight", &TAI::fittingWeight)

        .def("numVars", &TAI::numVars)
        .def("getVars", &TAI::getVars)
        .def("setVars", &TAI::setVars)

        .def("energy",   &TAI::energy  , py::arg("energyType") = TAIEType::Full)
        .def("gradient", &TAI::gradient, py::arg("energyType") = TAIEType::Full)

        .def("hessianSparsityPattern", &TAI::hessianSparsityPattern, py::arg("val") = 1.0)
        .def("hessian",  py::overload_cast<TAIEType>(&TAI::hessian, py::const_), py::arg("energyType") = TAIEType::Full)

        .def_readonly("nondimensionalization", &TAI::nondimensionalization)

        .def("cloneForRemeshedSheet", &TAI::cloneForRemeshedSheet, py::arg("sheet"))
        ;
    addSerializationBindings<TAI, PyTAI, TAI::StateBackwardsCompat>(pyTAI);

    py::class_<FusingCurveSmoothness, std::shared_ptr<FusingCurveSmoothness>> pyFusingCurveSmoothness(m, "FusingCurveSmoothness");
    pyFusingCurveSmoothness
        .def(py::init<const InflatableSheet &>(), py::arg("isheet"))
        .def("energy",   &FusingCurveSmoothness::energy,   py::arg("currMesh"), py::arg("origMesh"), py::arg("nondimensionalization"))
        .def("gradient", &FusingCurveSmoothness::gradient, py::arg("currMesh"), py::arg("origMesh"), py::arg("nondimensionalization"))
        .def_readonly("boundaryLoops", &FusingCurveSmoothness::boundaryLoops)
        .def_readonly("wallCurves",    &FusingCurveSmoothness::wallCurves)

        .def_readwrite("dirichletWeight",            &FusingCurveSmoothness::dirichletWeight)
        .def_readwrite("laplacianWeight",            &FusingCurveSmoothness::laplacianWeight)
        .def_readwrite("curvatureWeight",            &FusingCurveSmoothness::curvatureWeight)
        .def_readwrite("lengthScaleSmoothingWeight", &FusingCurveSmoothness::lengthScaleSmoothingWeight)
        .def_readwrite("interiorWeight",             &FusingCurveSmoothness::interiorWeight)
        .def_readwrite("boundaryWeight",             &FusingCurveSmoothness::boundaryWeight)
        .def_readwrite("curvatureSmoothingActivationThreshold", &FusingCurveSmoothness::curvatureSmoothingActivationThreshold)
        .def(py::pickle(&FusingCurveSmoothness::serialize, &FusingCurveSmoothness::deserialize))
        ;

    py::class_<CPModulation, std::shared_ptr<CPModulation>>(m, "CPModulation")
        .def("j",  &CPModulation::j)
        .def("dj", &CPModulation::dj)
        ;
    bindCPMod<CPModulationIdentity>(m, "CPModulationIdentity");
    bindCPMod<CPModulationTanh    >(m, "CPModulationTanh");
    bindCPMod<CPModulationPthRoot >(m, "CPModulationPthRoot")
        .def("set_p",         &CPModulationPthRoot::set_p)
        .def_readwrite("p",   &CPModulationPthRoot::p)
        .def_readwrite("eps", &CPModulationPthRoot::eps)
        ;
    bindCPMod<CPModulationCustom  >(m, "CPModulationCustom") // Unfortunately pickling this will fail since python cannot pickle PyCapsule objects. But at least the user gets an error...
        .def_readwrite("j_func",  &CPModulationCustom::j_func)
        .def_readwrite("dj_func", &CPModulationCustom::dj_func)
        ;

    using CP = CompressionPenalty;
    py::class_<CP, std::shared_ptr<CP>>(m, "CompressionPenalty")
        .def(py::init<std::shared_ptr<const InflatableSheet>>(), py::arg("sheet"))
        .def("J",     &CP::J)
        .def("dJ_dx", &CP::dJ_dx)
        .def("dJ_dX", &CP::dJ_dX)
        .def("sheet", &CP::sheet, py::return_value_policy::reference)
        .def_readwrite("includeSheetTri", &CP::includeSheetTri, py::return_value_policy::reference)
        .def_readwrite("Etft_weight",     &CP::Etft_weight)
        .def_readwrite("modulation",      &CP::modulation)
        .def(py::pickle(&CP::serialize, &CP::deserialize))
        ;

    py::class_<ReducedSheetOptimizer> pyReducedSheetOptimizer(m, "ReducedSheetOptimizer");

    using RSOEType = ReducedSheetOptimizer::EnergyType;
    py::enum_<RSOEType>(pyReducedSheetOptimizer, "EnergyType")
        .value("Full",               RSOEType::Full)
        .value("Fitting",            RSOEType::Fitting)
        .value("CollapseBarrier",    RSOEType::CollapseBarrier)
        .value("Smoothing",          RSOEType::Smoothing)
        .value("CompressionPenalty", RSOEType::CompressionPenalty)
        ;

    pyReducedSheetOptimizer
        .def(py::init<std::shared_ptr<TargetAttractedInflation>, const std::vector<size_t> &, const NewtonOptimizerOptions &, double, ReducedSheetOptimizer::VXd, const Mesh *>(),
                py::arg("targetAttractedInflation"), py::arg("fixedVars") = std::vector<size_t>(), py::arg("eopts") = NewtonOptimizerOptions(), py::arg("detActivationThreshold") = 0.9,
                py::arg("initialVars") = ReducedSheetOptimizer::VXd(),
                py::arg("originalDesignMesh") = nullptr)
        .def("mesh",                     py::overload_cast<>(&ReducedSheetOptimizer::mesh),                py::return_value_policy::reference)
        .def("originalMesh",                                 &ReducedSheetOptimizer::originalMesh,         py::return_value_policy::reference)
        .def("sheet",                    py::overload_cast<>(&ReducedSheetOptimizer::sheet),               py::return_value_policy::reference)
        .def("targetSurfaceFitter",      py::overload_cast<>(&ReducedSheetOptimizer::targetSurfaceFitter), py::return_value_policy::reference)
        .def("targetAttractedInflation", &ReducedSheetOptimizer::targetAttractedInflationPtr)

        .def("numVars", &ReducedSheetOptimizer::numVars)
        .def("getVars", &ReducedSheetOptimizer::getVars)
        .def("setVars", &ReducedSheetOptimizer::setVars, py::arg("vars"), py::arg("bailEarlyOnCollapse") = false, py::arg("bailEnergyThreshold") = std::numeric_limits<float>::max())

        .def("getCommittedVars",          &ReducedSheetOptimizer::getCommittedVars)
        .def("getCommittedEquilibrium",   &ReducedSheetOptimizer::getCommittedEquilibrium)
        .def("getCommittedRestPositions", &ReducedSheetOptimizer::getCommittedRestPositions)

        .def("varsForDesignMesh", &ReducedSheetOptimizer::varsForDesignMesh, py::arg("alteredMesh"))

        .def("forceEquilibriumUpdate", &ReducedSheetOptimizer::forceEquilibriumUpdate, "Force a re-solve of the line search iterate's equilbrium (e.g., to inflate into a different local minimum after perturbing targetAttractedInflation)")
        .def("fixedEquilibriumVars",   &ReducedSheetOptimizer::fixedEquilibriumVars)
        .def("forceAdjointStateUpdate", &ReducedSheetOptimizer::forceAdjointStateUpdate, "Force a recalculation of the adjoint states (e.g., one of the x-dependent objective terms is reconfigured)")

        .def("getEquilibriumSolver", &ReducedSheetOptimizer::getEquilibriumSolver, py::return_value_policy::reference)

        .def("energy",   &ReducedSheetOptimizer::energy,   py::arg("etype") = RSOEType::Full)
        .def("gradient", &ReducedSheetOptimizer::gradient, py::arg("etype") = RSOEType::Full)

        .def("apply_d2E_dxdX",               &ReducedSheetOptimizer::apply_d2E_dxdX,               py::arg("delta_x"))
        .def("apply_d2E_dxdX_unaccelerated", &ReducedSheetOptimizer::apply_d2E_dxdX_unaccelerated, py::arg("delta_x"))
        .def("adjointFittingState",          &ReducedSheetOptimizer::adjointFittingState)
        .def("commitDesign",                 &ReducedSheetOptimizer::commitDesign)

        .def("collapseBarrier",                           &ReducedSheetOptimizer::collapseBarrier,        py::return_value_policy::reference)
        .def("fusingCurveSmoothness", py::overload_cast<>(&ReducedSheetOptimizer::fusingCurveSmoothness), py::return_value_policy::reference)

        .def_property_readonly("compressionPenalty", py::overload_cast<>(&ReducedSheetOptimizer::compressionPenalty), py::return_value_policy::reference)

        .def("wallPositionInterpolator",                  &ReducedSheetOptimizer::wallPositionInterpolator)

        .def_readwrite("useFirstOrderPrediction",  &ReducedSheetOptimizer::useFirstOrderPrediction)
        .def_readwrite("compressionPenaltyWeight", &ReducedSheetOptimizer::compressionPenaltyWeight)

        .def("cloneForNewTAIAndFixedVars", &ReducedSheetOptimizer::cloneForNewTAIAndFixedVars, py::arg("tai"), py::arg("fv"))
        ;
    addSerializationBindings<ReducedSheetOptimizer, py::class_<ReducedSheetOptimizer>, ReducedSheetOptimizer::StateBackwardsCompat>(pyReducedSheetOptimizer);

    ////////////////////////////////////////////////////////////////////////////////
    // Analysis
    ////////////////////////////////////////////////////////////////////////////////
    py::class_<CurvatureInfo>(m, "CurvatureInfo")
        .def(py::init<const Eigen::MatrixXd &, const Eigen::MatrixXi &>(), py::arg("V"), py::arg("F"))
        .def_readonly("d_1", &CurvatureInfo::d_1)
        .def_readonly("d_2", &CurvatureInfo::d_2)
        .def_readonly("kappa_1", &CurvatureInfo::kappa_1)
        .def_readonly("kappa_2", &CurvatureInfo::kappa_2)
        .def("meanCurvature", &CurvatureInfo::meanCurvature)
        .def("gaussianCurvature", &CurvatureInfo::gaussianCurvature)
        ;

    auto isa = py::class_<InflatedSurfaceAnalysis>(m, "InflatedSurfaceAnalysis");

    using MI = InflatedSurfaceAnalysis::MetricInfo;
    py::class_<MI>(isa, "MetricInfo")
        .def_readonly("sigma_1", &MI::sigma_1)
        .def_readonly("sigma_2", &MI::sigma_2)
        .def_readonly("left_stretch", &MI::left_stretch)
        .def_readonly("right_stretch", &MI::right_stretch)
        ;

    isa.def(py::init<const InflatableSheet &>(), py::arg("sheet"))
       .def("mesh", &InflatedSurfaceAnalysis::mesh, py::return_value_policy::reference)
       .def("curvature", &InflatedSurfaceAnalysis::curvature)
       .def("metric", &InflatedSurfaceAnalysis::metric)
       .def("inflatedSurface", &InflatedSurfaceAnalysis::inflatedSurface)
       .def_readonly("inflatedPositions", &InflatedSurfaceAnalysis::inflatedPositions)
       ;

    ////////////////////////////////////////////////////////////////////////////////
    // Energy Densities
    ////////////////////////////////////////////////////////////////////////////////
    using OTFE = OptionalTensionFieldEnergy<InflatableSheet::Real>;
    using OTFE_M2d = OTFE::M2d;
    py::class_<OTFE>(m, "OptionalTensionFieldEnergy")
        .def(py::init<const OTFE_M2d &>())
        .def(  "energy",         &OTFE::  energy)
        .def( "denergy",         py::overload_cast<const OTFE_M2d &>(&OTFE::denergy, py::const_), py::arg("dC"))
        .def( "denergy",         py::overload_cast<                >(&OTFE::denergy, py::const_))
        .def("d2energy",         &OTFE::d2energy)
        .def("tensionState",     &OTFE::tensionState)
        .def("eigSensitivities", &OTFE::eigSensitivities)
        .def("setMatrix", [](OTFE &tfe, OTFE_M2d &mat) { tfe.setMatrix(mat); })

        .def("setEigs",   &OTFE::setEigs,   py::arg("l1"), py::arg("l2"))
        .def("psi",       &OTFE::psi)
        .def("dpsi_dl",   &OTFE::dpsi_dl)
        .def("d2psi_dl2", &OTFE::d2psi_dl2)

        .def_readwrite("useTensionField", &OTFE::useTensionField)
        ;

    // Note: we cannot directly bind pointers to methods/memebers inherited from OptionalTensionFieldEnergy
    // (hence all the lambda functions).
    using OTFEJB = EnergyDensityFBasedFromCBased<OptionalTensionFieldEnergy<InflatableSheet::Real>, 3>;
    using OTFEJB_M32d = typename OTFEJB::Matrix;
    py::class_<OTFEJB>(m, "OptionalTFEJacobianBased")
        .def(py::init<>())
        .def("energy",           [](const OTFEJB &t) { return t.energy(); })
        .def("denergy",          [](const OTFEJB &t) { return t.denergy(); })
        .def("PK2Stress",        [](const OTFEJB &t) { return t.PK2Stress(); })
        .def("tensionState",     [](const OTFEJB &t) { return t.tensionState(); })
        .def("eigSensitivities", [](const OTFEJB &t) { return t.eigSensitivities(); })
        .def("setF", [](OTFEJB &tfe, const OTFEJB_M32d &mat) { tfe.setDeformationGradient(mat); })
        .def("delta_denergy",    [](const OTFEJB &psi, const OTFEJB_M32d &dF) { return psi.delta_denergy(dF); }, py::arg("dF"))
        .def_property("useTensionField", [](const OTFEJB &t) { return t.useTensionField; }, [](OTFEJB &t, bool yesno) { return t.useTensionField = yesno; })
        .def_property("stiffness", [](const OTFEJB &t) { return t.stiffness(); }, [](OTFEJB &t, Real val) { return t.setStiffness(val); })
        .def_property("relaxedStiffnessEpsilon", [](const OTFEJB &t) { return t.getRelaxedStiffnessEpsilon(); }, [](OTFEJB &t, Real val) { return t.setRelaxedStiffnessEpsilon(val); })
        ;

    using IBE = IncompressibleBalloonEnergy<InflatableSheet::Real>;
    py::class_<IBE>(m, "IncompressibleBalloonEnergy")
        .def(py::init<>())
        .def("setMatrix", [](IBE &psi, const OTFE_M2d &C) { psi.setMatrix(C); })
        .def("energy",    &IBE::energy)
        .def("denergy",   [](const IBE &psi) { return psi.denergy(); })
        .def("delta_denergy",            [](const IBE &psi, const OTFE_M2d &dC) { return psi.delta_denergy(dC); })
        .def("delta_denergy_undeformed", [](const IBE &psi, const OTFE_M2d &dC) { return psi.delta_denergy_undeformed(dC); })
        ;

    using IBEWHP = IncompressibleBalloonEnergyWithHessProjection<InflatableSheet::Real>;
    using IBEWHP_M32d = IBEWHP::M32d;

    py::class_<IBEWHP>(m, "IncompressibleBalloonEnergyWithHessProjection")
        .def(py::init<const IBEWHP_M32d &>())
        .def("setF", [&](IBEWHP &psi, const IBEWHP_M32d &F) { psi.setF(F); })
        .def("energy", &IBEWHP::energy)
        .def("denergy", [](const IBEWHP &ibewhp                       ) { return ibewhp.denergy(  ); })
        .def("denergy", [](const IBEWHP &ibewhp, const IBEWHP_M32d &dF) { return ibewhp.denergy(dF); }, py::arg("dF"))
        .def("delta_denergy", &IBEWHP::delta_denergy<IBEWHP_M32d>, py::arg("dF"))
        .def("d2energy",      &IBEWHP::     d2energy<IBEWHP_M32d>, py::arg("dF_a"), py::arg("dF_b"))
        .def_property("stiffness", [](const IBEWHP &e) { return e.stiffness; }, [](IBEWHP &e, Real val) { return e.stiffness = val; })
        ;

    using ES = EigSensitivity<InflatableSheet::Real>;
    py::class_<ES>(m, "EigSensitivity")
        .def(py::init<const Eigen::Matrix2d &>())
        .def("Q",        &ES::Q)
        .def("Lambda",   &ES::Lambda)
        .def("dLambda",  py::overload_cast<                      size_t>(&ES::dLambda, py::const_), py::arg("i"))
        .def("dLambda",  py::overload_cast<const InflatableSheet::M2d &>(&ES::dLambda, py::const_), py::arg("dA"))
        .def("d2Lambda", &ES::d2Lambda)
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // Enable output redirection from Python side
    ////////////////////////////////////////////////////////////////////////////////
    py::add_ostream_redirect(m, "ostream_redirect");
}

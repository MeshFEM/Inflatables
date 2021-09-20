////////////////////////////////////////////////////////////////////////////////
// TargetAttractedInflation.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Inflation simulation with additional springs tugging the wall vertices
//  toward the target surface. Note that we no longer need constraints to pin
//  down the global rigid motion of the surface.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  07/21/2019 15:47:04
////////////////////////////////////////////////////////////////////////////////
#ifndef TARGETATTRACTEDINFLATION_HH
#define TARGETATTRACTEDINFLATION_HH

#include "InflatableSheet.hh"
#include "TargetSurfaceFitter.hh"
#include "Nondimensionalization.hh"

#include <memory>

struct TargetAttractedInflation {
    enum class EnergyType { Full, Simulation, Fitting };
    using Mesh = InflatableSheet::Mesh;
    using Real = InflatableSheet::Real;

    using  V2d = InflatableSheet:: V2d;
    using  VXd = InflatableSheet:: VXd;
    using  M2d = InflatableSheet:: M2d;
    using  M3d = InflatableSheet:: M3d;
    using M23d = InflatableSheet::M23d;
    using M32d = InflatableSheet::M32d;
    using VSFJ = VectorizedShapeFunctionJacobian<3, V2d>;

    TargetAttractedInflation(std::shared_ptr<InflatableSheet> s, const Mesh &targetSurface);

          InflatableSheet &sheet()       { return *m_sheet; }
    const InflatableSheet &sheet() const { return *m_sheet; }

    std::shared_ptr<      InflatableSheet> sheetPtr()       { return m_sheet; }
    std::shared_ptr<const InflatableSheet> sheetPtr() const { return m_sheet; }

          Mesh &mesh()       { return sheet().mesh(); }
    const Mesh &mesh() const { return sheet().mesh(); }

          TargetSurfaceFitter &targetSurfaceFitter()       { return *m_targetSurfaceFitter; }
    const TargetSurfaceFitter &targetSurfaceFitter() const { return *m_targetSurfaceFitter; }

    std::shared_ptr<      TargetSurfaceFitter> targetSurfaceFitterPtr()       { return m_targetSurfaceFitter; }
    std::shared_ptr<const TargetSurfaceFitter> targetSurfaceFitterPtr() const { return m_targetSurfaceFitter; }

    size_t numVars() const { return sheet().numVars(); }
    VXd    getVars() const { return sheet().getVars() / nondimensionalization.equilibriumVarScale(); }

    void setVars(const VXd &vars) {
        auto &s = sheet();
        s.setVars(vars * nondimensionalization.equilibriumVarScale());
        m_targetSurfaceFitter->updateClosestPoints(s.deformedWallVertexPositions(), m_wallVtxOnBoundary);
    }

    Real energy(EnergyType etype = EnergyType::Full) const;
    VXd  gradient(EnergyType etype = EnergyType::Full) const;

    // Gradient of the unweighted target attraction energy
    // for reuse in sensitivity analysis for the target-fitting.
    // **With respect to the rescaled equilibrium variables!**
    VXd gradUnweightedTargetFit() const;

    size_t hessianNNZ() const { return hessianSparsityPattern().nz; } // TODO: predict without constructing
    SuiteSparseMatrix hessianSparsityPattern(Real val = 0.0) const;
    void              hessian(SuiteSparseMatrix &H, EnergyType etype = EnergyType::Full) const; // accumulate Hessian to H
    SuiteSparseMatrix hessian(                      EnergyType etype = EnergyType::Full) const; // construct and return Hessian

    std::shared_ptr<TargetAttractedInflation> cloneForRemeshedSheet(std::shared_ptr<InflatableSheet> isheet) const {
        if (isheet->wallVertices().size() != m_wallVtxOnBoundary.size()) {
            throw std::runtime_error("Remeshed sheet must have the same number of wall vertices (only the tubes should be remeshed)");
        }

        auto state = serialize(*this);
        std::get<0>(state) = isheet;                      // Replace the sheet with the remeshed sheet
        std::get<1>(state) = std::get<1>(state)->clone(); // Don't have the clone share our target surface

        return deserialize(state);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Serialization + cloning support (for pickling)
    // Must return std::shared_ptr to avoid memory bugs in pybind11 that
    // occur when mixing unique/shared holder types
    ////////////////////////////////////////////////////////////////////////////
    using State                = std::tuple<std::shared_ptr<InflatableSheet>, std::shared_ptr<TargetSurfaceFitter>, Real, Nondimensionalization, std::vector<bool>>;
    using StateBackwardsCompat = std::tuple<std::shared_ptr<InflatableSheet>, std::shared_ptr<TargetSurfaceFitter>, Real,                        std::vector<bool>>;
    static State serialize(const TargetAttractedInflation &tai)                      { return std::make_tuple(tai.m_sheet, tai.m_targetSurfaceFitter->clone(), tai.fittingWeight, tai.nondimensionalization, tai.m_wallVtxOnBoundary); }
    static std::shared_ptr<TargetAttractedInflation> deserialize(const State                &state) { return std::shared_ptr<TargetAttractedInflation>(new TargetAttractedInflation(std::get<0>(state), std::get<1>(state), std::get<2>(state), std::get<3>(state), std::get<4>(state))); } // can't use make_shared since constructor is private
    static std::shared_ptr<TargetAttractedInflation> deserialize(const StateBackwardsCompat &state) { return std::shared_ptr<TargetAttractedInflation>(new TargetAttractedInflation(std::get<0>(state), std::get<1>(state), std::get<2>(state), std::get<3>(state))); } // omit nondimensionalization

    Real fittingWeight = 1.0;
    Nondimensionalization nondimensionalization;

private:
    std::shared_ptr<InflatableSheet>     m_sheet;
    std::shared_ptr<TargetSurfaceFitter> m_targetSurfaceFitter;
    std::vector<bool>                    m_wallVtxOnBoundary;

    // Constructors for deserialization
    TargetAttractedInflation(std::shared_ptr<InflatableSheet> s, std::shared_ptr<TargetSurfaceFitter> tsf, Real fw, std::vector<bool> wallVtxOnBoundary)
        : fittingWeight(fw), nondimensionalization(*s), m_sheet(s), m_targetSurfaceFitter(std::move(tsf)), m_wallVtxOnBoundary(wallVtxOnBoundary) { }
    TargetAttractedInflation(std::shared_ptr<InflatableSheet> s, std::shared_ptr<TargetSurfaceFitter> tsf, Real fw, const Nondimensionalization &n, std::vector<bool> wallVtxOnBoundary)
        : TargetAttractedInflation(s, tsf, fw, wallVtxOnBoundary) { nondimensionalization = n; }
};

#endif /* end of include guard: TARGETATTRACTEDINFLATION_HH */

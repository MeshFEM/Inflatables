////////////////////////////////////////////////////////////////////////////////
// SheetOptimizer.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  A class for optimizing the rest vertex positions of the inflatable sheet
//  to better fit a target surface.
//
//  We assume that the sheet design is flat and lies on the z = 0 plane.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  07/13/2019 19:27:21
////////////////////////////////////////////////////////////////////////////////
#ifndef SHEETOPTIMIZER_HH
#define SHEETOPTIMIZER_HH

#include "InflatableSheet.hh"
#include "TargetSurfaceFitter.hh"
#include "CollapseBarrier.hh"

#include <memory>

// variables: equilibrium vars (deformed positions), followed by
//            design parameters (rest positions)
struct SheetOptimizer {
    enum class EnergyType { Full, Simulation, Fitting, Smoothing, CollapseBarrier };
    using Mesh = InflatableSheet::Mesh;
    using Real = InflatableSheet::Real;

    using  V2d = InflatableSheet:: V2d;
    using  VXd = InflatableSheet:: VXd;
    using  M2d = InflatableSheet:: M2d;
    using  M3d = InflatableSheet:: M3d;
    using M23d = InflatableSheet::M23d;
    using M32d = InflatableSheet::M32d;
    using MX2d = Eigen::Matrix<Real, Eigen::Dynamic, 2>;
    using VSFJ = VectorizedShapeFunctionJacobian<3, V2d>;

    SheetOptimizer(std::shared_ptr<InflatableSheet> s, const Mesh &targetSurface);

          InflatableSheet &sheet()       { return *m_sheet; }
    const InflatableSheet &sheet() const { return *m_sheet; }

          Mesh &mesh()       { return sheet().mesh(); }
    const Mesh &mesh() const { return sheet().mesh(); }

    const TargetSurfaceFitter &targetSurfaceFitter() const { return m_targetSurfaceFitter; }
          TargetSurfaceFitter &targetSurfaceFitter()       { return m_targetSurfaceFitter; }

    size_t numVars() const { return numEquilibriumVars() + numDesignVars(); }
    const VXd &getVars() const { return m_currVars; }

    size_t numEquilibriumVars() const { return sheet().numVars(); }
    size_t numDesignVars()      const { return 2 * mesh().numVertices(); }

    size_t equilibriumVarOffset() const { return 0; }
    size_t      designVarOffset() const { return equilibriumVarOffset() + numEquilibriumVars(); }

    // Construct a view accessing the design variables part of the full
    // variables vector "allVars" as a |V|x2 matrix of rest vertex positions.
    template<typename Vector>
    auto restPositionsFromVariables(Vector &allVars) const {
        using MapType = Eigen::Map<std::conditional_t<std::is_const<std::remove_pointer_t<decltype(allVars.data())>>::value, const MX2d, MX2d>>;
        return MapType(allVars.tail(numDesignVars()).data(), mesh().numVertices(), 2);
    }

    M23d getTriRestPositions(size_t ti) const {
        M23d out;
        const auto &tri = mesh().element(ti);
        for (const auto &v : tri.vertices())
            out.col(v.localIndex()) = restPositionsFromVariables(m_currVars).col(v.index());
        return out;
    }

    void setVars(const VXd &vars) {
        m_currVars = vars;
        auto &s = sheet();
        s.setRestVertexPositions(restPositionsFromVariables(vars),
                                 vars.head(s.numVars()));

        m_targetSurfaceFitter.updateClosestPoints(s.deformedWallVertexPositions(), m_wallVtxOnBoundary);
        m_collapseBarrier.setPositions(restPositionsFromVariables(vars));
    }

    Real  energy(EnergyType etype = EnergyType::Full) const;
    VXd gradient(EnergyType etype = EnergyType::Full) const;

    const Real &weight(EnergyType etype) const {
        if (etype == EnergyType::Simulation) return m_weights[0];
        if (etype == EnergyType::Fitting   ) return m_weights[1];
        if (etype == EnergyType::Smoothing ) return m_weights[2];
        throw std::runtime_error("Unexpected EnergyType");
    }
    Real &weight(EnergyType etype) { return const_cast<Real &>(const_cast<const SheetOptimizer &>(*this).weight(etype)); }

    size_t hessianNNZ() const { return hessianSparsityPattern().nz; } // TODO: predict without constructing
    SuiteSparseMatrix hessianSparsityPattern(Real val = 0.0) const;
    void              hessian(SuiteSparseMatrix &H, EnergyType etype = EnergyType::Full) const; // accumulate Hessian to H
    SuiteSparseMatrix hessian(                      EnergyType etype = EnergyType::Full) const; // construct and return Hessian

private:
    std::shared_ptr<InflatableSheet>      m_sheet;
    TargetSurfaceFitter                   m_targetSurfaceFitter;
    CollapseBarrier<>                     m_collapseBarrier;
    std::array<Real, 3>                   m_weights{{1.0, 1.0, 1.0}};
    VXd                                   m_currVars;
    SuiteSparseMatrix                     m_restLaplacian;
    std::vector<bool>                     m_wallVtxOnBoundary;
};

#endif /* end of include guard: SHEETOPTIMIZER_HH */

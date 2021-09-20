////////////////////////////////////////////////////////////////////////////////
// InflatableSheet.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  An "inflatable sheet" is a structure formed by fusing together two
//  identical sheets of inextensible material along their boundaries and along
//  some internal curves to form air channels.
//
//  The "top" and "bottom" sheet are two oppositely oriented copies of a single
//  planar triangle mesh. The sheets are fused together by making each copy
//  share variables controlling the "fused vertices." This is done by
//  introducing a "reduced vertex set" whose positions determine the positions
//  of all vertices on the top and bottom sheets.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  04/05/2019 17:46:33
////////////////////////////////////////////////////////////////////////////////
#ifndef INFLATABLESHEET_HH
#define INFLATABLESHEET_HH

#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/Utilities/ArrayPadder.hh>
#include <MeshFEM/Utilities/MeshConversion.hh>
#include <memory>
#include <string>

#include <MeshFEM/EnergyDensities/StVenantKirchhoff.hh>
#include <MeshFEM/EnergyDensities/TensionFieldTheory.hh>
#include <MeshFEM/EnergyDensities/TangentElasticityTensor.hh>
#include <MeshFEM/EnergyDensities/IsoCRLETensionFieldMembrane.hh>

#include "TensionFieldEnergy.hh"
#include "IncompressibleBalloonEnergyWithHessProjection.hh"

struct InflatableSheet {
#if INFLATABLES_LONG_DOUBLE
    using Real = long double;
#else
    using Real = double;
#endif

#if 0
    using INeo_TFT_CBased = RelaxedEnergyDensity<IncompressibleNeoHookeanEnergyCBased<Real>>;
#else
    using INeo_TFT_CBased = OptionalTensionFieldEnergy<InflatableSheet::Real>;
#endif
    using StVk_TFT_CBased = RelaxedEnergyDensity<StVenantKirchhoffEnergyCBased<Real, 2>>;
#if 1
    using EnergyDensityCBased = INeo_TFT_CBased;
#else
    using EnergyDensityCBased = StVk_TFT_CBased;
#endif
#if 1
    using EnergyDensity = EnergyDensityFBasedFromCBased<EnergyDensityCBased, 3>;
#else
    using EnergyDensity = IsoCRLETensionFieldMembrane<Real>;
#endif

    using  V2d = Eigen::Matrix<Real, 2, 1>;
    using  V3d = Eigen::Matrix<Real, 3, 1>;
    using  V4d = Eigen::Matrix<Real, 4, 1>;
    using  VXd = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using  M2d = Eigen::Matrix<Real, 2, 2>;
    using  M3d = Eigen::Matrix<Real, 3, 3>;
    using M23d = Eigen::Matrix<Real, 2, 3>;
    using M32d = Eigen::Matrix<Real, 3, 2>;
    using MX2d = Eigen::Matrix<Real, Eigen::Dynamic, 2>;
    using MX3d = Eigen::Matrix<Real, Eigen::Dynamic, 3>;
    using M3Xd = Eigen::Matrix<Real, 3, Eigen::Dynamic>;
    using M34d = Eigen::Matrix<Real, 3, 4>;
    using  M4d = Eigen::Matrix<Real, 4, 4>;
    using VSFJ = VectorizedShapeFunctionJacobian<3, V2d>;
    using ETensor = ElasticityTensor<Real, 2>;

    using Mesh = FEMMesh<2, 1, V3d>; // Piecewise linear triangle mesh embedded in R^3
    enum class EnergyType { Full, Elastic, Pressure };

    // Build from a triangle mesh representing the "top" sheet.
    // The "bottom" sheet is an oppositely oriented copy.
    InflatableSheet(const std::shared_ptr<Mesh> &inMesh, const std::vector<bool> &fusedVtx = std::vector<bool>());

    void setMaterial(const EnergyDensity &psi) {
        for (auto &ted : m_triEnergyDensity)
            ted.copyMaterialProperties(psi);
    }

    // Update the sheet design (repositioning rest vertices).
    // Also optionally update the equilibrium variables (i.e. the deformed configuration).
    template<class Derived>
    void setRestVertexPositions(const Eigen::MatrixBase<Derived> &X) {
        mesh().setNodePositions(pad_columns<1>(X));
        m_updateB();
        setVars(getVars()); // Also update the gradient quantities
    }
    template<class Derived>
    void setRestVertexPositions(const Eigen::MatrixBase<Derived> &X, Eigen::Ref<const VXd> vars) {
        mesh().setNodePositions(pad_columns<1>(X));
        m_updateB();
        setVars(vars); // Also update the gradient quantities
    }

    size_t numVars() const { return 3 * m_numReducedVertices; }
    const VXd &getVars() const { return m_currVars; }
    void setVars(Eigen::Ref<const VXd> vars);
    size_t numSheetTris() const { return 2 * mesh().numTris(); }
    size_t numTubeTris() const { return numSheetTris() - 2 * m_numTopSheetWallTris; }

    // Use a rigid transformation of the passed (top) sheet vertex positions as
    // an initial deformed configuration for both the top and bottom sheets
    // (resulting in an uninflated structure).
    // This rigid transformation is chosen to enable pinning rigid motion with
    // 6 variable pin constraints (subsequently accessed by rigidMotionPinVars()).
    // If "prepareRigidMotionPinConstraints" is false, then the unmodified P is used,
    // but rigid motion pin constraints are not set up.
    void setUninflatedDeformation(M3Xd P /* copy modified inside */, bool prepareRigidMotionPinConstraints = true);
    void setIdentityDeformation() {
        M3Xd P(3, mesh().numVertices());
        for (const auto v : mesh().vertices())
            P.col(v.index()) = v.node()->p.cast<Real>();
        setUninflatedDeformation(P);
    }

    void setUseTensionFieldEnergy(bool useTFE) {
        for (auto &ted : m_triEnergyDensity)
            ted.setRelaxationEnabled(useTFE);
    }

    // Note: enabling the Hessian projected energy necessarily disables the tension field energy.
    void setUseHessianProjectedEnergy(bool useHPE) {
        for (auto &ted : m_projectedTriEnergyDensity)
            ted.applyHessianProjection = useHPE;
        m_useHessianProjectedEnergy.assign(numSheetTris(), useHPE);
        if (useHPE) setVars(getVars()); // The hessian-projected energy density has not necessarily been updated for the current variables...
    }

    // Disable the tension field theory approximation in the fused regions to model compressive
    // forces--with or without analytic Hessian projection.
    void disableFusedRegionTensionFieldTheory(bool useHessianProjection) {
        for (const auto t : mesh().tris()) {
            if (m_airChannelForTri[t.index()] == 0) { // Is this a fused triangle?
                for (size_t sheetIdx = 0; sheetIdx < 2; ++sheetIdx) {
                    size_t sti = sheetTriIdx(sheetIdx, t.index());
                    m_triEnergyDensity         [sti].setRelaxationEnabled(false);
                    m_useHessianProjectedEnergy[sti] = useHessianProjection;
                    m_projectedTriEnergyDensity[sti].applyHessianProjection = useHessianProjection;
                }
                m_useHessianProjectedEnergy[sheetTriIdx(1, t.index())] = useHessianProjection;
            }
        }
        if (useHessianProjection) setVars(getVars()); // The hessian-projected energy density has not necessarily been updated for the current variables...
    }

    // Note: the behavior here is undefined if the sheets' energy density types are inhomogeneous
    bool usingHessianProjectedEnergy(size_t i) const { return m_useHessianProjectedEnergy.at(i); }
    bool usingTensionFieldEnergy(size_t i)     const { return !m_useHessianProjectedEnergy.at(i)
                                                            && m_triEnergyDensity.at(i).getRelaxationEnabled(); }

    std::vector<bool> usingHessianProjectedEnergy() const { return m_useHessianProjectedEnergy; }

    void setRelaxedStiffnessEpsilon(Real val) {
        for (auto &ted : m_triEnergyDensity)
            ted.setRelaxedStiffnessEpsilon(val);
    }

    std::array<size_t, 3> tensionStateHistogram() const {
        std::array<size_t, 3> counts = {{0, 0, 0}};
        for (const auto &ted : m_triEnergyDensity)
            ++counts[ted.tensionState()];
        return counts;
    }

    const std::array<size_t, 6> &rigidMotionPinVars() const { return m_rigidMotionPinVars; }
    void setRigidMotionPinVars(const std::array<size_t, 6> &pinVars) { m_rigidMotionPinVars = pinVars; }

    void setPressure(Real p) { m_pressure = p; }
    Real getPressure() const { return m_pressure; }

    void setThickness   (Real h) { m_thickness    = h; m_updateMaterialProperties(); }
    void setYoungModulus(Real E) { m_youngModulus = E; m_updateMaterialProperties(); }
    Real getThickness()    const { return m_thickness; }
    Real getYoungModulus() const { return m_youngModulus; }

    // Volume enclosed by the sheet's tubes
    Real volume() const;
    Real referenceVolume() const { return m_referenceVolume; }
    void setReferenceVolume(Real V0) { m_referenceVolume = V0; }

    Real energy(EnergyType etype = EnergyType::Full) const;
    Real energyPressurePotential() const;

    VXd gradientPressurePotential() const;
    VXd gradient(EnergyType etype = EnergyType::Full) const;

    size_t hessianNNZ() const { return hessianSparsityPattern().nz; } // TODO: predict without constructing
    SuiteSparseMatrix hessianSparsityPattern(Real val = 0.0) const;
    void              hessian(SuiteSparseMatrix &H, EnergyType etype = EnergyType::Full) const; // accumulate Hessian to H
    SuiteSparseMatrix hessian(                      EnergyType etype = EnergyType::Full) const; // construct and return Hessian

          Mesh &mesh()       { return *m_topSheetMesh; }
    const Mesh &mesh() const { return *m_topSheetMesh; }

    // Access the mesh shared pointer from this instance
    std::shared_ptr<Mesh>       meshPtr()       { return m_topSheetMesh; }
    std::shared_ptr<const Mesh> meshPtr() const { return m_topSheetMesh; }

    auto getDeformedVtxPosition(size_t vi, size_t sheetIdx) const {
        return m_currVars.segment<3>(3 * m_reducedVtxForVertex(vi, sheetIdx));
    }

    void getDeformedTriCornerPositions(size_t ti, size_t sheetIdx, M3d &out) const {
        if (sheetIdx > 1) throw std::runtime_error("sheetIdx out of bounds");
        const auto &tri = mesh().element(ti);
        for (const auto v : tri.vertices())
            out.col(v.localIndex()) = getDeformedVtxPosition(v.index(), sheetIdx);
    }

    void getDeformedTriCornerDisplacement(size_t ti, size_t sheetIdx, M3d &out) const {
        getDeformedTriCornerPositions(ti, sheetIdx, out);
        const auto &tri = mesh().element(ti);
        for (const auto v : tri.vertices())
            out.col(v.localIndex()) -= v.node()->p.cast<Real>();
    }

    // System variable corresponding to component "compIdx" of vertex "vtxIdx"
    // on top/bottom sheet "sheetIdx"
    size_t varIdx(size_t sheetIdx, size_t vtxIdx, size_t compIdx = 0) const {
        return 3 * m_reducedVtxForVertex(vtxIdx, sheetIdx) + compIdx;
    }

    size_t sheetTriIdx(size_t sheetIdx, size_t triIdx) const {
        return mesh().numTris() * sheetIdx + triIdx;
    }

    bool isWallTri (size_t idx) const { return m_airChannelForTri.at(idx) == 0; }
    bool isWallVtx (size_t idx) const { return m_fusedVtx.at(idx); }
    bool isFusedVtx(size_t idx) const { return m_fusedVtx.at(idx); }
    const std::vector<bool> fusedVtx() const { return m_fusedVtx; }

    // WARNING: when consider1D is true, this considers a half edge bridging between two
    // wall regions to be a wall boundary half edge (because of the 1d wall
    // curve criterion).
    bool isWallBoundaryHE(size_t heidx, bool consider1D = true) const {
        auto he = mesh().halfEdge(heidx);
        if (he.isBoundary()) return true;
        size_t airChannelLeft = m_airChannelForTri.at(he.tri().index()),
               airChannelRight = m_airChannelForTri.at(he.opposite().tri().index());
        if (airChannelLeft != airChannelRight) return true; // Border between different air channels.
        if (consider1D && (airChannelLeft > 0)) {
            // Also include edges lying on 1d wall curves (fused edges that separate two air channel triangles)
            if (isWallVtx(he.tip().index()) && isWallVtx(he.tail().index()))
                return true;
        }
        return false;
    }

    // Get the index of the vertex m_topSheetMesh associated with a
    // given variable. Note, the variable could control the vertex in the "top" or "bottom"
    // copies of the mesh
    struct ISheetVtx {
        size_t vi;  // The vertex in question.
        int sheet; // The mesh copy in question. 0 = none, 1 = top, 2 = bot, 3 = both (fused)
    };
    ISheetVtx vtxForVar(int var) const {
        if (size_t(var) >= numVars()) throw std::runtime_error("var out of bounds");
        int reducedVtxIdx = var / 3;
        ISheetVtx result;
        for (const auto v : m_topSheetMesh->vertices()) {
            result.vi = v.index();
            result.sheet = 0;
            if (m_reducedVtxForVertex(v.index(), 0) == reducedVtxIdx) result.sheet += 1;
            if (m_reducedVtxForVertex(v.index(), 1) == reducedVtxIdx) result.sheet += 2;
            if (result.sheet > 0) return result;
        }
        ++result.vi; // one past the last vertex to indicate "not found"
        return result;
    }

    size_t numWallVertices() const { return m_wallVertices.size(); }
    const std::vector<size_t> &wallVertices() const { return m_wallVertices; }
    const std::vector<std::pair<size_t, size_t>> wallBoundaryEdges() const {
        std::vector<std::pair<size_t, size_t>> result;
        for (const auto he : mesh().halfEdges()) {
            if (!he.isPrimary()) continue;
            if (isWallBoundaryHE(he.index())) result.push_back({he.tip().index(), he.tail().index()});
        }
        return result;
    }
    // Get a list of polylines (each represented as a sequence of top sheet mesh vertex indices)
    // representing the boolean intersection of the fused regions with the sheet boundary.
    // These will consist of the boundary halfedges whose opposite triangle is a wall triangle.
    std::list<std::vector<size_t>> fusedRegionBooleanIntersectSheetBoundary() const;

    const std::vector<size_t> &airChannelIndices() const { return m_airChannelForTri; }

    // A vertex is a true wall vertex if it lies in a fused triangle or is the
    // endpoint of a non-mesh-boundary wall halfedge.
    // (i.e., we omit vertices that are only fused because they lie on the mesh
    // boundary.)
    const std::vector<size_t> trueWallVertices() const {
        std::vector<size_t> result;
        result.reserve(mesh().numVertices());
        for (const auto v : mesh().vertices()) {
            bool isWall = false;
            for (const auto he : v.incidentHalfEdges()) {
                if (he.tri() && isWallTri(he.tri().index())) {
                    isWall = true; break;
                }
                if (he.isBoundary()) continue;
                if (isWallBoundaryHE(he.index())) {
                    isWall = true; break;
                }
            }
            if (isWall) result.push_back(v.index());
        }
        return result;
    }

    MX3d deformedWallVertexPositions() const {
        MX3d result(m_wallVertices.size(), 3);
        for (size_t i = 0; i < m_wallVertices.size(); ++i)
            result.row(i) = getDeformedVtxPosition(m_wallVertices[i], 0).transpose();
        return result;
    }

    // Get the wall vertex positions defined by a (potentially altered) top sheet mesh m.
    MX3d wallVertexPositionsFromMesh(const Mesh &m) const {
        MX3d result(m_wallVertices.size(), 3);
        for (size_t i = 0; i < m_wallVertices.size(); ++i) {
            V3d p = m.vertex(m_wallVertices[i]).node()->p.cast<Real>();
            result.row(i) = p.transpose();
        }
        return result;
    }

    MX3d restWallVertexPositions() const { return wallVertexPositionsFromMesh(mesh()); }

    MX3d restVertexPositions() const {
        const auto &m = mesh();
        MX3d result(m.numVertices(), 3);
        for (const auto v : m.vertices())
            result.row(v.index()) = v.node()->p.transpose().cast<Real>();
        return result;
    }

    M2d greenLagrangianStrain(size_t sheetIdx, size_t triIdx) const {
        const auto &JB = m_JB[triIdx + sheetIdx * mesh().numTris()];
        return 0.5 * (JB.transpose() * JB - M2d::Identity());
    }

#if 0
    std::vector<ETensor> tangentElasticityTensors() const {
        std::vector<ETensor> result;
        for (const auto &ted : m_triEnergyDensity) {
            EnergyDensityCBased psi_C;
            psi_C.copyMaterialProperties(ted);
            const M32d &F = ted.getDeformationGradient();
            result.push_back(tangentElasticityTensor(psi_C, F.transpose() * F));
        }
        return result;
    }
#endif

    const aligned_std_vector<EnergyDensity> &triEnergyDensities() const { return m_triEnergyDensity; }
    const M3d &deformationGradient3D(size_t sheet_tri_idx) const { return m_J.at(sheet_tri_idx); }
    // Gradients of the shape functions expressed in the triangle's 2D tangent plane basis
    // (one gradient per column).
    const std::vector<M23d> &shapeFunctionGradients() const { return m_BtGradLambda; }

    const std::vector<M2d> cauchyGreenDeformationTensors() const {
        std::vector<M2d> result(m_JB.size());
        for (size_t sti = 0; sti < m_JB.size(); ++sti) {
            result[sti] = m_JB[sti].transpose() * m_JB[sti];
        }
        return result;
    }
    const VXd &deformedAreas() const { return m_deformed_areas; }
    VXd      undeformedAreas() const {
        const size_t nt = mesh().numTris();
        VXd result(2 * nt);
        for (const auto tri : mesh().tris())
            result[tri.index() + nt] = result[tri.index()] = tri->volume();
        return result;
    }

    std::shared_ptr<Mesh> visualizationMesh(bool duplicateFusedTris = false) const;

    Eigen::MatrixXd visualizationField(Eigen::MatrixXd field, bool duplicateFusedTris = false);

    void writeDebugMesh(const std::string &path) const;

    // Get the boundary (first) and interior (second) fusing curves as closed
    // polylines, represented as a sequence of vertex indices.
    using IdxPolyline = std::vector<size_t>;
    std::pair<std::vector<IdxPolyline>, std::vector<IdxPolyline>> getFusingPolylines() const;

    // Helper routines for serialization/restore
    using MaterialConfiguration = std::tuple<Real, // m_triEnergyDensity          stiffness
                                             bool, // m_triEnergyDensity          useTensionField
                                             Real, // m_projectedTriEnergyDensity stiffness
                                             bool>;// m_projectedTriEnergyDensity applyHessianProjection
    std::vector<MaterialConfiguration> getMaterialConfiguration() const {
        std::vector<MaterialConfiguration> result;
        const size_t nst = numSheetTris();
        assert(m_triEnergyDensity.size() == nst);
        assert(m_projectedTriEnergyDensity.size() == nst);
        for (size_t i = 0; i < nst; ++i) {
            result.emplace_back(m_triEnergyDensity[i].stiffness(), m_triEnergyDensity[i].getRelaxationEnabled(),
                                m_projectedTriEnergyDensity[i].stiffness, m_projectedTriEnergyDensity[i].applyHessianProjection);
        }
        return result;
    }
    void applyMaterialConfiguration(const std::vector<MaterialConfiguration> &c) {
        const size_t nst = numSheetTris();
        if ((m_triEnergyDensity.size() != nst) || (m_projectedTriEnergyDensity.size() != nst))
            throw std::runtime_error("Material configuration size mismatch");
        for (size_t i = 0; i < nst; ++i) {
            m_triEnergyDensity[i].setStiffness(                     std::get<0>(c[i]));
            m_triEnergyDensity[i].setRelaxationEnabled(             std::get<1>(c[i]));
            m_projectedTriEnergyDensity[i].stiffness              = std::get<2>(c[i]);
            m_projectedTriEnergyDensity[i].applyHessianProjection = std::get<3>(c[i]);
        }
    }

    const std::vector<M32d> &getJB() const { return m_JB; }

private:
    size_t m_numReducedVertices;
    Eigen::Matrix<int, Eigen::Dynamic, 2> m_reducedVtxForVertex;

    std::vector<bool  > m_fusedVtx;
    std::vector<size_t> m_airChannelForTri;
    std::vector<size_t> m_wallVertices;
    size_t m_numTopSheetWallTris;

    std::shared_ptr<Mesh> m_topSheetMesh;

    VXd m_currVars;
    Real m_pressure = 0.0; // current inflation pressure
    Real m_referenceVolume = 0.0; // V_0 used for defining the pressure potential.
    Real m_thickness = 0.075, m_youngModulus = 300; // Material properties used to set the energy density stiffness.

    // Set all energy densities' stiffness parameters based on the thickness,
    // Young's modulus parameters configured for this sheet.
    void m_updateMaterialProperties() {
        const size_t nst = numSheetTris();
        Real stiffness = m_youngModulus * m_thickness / 6.0;
        for (size_t i = 0; i < nst; ++i) {
            m_triEnergyDensity[i].setStiffness(stiffness);
            m_projectedTriEnergyDensity[i].stiffness = stiffness;
        }
    }

    // Orthonormal basis for each triangle's tangent space (both top and bottom sheet)
    std::vector<M32d> m_B;

    // Method to update the tangent space basis for each triangle (call when rest positions change)
    void m_updateB();

    ////////////////////////////////////////////////////////////////////////////
    // Quantities computed from the current deformation
    ////////////////////////////////////////////////////////////////////////////
    // Jacobian for each triangle (mapping from the triangle's 2D tangent space to 3D)
    // in the top sheet (first) and bottom sheet (after)
    std::vector<M32d> m_JB;
    std::vector<M3d > m_J;  // mapping from 3D to 3D (with J n = 0)
    std::vector<M23d> m_BtGradLambda;
    M3Xd m_deformed_normals;
    VXd  m_deformed_areas;

    std::array<size_t, 6> m_rigidMotionPinVars;

    aligned_std_vector<EnergyDensity> m_triEnergyDensity;
    aligned_std_vector<IncompressibleBalloonEnergyWithHessProjection<Real>> m_projectedTriEnergyDensity;
    std::vector<bool> m_useHessianProjectedEnergy;
};

#endif /* end of include guard: INFLATABLESHEET_HH */

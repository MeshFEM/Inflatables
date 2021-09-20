////////////////////////////////////////////////////////////////////////////////
// ReducedSheetOptimizer.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Optimization of an inflatable sheet where the equilibrium constraints are
//  imposed at each step (optionally with a force tugging the inflated sheet
//  toward the target surface).
//
//  We keep track of two equilibrium states: the "committed" state and the
//  "linesearch" state. The committed state is used as a starting point
//  for the equilibrium solve at new parameters (in case the linesearch state
//  is bad). When a good design iterate is found, its corresponding
//  equilibrium state is committed.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  07/22/2019 16:48:09
////////////////////////////////////////////////////////////////////////////////
#ifndef REDUCEDSHEETOPTIMIZER_HH
#define REDUCEDSHEETOPTIMIZER_HH

#include "TargetAttractedInflation.hh"
#include "CollapsePreventionEnergy.hh"
#include "WallPositionInterpolator.hh"
#include "FusingCurveSmoothness.hh"
#include "CompressionPenalty.hh"

#include "inflation_newton.hh"
#include <limits>

#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/ParallelAssembly.hh>
#include <memory>

// variables: equilibrium vars (deformed positions), followed by
//            design parameters (rest positions)
struct ReducedSheetOptimizer {
    enum class EnergyType { Full, Fitting, CollapseBarrier, Smoothing, CompressionPenalty };
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

    using CB  = CollapseBarrier<CollapsePreventionEnergySV>;
    using WPI = WallPositionInterpolator<Real>;

    // Construct a view accessing the wall rest positions from the vector of
    // variables.
    template<typename Vector>
    auto wallRestPositionsFromVariables(Vector &vars) const {
        using MapType = Eigen::Map<std::conditional_t<std::is_const<std::remove_pointer_t<decltype(vars.data())>>::value, const MX2d, MX2d>>;
        return MapType(vars.data(), numWallVertices(), 2);
    }

    // `originalDesignMesh` is the sheet mesh with respect to which the smoothness and collapse barrier terms are measured.
    // If unspecified, this defaults to tai->mesh() (the current sheet design passed to the constructor), but if multiple
    // stages of optimizations are to be run on the same design with different ReducedSheetOptimizer instances, `originalDesignMesh`
    // should generally remain the design before any optimization was performed (so that the non-smoothness and wall triangle shrinkage
    // permitted by each stage cannot accumulate).
    // Note: the WallPositionInterpolator is rebuilt for the current design mesh. However, this should produce the same
    // sheet mesh (up to roundoff error) since the FEM Laplacian of the coordinate function on a planar mesh is 0.
    ReducedSheetOptimizer(std::shared_ptr<TargetAttractedInflation> tai, const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &eopts = NewtonOptimizerOptions(), double detActivationThreshold = 0.9,
                          VXd initialVars = VXd(), const Mesh *originalDesignMesh = nullptr /* ownership not transferred; copy is made */)
        : m_targetAttractedInflation(tai),
          m_smoothness(std::make_shared<FusingCurveSmoothness>(tai->sheet())), // Note: only uses the combinatorics of tai->sheet(), so we needn't use the originalDesignMesh
          m_wallPosInterp(std::make_shared<WPI>(tai->sheet()))
    {
        // Collapse prevention kicks in when element compressed to detActivationThreshold * its original area
        if (originalDesignMesh) m_collapseBarrier = std::make_shared<CB>(*originalDesignMesh, detActivationThreshold);
        else                    m_collapseBarrier = std::make_shared<CB>(tai->mesh(),         detActivationThreshold);

        m_compressionPenalty = std::make_shared<CompressionPenalty>(tai->sheetPtr());

        const auto &s = sheet();

        if (initialVars.rows() == 0) {
            initialVars = varsForDesignMesh(mesh());
        }
        if (size_t(initialVars.rows()) != numVars()) {
            throw std::runtime_error("Invalid variable vector size");
        }

        m_linesearchVars = VXd::Zero(numVars());
        m_committedVars  = initialVars;

        m_committedEquilibrium  = tai->getVars();
        m_linesearchEquilibrium = VXd::Zero(numEquilibriumVars());

        m_committedRestPositions  = MX2d::Zero(numVertices(), 2);
        m_linesearchRestPositions = MX2d::Zero(numVertices(), 2);

        // Set up equilibrium solver
        m_equilibriumSolver = get_inflation_optimizer(*m_targetAttractedInflation, fixedVars, eopts);

        // Initialize remaining state
        bool success = setVars(getCommittedVars());
        if (!success) throw std::runtime_error("Failed to solve equilbrium problem");

        // Stash the committed equilibrium's Hessian factorization for use
        // with first-order equilibrium prediction.
        commitDesign();
    }

    const TargetAttractedInflation &targetAttractedInflation() const { return *m_targetAttractedInflation; }
          TargetAttractedInflation &targetAttractedInflation()       { return *m_targetAttractedInflation; }
    std::shared_ptr<TargetAttractedInflation> targetAttractedInflationPtr() const { return m_targetAttractedInflation; }

    const InflatableSheet &sheet() const { return targetAttractedInflation().sheet(); }
          InflatableSheet &sheet()       { return targetAttractedInflation().sheet(); }

    const Mesh &mesh() const { return targetAttractedInflation().mesh(); }
          Mesh &mesh()       { return targetAttractedInflation().mesh(); }
    const Mesh &originalMesh() const { return m_collapseBarrier->mesh(); }
    std::shared_ptr<Mesh> meshPtr() { return targetAttractedInflation().sheet().meshPtr(); }

    const TargetSurfaceFitter &targetSurfaceFitter() const { return targetAttractedInflation().targetSurfaceFitter(); }
          TargetSurfaceFitter &targetSurfaceFitter()       { return targetAttractedInflation().targetSurfaceFitter(); }

    const Nondimensionalization &nondimensionalization() const { return targetAttractedInflation().nondimensionalization; }

    size_t numWallVertices() const { return m_wallPosInterp->numWallVertices(); }
    size_t     numVertices() const { return mesh().numVertices(); }
    size_t         numVars() const { return 2 * numWallVertices(); }

    // Note: the variables we expose publicly are re-scaled to make the gradient invariant to
    // design scalings. Therefore `getVars` and `setVars` both apply the appropriate change of variables
    // from the un-normalized private variables.
    // All internal implementations use the un-normalized private variables directly.
    const VXd  getVars()                   const { return (1.0 / nondimensionalization().restVarScale()) * m_linesearchVars; }
    const VXd  getCommittedVars()          const { return (1.0 / nondimensionalization().restVarScale()) * m_committedVars;          }
    const MX2d getCommittedRestPositions() const { return (1.0 / nondimensionalization().restVarScale()) * m_committedRestPositions; }
    const VXd  getCommittedEquilibrium()   const { return                                                  m_committedEquilibrium;   }

    // Get the values of an inflatable sheet's design variables that correspond
    // to a particular altered top sheet mesh.
    // Note: we return the variables that will reconstruct the fused vertices
    //       exactly, while the tube mesh vertices will not be correctly
    //       reconstructed unless the same wall position interpolater was used
    //       to position them in `m`.
    VXd varsForDesignMesh(const Mesh &m) const {
        if (m.numVertices() != mesh().numVertices()) throw std::runtime_error("Mesh size mismatch");

        const auto &wv = sheet().wallVertices();
        MX2d restWallVtxPos(wv.size(), 2);
        for (size_t i = 0; i < wv.size(); ++i) {
            restWallVtxPos.row(i) = truncateFrom3D<Point2D>(m.vertex(wv[i]).node()->p.cast<Real>().eval());
        }

        VXd result(numVars());
        wallRestPositionsFromVariables(result) = restWallVtxPos;
        return result;
    }

    // Number of variables in the inner equilibrium solve; note that these
    // variables are NOT counted as variables of this sheet optimizer.
    size_t numEquilibriumVars() const { return targetAttractedInflation().numVars(); }
    const std::vector<size_t> &fixedEquilibriumVars() const { return m_equilibriumSolver->get_problem().fixedVars(); }

    // bailEarlyOnCollapse: determine if `vars` corresponds to
    // a collapsed/inverted state and if so, return immediately
    // (without running the equilibrium solve). This iterate
    // will have infinite energy and usually is not worth evaluating fully.
    // Likewise, we bail early if the collapse barrier and smoothing energies alone exceed `bailEnergyThreshold`.
    // The caller can detect if this bail happened by inspecting the return type (false for a bailed setVars)
    bool setVars(VXd vars, bool bailEarlyOnCollapse = false, Real bailEnergyThreshold = std::numeric_limits<float>::max()) {
        vars *= nondimensionalization().restVarScale();

        // std::cout << "Evaluating at squared dist from linesearch iterate " << (vars - m_linesearchVars).squaredNorm() << std::endl;
        // Note: now that we divide and multiply variables by `restVarscale`,
        // we won't get exact equality with past linesearch/committed vars and
        // need a small tolerance for these checks.
        // We always use m_committedVars to measure the distance in case the
        // previous linesearch design happens to be bad.
        const Real distTol = 1e-15;
        const Real relLinesearchDist = (vars - m_linesearchVars).norm() / m_committedVars.norm();

        if (size_t(vars.rows()) != numVars()) throw std::runtime_error("Incorrect number of variables passed");
        if (relLinesearchDist < distTol) return true; // linesearch iterate has not changed; nothing to update
                                                      // Note: if vars, m_linesearchVars correspond to a bailed iterate,
                                                      //       this will return `true`, incorrectly reporting that the
                                                      //       evaluation completed without bailing. (However,
                                                      //       the evaluated energy will still be infinite)

        // Interpolate and apply the new rest vertex positions.
        m_linesearchVars = vars;
        m_linesearchRestPositions.resize(numVertices(), 2);
        m_wallPosInterp->interpolate(wallRestPositionsFromVariables(m_linesearchVars), m_linesearchRestPositions);
        sheet().setRestVertexPositions(m_linesearchRestPositions);
        // MeshIO::save("debug.msh", sheet().mesh());
        m_collapseBarrier->setPositions(m_linesearchRestPositions);

        Real cbEnergy = energy(EnergyType::CollapseBarrier);
        if (bailEarlyOnCollapse && std::isinf(cbEnergy))
            return false;
        if (cbEnergy + energy(EnergyType::Smoothing) > bailEnergyThreshold)
            return false;

        // Determine the change from the committed rest positions
        VXd delta_vars = vars - m_committedVars;
        bool evalAtCommittedDesign = (delta_vars.norm() / m_committedVars.norm()) < distTol;

        ///////////////////////////////////////////////////////////////////////
        // Equilibrium update with first order prediction
        ///////////////////////////////////////////////////////////////////////
        auto &solver = m_equilibriumSolver->solver;

        // Evaluate 0th order prediction of equilibrium
        m_targetAttractedInflation->setVars(m_committedEquilibrium);
        Real energy0thOrder = m_targetAttractedInflation->energy();

        BENCHMARK_START_TIMER_SECTION("First order prediction");
        if (useFirstOrderPrediction && !evalAtCommittedDesign) {
            if (solver.hasStashedFactorization()) {
                // Use the factorized Hessian for the committed design
                solver.swapStashedFactorization();
                // Roll back to the committed rest positions.
                sheet().setRestVertexPositions(m_committedRestPositions);

                {
                    MX2d delta_X(numVertices(), 2);
                    m_wallPosInterp->interpolate(wallRestPositionsFromVariables(delta_vars), delta_X);
                    // Run first order prediction (using Taylor expansion around committed design)
                    Eigen::VectorXd rhs = m_equilibriumSolver->removeFixedEntries(
                            -apply_d2E_dxdX(delta_X).cast<double>()
                        );
                    VXd delta_x = m_equilibriumSolver->extractFullSolution(solver.solve(rhs)).cast<Real>();
                    m_targetAttractedInflation->setVars(m_committedEquilibrium + delta_x);
                }

                // Re-apply the new rest positions to compute energy
                sheet().setRestVertexPositions(m_linesearchRestPositions);
                // Place the committed design's factorized Hessian back in the stash.
                solver.swapStashedFactorization();

                // Verify that prediction lowered energy at the new design; otherwise, roll back.
                Real energy1stOrder = m_targetAttractedInflation->energy();
                if (energy1stOrder < energy0thOrder) { std::cout << " used first order prediction, energy reduction " << energy0thOrder - energy1stOrder << std::endl; }
                else                                 { std::cout << "First order prediction failed" << std::endl;
                                                       m_targetAttractedInflation->setVars(m_committedEquilibrium); }
            }
            else {
                std::cout << "Error: no stashed factorization found." << std::endl;
            }
        }
        BENCHMARK_STOP_TIMER_SECTION("First order prediction");

        if (evalAtCommittedDesign && solver.hasStashedFactorization()) {
            // If we're re-evaluating at the committed design (and the Hessian factorization
            // for this design has already been stashed--i.e., we weren't called from the
            // constructor), simply copy it into the current factorization, preserving
            // the stash.
            solver.swapStashedFactorization();
            solver.stashFactorization();
            m_linesearchEquilibrium = m_targetAttractedInflation->getVars();
            m_updateAdjointState();
            return true;
        }

        // We're evaluating at a new design; equilibrium needs update.
        return forceEquilibriumUpdate();
    }

    // Force an re-solve of the linesearch iterate's equilbrium
    // (automatically called when the design has changed, but the user can call this
    // after modifying the equilibrium stored in m_targetAttractedInflation to
    // try to force the sheet to reinflate into a different local minimum).
    bool forceEquilibriumUpdate() {
        auto result = m_equilibriumSolver->optimize();
        // TODO: decide whether this is helpful--often the equilibrium solver
        // fails to converge due to backtracking failures. We keep an eye
        // on the equilibrium solver's progress during the optimization and
        // do not normally see it failing at high net force configurations.
        // if (!result.success) return false; // Bail! the equilibrium was invalid!

        // Use the Hessian at the actual equilibrium (instead of the
        // second-to-last newton iteration) for subsequent sensitivity
        // analysis.
        try {
            m_equilibriumSolver->update_factorizations();
        }
        catch (...) {
            std::cout << "Factorization update failed" << std::endl;
            // If the equilibrium solve failed so badly that we cannot
            // update the Hessian factorization (e.g., Tau runaway) bail from
            // this iterate (to prevent optimization from halting)
            return false;
        }
        m_linesearchEquilibrium = m_targetAttractedInflation->getVars();
        m_updateAdjointState();
        return true;
    }

    void forceAdjointStateUpdate() { m_updateAdjointState(); }

    // Commit the current linesearch iterate, starting all future linesearches from it.
    void commitDesign() {
        m_committedEquilibrium   = m_linesearchEquilibrium;
        m_committedRestPositions = m_linesearchRestPositions;
        m_committedVars          = m_linesearchVars;

        // Save the committed factorization in the stash
        m_equilibriumSolver->solver.stashFactorization();
    }

    Real energy(EnergyType etype = EnergyType::Full) const {
        const auto &n = nondimensionalization();
        Real result = 0.0;
        if ((etype == EnergyType::Full) || (etype == EnergyType::Fitting))
            result += n.fittingEnergyScale() * targetSurfaceFitter().energy();
        if ((etype == EnergyType::Full) || (etype == EnergyType::CollapseBarrier))
            result += n.collapseBarrierScale() * m_collapseBarrier->energy();
        if ((etype == EnergyType::Full) || (etype == EnergyType::Smoothing))
            result += m_smoothness->energy(mesh(), originalMesh(), n);
        if ((etype == EnergyType::Full) || (etype == EnergyType::CompressionPenalty)) {
            if (compressionPenaltyWeight != 0.0)
                result += compressionPenaltyWeight * n.fittingEnergyScale() * m_compressionPenalty->J();
        }

        return result;
    }

    // The sensitivity analysis formulas need the term d2E/dxdX
    // and its transpose (where E is the total potential energy).
    // The pressure and fitting terms are independent of the rest configuration, so
    // only the elastic membrane term contributes nonzero values.
    // We must account for the scaling of dE/dx in targetAttractedInflation
    // (the membrane energy scaling and change of deformation variables to "xtilde").
    // However, we do *not* account for the change of variables for rest positions
    // (to "Xtilde") here since this is handled directly in gradient();
    Real membraneMixedHessianNondimScale() const {
        return nondimensionalization().potentialEnergyScale()
             * nondimensionalization().equilibriumVarScale();
    }

    // Visit (contributions to) each entry of the (deformed, rest) vertex
    // position block of the elastic energy Hessian, calling visitor with
    // arguments:
    // (deformed var index, rest vertex index, rest component index, entry)
    template<class Visitor>
    void visit_d2E_dxdX(const Visitor &visitor, size_t element_index) const {
        const auto &s             = sheet();
        const auto &teds          = s.triEnergyDensities();
        const auto &BtGradLambdas = s.shapeFunctionGradients();

        Real nondimScale = membraneMixedHessianNondimScale();

        const auto &tri = mesh().tri(element_index);
        for (size_t sheetIdx = 0; sheetIdx < 2; ++sheetIdx) {
            const size_t sheet_tri_idx = s.sheetTriIdx(sheetIdx, tri.index());
            const auto &ted            = teds         [sheet_tri_idx];
            const auto &J              = s.deformationGradient3D(sheet_tri_idx);
            const auto &BtGradLambda   = BtGradLambdas[sheet_tri_idx];
            const auto &gradLambda     = tri->gradBarycentric();

            M3d de_BtGradLambda = ted.denergy() * BtGradLambda;

            for (size_t comp_b = 0; comp_b < 2; ++comp_b) {
                for (const auto &v_b : tri.vertices()) {
                    M32d J_grad_v_b = J.col(comp_b) * BtGradLambda.col(v_b.localIndex()).transpose();
                    M3d delta_d2E_J_grad_v_b_BtGradLambda = ted.delta_denergy(J_grad_v_b) * BtGradLambda;
                    const Real div_v_b  = gradLambda(comp_b, v_b.localIndex());

                    for (const auto &v_a : tri.vertices()) {
                        for (size_t comp_a = 0; comp_a < 3; ++comp_a) {
                            Real contrib = -delta_d2E_J_grad_v_b_BtGradLambda(comp_a, v_a.localIndex()) +
                                            div_v_b * de_BtGradLambda(comp_a, v_a.localIndex()) - gradLambda(comp_b, v_a.localIndex()) * de_BtGradLambda(comp_a, v_b.localIndex());
                            visitor(s.varIdx(sheetIdx, v_a.index(), comp_a),
                                    v_b.index(), comp_b, nondimScale * contrib * tri->volume());
                        }
                    }
                }
            }
        }
    }

    // Original, unoptimized implementation of visit_d2E_dxdX
    // kept for validation of the optimized implementation and because
    // it is a more readable version of the formulas.
    template<class Visitor>
    void visit_d2E_dxdX_unaccelerated(const Visitor &visitor) const {
        const auto &s             = sheet();
        const auto &teds          = s.triEnergyDensities();
        const auto &BtGradLambdas = s.shapeFunctionGradients();

        Real nondimScale = membraneMixedHessianNondimScale();

        for (const auto &tri : mesh().elements()) {
            for (size_t sheetIdx = 0; sheetIdx < 2; ++sheetIdx) {
                const size_t sheet_tri_idx = s.sheetTriIdx(sheetIdx, tri.index());
                const auto &ted            = teds         [sheet_tri_idx];
                const auto &J              = s.deformationGradient3D(sheet_tri_idx);
                const auto &BtGradLambda   = BtGradLambdas[sheet_tri_idx];
                const auto &gradLambda     = tri->gradBarycentric();

                for (size_t comp_b = 0; comp_b < 2; ++comp_b) {
                    for (const auto &v_b : tri.vertices()) {
                        M32d J_grad_v_b = J.col(comp_b) * BtGradLambda.col(v_b.localIndex()).transpose();
                        const Real div_v_b  = gradLambda(comp_b, v_b.localIndex());

                        for (size_t comp_a = 0; comp_a < 3; ++comp_a) {
                            for (const auto &v_a : tri.vertices()) {
                                VSFJ dF_a(comp_a, BtGradLambda.col(v_a.localIndex()));
                                VSFJ div_v_b_dF_a_minus_dF_a_grad_v_b(comp_a, div_v_b                               * BtGradLambda.col(v_a.localIndex())
                                                                             - gradLambda(comp_b, v_a.localIndex()) * BtGradLambda.col(v_b.localIndex()));

                                using DeltaF = typename VSFJ::MatrixType;
                                Real contrib = -ted.d2energy(J_grad_v_b,                 dF_a.operator DeltaF()) +
                                                ted. denergy(div_v_b_dF_a_minus_dF_a_grad_v_b.operator DeltaF());
                                visitor(s.varIdx(sheetIdx, v_a.index(), comp_a),
                                        v_b.index(), comp_b, nondimScale * contrib * tri->volume());
                            }
                        }
                    }
                }
            }
        }
    }

    MX2d apply_d2E_dXdx(Eigen::Ref<const VXd> delta_x) const {
        BENCHMARK_START_TIMER_SECTION("apply_d2E_dXdx");
        if (size_t(delta_x.rows()) != numEquilibriumVars()) throw std::runtime_error("Incorrect size of delta_x");
        MX2d result = MX2d::Zero(numVertices(), 2);

        auto accumulatePerTriContrib = [this, &delta_x](size_t tri_idx, MX2d &out) {
                visit_d2E_dxdX([&out, &delta_x](size_t deformedVarIdx, size_t restVtx, size_t restVtxComponent, Real entry) {
                    out(restVtx, restVtxComponent) += delta_x[deformedVarIdx] * entry;
                }, tri_idx);
            };

        assemble_parallel(accumulatePerTriContrib, result, mesh().numElements());
        BENCHMARK_STOP_TIMER_SECTION("apply_d2E_dXdx");
        return result;
    }

    VXd apply_d2E_dxdX(Eigen::Ref<const MX2d> delta_X) const {
        BENCHMARK_START_TIMER_SECTION("apply_d2E_dxdX");
        if (size_t(delta_X.rows()) != numVertices()) throw std::runtime_error("Incorrect size of delta_X");
        VXd result = VXd::Zero(numEquilibriumVars());

        auto accumulatePerTriContrib = [this, &delta_X](size_t tri_idx, VXd &out) {
                visit_d2E_dxdX([&out, &delta_X](size_t deformedVarIdx, size_t restVtx, size_t restVtxComponent, Real entry) {
                                    out[deformedVarIdx] += delta_X(restVtx, restVtxComponent) * entry;
                }, tri_idx);
            };
        assemble_parallel(accumulatePerTriContrib, result, mesh().numElements());
        BENCHMARK_STOP_TIMER_SECTION("apply_d2E_dxdX");
        return result;
    }

    VXd apply_d2E_dxdX_unaccelerated(Eigen::Ref<const MX2d> delta_X) const {
        BENCHMARK_START_TIMER_SECTION("apply_d2E_dxdX");
        if (size_t(delta_X.rows()) != numVertices()) throw std::runtime_error("Incorrect size of delta_X");
        VXd result = VXd::Zero(numEquilibriumVars());

        visit_d2E_dxdX_unaccelerated([&result, &delta_X](size_t deformedVarIdx, size_t restVtx, size_t restVtxComponent, Real entry) {
                result[deformedVarIdx] += delta_X(restVtx, restVtxComponent) * entry;
            });
        BENCHMARK_STOP_TIMER_SECTION("apply_d2E_dxdX");
        return result;
    }

    const VXd &adjointFittingState() const { return m_adjointStateFitting; }

    VXd gradient(EnergyType etype = EnergyType::Full) const {
        const auto &n = nondimensionalization();
        MX2d perRestVtxGradient(MX2d::Zero(numVertices(), 2));
        if ((etype == EnergyType::Full) || (etype == EnergyType::CollapseBarrier)) {
            m_collapseBarrier->accumulateGradient(perRestVtxGradient);
            perRestVtxGradient *= n.collapseBarrierScale();
        }
        if ((etype == EnergyType::Full) || (etype == EnergyType::Fitting))
            perRestVtxGradient -= n.fittingEnergyScale() * apply_d2E_dXdx(m_adjointStateFitting);
        if ((etype == EnergyType::Full) || (etype == EnergyType::Smoothing))
            m_smoothness->accumulateGradient(perRestVtxGradient, mesh(), originalMesh(), n);
        if ((etype == EnergyType::Full) || (etype == EnergyType::CompressionPenalty)) {
            if (compressionPenaltyWeight != 0.0) {
                perRestVtxGradient += (compressionPenaltyWeight * n.fittingEnergyScale()) * (
                                        m_compressionPenalty->dJ_dX() - apply_d2E_dXdx(m_adjointStateCompressionPenalty));
            }
        }

        perRestVtxGradient *= n.restVarScale(); // account for nondimensionalizing change of variables
        VXd result(numVars());
        m_wallPosInterp->adjoint(perRestVtxGradient, wallRestPositionsFromVariables(result));
        return result;
    }

    const CB &collapseBarrier() const { return *m_collapseBarrier; }
    const FusingCurveSmoothness &fusingCurveSmoothness() const { return *m_smoothness; }
          FusingCurveSmoothness &fusingCurveSmoothness()       { return *m_smoothness; }

    const CompressionPenalty &compressionPenalty() const { return *m_compressionPenalty; }
          CompressionPenalty &compressionPenalty()       { return *m_compressionPenalty; }

    std::shared_ptr<WPI> wallPositionInterpolator() const  { return m_wallPosInterp; }

    NewtonOptimizer &getEquilibriumSolver() { return *m_equilibriumSolver; }

    ////////////////////////////////////////////////////////////////////////////
    // Serialization + cloning support (for pickling)
    // Note: this does *not* recover the linesearch state, but rather it resets
    // the last committed (known good) state.
    // Also, the *un-normalized* design variables are serialized.
    ////////////////////////////////////////////////////////////////////////////
    using StateBackwardsCompat = std::tuple<std::shared_ptr<TargetAttractedInflation>, std::shared_ptr<CB>, std::shared_ptr<FusingCurveSmoothness>, std::shared_ptr<WPI>, std::vector<size_t>, VXd, VXd, NewtonOptimizerOptions, bool>; // Before CompressionPenalty was added
    using State                = std::tuple<std::shared_ptr<TargetAttractedInflation>, std::shared_ptr<CB>, std::shared_ptr<FusingCurveSmoothness>, std::shared_ptr<WPI>, std::vector<size_t>, VXd, VXd, NewtonOptimizerOptions, bool, std::shared_ptr<CompressionPenalty>, Real>; // Before CompressionPenalty was added
    static State serialize(const ReducedSheetOptimizer &rso) {
        return std::make_tuple(rso.m_targetAttractedInflation, rso.m_collapseBarrier, rso.m_smoothness, rso.m_wallPosInterp,
                               rso.fixedEquilibriumVars(),
                               rso.m_committedVars,
                               rso.m_committedEquilibrium,
                               rso.m_equilibriumSolver->options,
                               rso.useFirstOrderPrediction,
                               rso.m_compressionPenalty,
                               rso.compressionPenaltyWeight);
    }

    static std::unique_ptr<ReducedSheetOptimizer> deserialize(const StateBackwardsCompat &state, std::shared_ptr<CompressionPenalty> cp = nullptr) {
        auto rso = std::unique_ptr<ReducedSheetOptimizer>(new ReducedSheetOptimizer()); // std::make_shared can't call private constructor
        rso->m_targetAttractedInflation     = std::get<0>(state);
        rso->m_collapseBarrier              = std::get<1>(state);
        rso->m_smoothness                   = std::get<2>(state);
        rso->m_wallPosInterp                = std::get<3>(state);

        const std::vector<size_t> &fv       = std::get<4>(state);
        const VXd &committedVars            = std::get<5>(state);
        const VXd &committedEq              = std::get<6>(state);
        const NewtonOptimizerOptions &eopts = std::get<7>(state);
        rso->useFirstOrderPrediction        = std::get<8>(state);

        rso->m_compressionPenalty = cp;
        if (!rso->m_compressionPenalty) {
            // The compressionPenalty member wasn't serialized, so we need to create a fresh one
            // (otherwise `m_updateAdjointState` will crash, as will `energy` if
            // the user later sets a nonzero compressionPenaltyWeight.)
            rso->m_compressionPenalty = std::make_shared<CompressionPenalty>(rso->targetAttractedInflation().sheetPtr());
        }

        auto &tai = *rso->m_targetAttractedInflation;

        rso->m_committedVars        = committedVars;
        rso->m_committedEquilibrium = committedEq;
        tai.setVars(committedEq); // Roll back sheet to committed equilibrium (in case it was pickled in a linesearch state)

        // Set up equilibrium solver
        rso->m_equilibriumSolver = get_inflation_optimizer(*(rso->m_targetAttractedInflation), fv, eopts);

        // Initialize remaining state
        rso->m_linesearchVars = VXd::Zero(committedVars.size()); // needed for setVars to detect a "new" equilibrium
        bool success = rso->setVars(rso->getCommittedVars()); // getCommittedVars() gets the normalized vars
        if (!success) throw std::runtime_error("Failed to solve equilbrium problem");
        rso->commitDesign();

        return rso;
    }
    static std::unique_ptr<ReducedSheetOptimizer> deserialize(const State &state) {
        auto rso = deserialize(std::make_tuple(std::get<0>(state), std::get<1>(state), std::get<2>(state), std::get<3>(state), std::get<4>(state), std::get<5>(state), std::get<6>(state), std::get<7>(state), std::get<8>(state)), std::get<9>(state));
        rso->compressionPenaltyWeight = std::get<10>(state);
        return rso;
    }

    std::unique_ptr<ReducedSheetOptimizer> cloneForNewTAIAndFixedVars(std::shared_ptr<TargetAttractedInflation> tas, const std::vector<size_t> &fv) const {
        // Clone the compression penalty for the new sheet.
        auto cp = std::make_shared<CompressionPenalty>(tas->sheetPtr());
        cp->modulation      = m_compressionPenalty->modulation->clone();
        cp->includeSheetTri = m_compressionPenalty->includeSheetTri;
        cp->Etft_weight     = m_compressionPenalty->Etft_weight;

        return deserialize(
                std::make_tuple(tas, // already a clone
                                m_collapseBarrier->clone(),
                                m_smoothness->clone(),
                                m_wallPosInterp->clone(),
                                fv, m_committedVars, tas->getVars(), m_equilibriumSolver->options, useFirstOrderPrediction, // ordinary values not needing needing cloning
                                cp,
                                compressionPenaltyWeight));
    }

    ////////////////////////////////////////////////////////////////////////////
    // Public memeber variables
    ////////////////////////////////////////////////////////////////////////////
    bool useFirstOrderPrediction = true;
    Real compressionPenaltyWeight = 0.0;
private:
    ReducedSheetOptimizer() { } // Empty constructor needed for deserialization

    std::shared_ptr<TargetAttractedInflation> m_targetAttractedInflation;
    std::shared_ptr<CB>                       m_collapseBarrier;
    std::shared_ptr<FusingCurveSmoothness>    m_smoothness;
    std::shared_ptr<WPI>                      m_wallPosInterp;
    std::shared_ptr<CompressionPenalty>       m_compressionPenalty;
    std::vector<size_t>                       m_fixedVars;
    VXd                                       m_committedVars, m_linesearchVars;
    VXd                                       m_committedEquilibrium, m_linesearchEquilibrium;
    VXd                                       m_adjointStateFitting, m_adjointStateCompressionPenalty;
    MX2d                                      m_committedRestPositions, m_linesearchRestPositions;
    std::unique_ptr<NewtonOptimizer>          m_equilibriumSolver;

    void m_updateAdjointState() {
        auto &solver = m_equilibriumSolver->solver;
        ///////////////////////////////////////////////////////////////////////
        // Sensitivity analysis
        ///////////////////////////////////////////////////////////////////////
        // Update the adjoint state for the target fitting objective
        // Note: if the Hessian modification failed (tau runaway), the adjoint state
        // solves will fail. To keep the optimizer from giving up entirely, we simply
        // set the adjoint state to 0 in these cases. Presumably this only happens
        // at bad iterates that will be discarded anyway.
        //
        // It would be more efficient to solve for a single adjoint state vector
        // for the composite objective, but we currently compute separate adjoint
        // state vectors for each objective term depending on `x`. The extra
        // backsolve shouldn't be a significant bottleneck, and it gives us the
        // flexibility to change weights/exclude terms (as allowed by our
        // energy/gradient interface) without calling `m_updateAdjointState`.
        try {
            Eigen::VectorXd dFit_dx = m_equilibriumSolver->removeFixedEntries(m_targetAttractedInflation->gradUnweightedTargetFit().cast<double>());
            Eigen::VectorXd dCP_dx  = m_equilibriumSolver->removeFixedEntries(nondimensionalization().equilibriumVarScale() * m_compressionPenalty->dJ_dx().cast<double>());
            // expressed in terms of rescaled variables
            m_adjointStateFitting            = m_equilibriumSolver->extractFullSolution(solver.solve(dFit_dx)).cast<Real>();
            m_adjointStateCompressionPenalty = m_equilibriumSolver->extractFullSolution(solver.solve(dCP_dx )).cast<Real>();
        }
        catch (...) {
            std::cerr << "Adjoint state solve failed." << std::endl;
            m_adjointStateFitting.setZero(numEquilibriumVars());
        }
    }
};

#endif /* end of include guard: REDUCEDSHEETOPTIMIZER_HH */

import numpy as np
from numpy.linalg import norm
import pickle, gzip, os
from enum import Enum

import MeshFEM
import py_newton_optimizer, benchmark
import inflation, mesh, utils
import copy

from datetime import datetime

from opt_config import Solver

class OptimizationReport:
    def __init__(self):
        self.gradNorms = []
        self.energies = []

    def add(self, rso):
        self.gradNorms.append(norm(rso.gradient()))
        self.energies.append({name: rso.energy(etype) for name, etype in rso.EnergyType.__members__.items()})

# A common failure case is when the commited equilibrium is either not actually
# an equilibrium (backtracking failed) or is a suboptimal local optimum that we
# can be perturbed out of by arbitrarily small design changes; when the lower energy
# equilibrium that these perturbed iterates settle down into has a higher objective
# value, the optimizer is stuck because it refuses to accept an energy increase.
# We check for this situation whenever the objective value increases in the linesearch
# by checking if the new equilibrium is actually a lower energy state of the committed design.
# If so, we force the optimizer to restart with the perturbed design (at a higher objective value).
class BadCommittedEquilibrium(Exception):
    pass

class CustomOptimizerResult:
    def __init__(self, msg):
        self.message = msg

class RenderMode(Enum):
    NONE      = 0
    PYTHREEJS = 1
    OFFSCREEN = 2

# Load a sheet_opt instance from a pickle file
def load(path):
    sheet_opt = pickle.load(gzip.open(path, 'rb'))
    sheet_opt.targetAttractedSheet = sheet_opt.rso.targetAttractedInflation()

    # Viewers weren't pickled... but reconstruct at least their viewpoint.
    sheet_opt._generateViewers()
    if hasattr(sheet_opt, 'viewer_pickle_state'):
        sheet_opt.deploy_viewer.setCameraParams(sheet_opt.viewer_pickle_state[0])
        sheet_opt.  flat_viewer.setCameraParams(sheet_opt.viewer_pickle_state[2])
        sheet_opt.deploy_viewer.renderer.matModel = sheet_opt.viewer_pickle_state[1]
        sheet_opt.  flat_viewer.renderer.matModel = sheet_opt.viewer_pickle_state[3]

    # Reduce disk i/o (these settings weren't pickled :()
    sheet_opt.opts.stdoutFlushInterval = 20
    sheet_opt.rso.getEquilibriumSolver().options.stdoutFlushInterval = 20

    # Resume recording video in a fresh file (if the pickled optimizer was recording)
    sheet_opt.screenshotPath = sheet_opt._screenshotPath

    # `sheet_opt.minimize` function was not pickled; reconstruct it
    sheet_opt.setSolver(sheet_opt.solver)

    return sheet_opt

class PySheetOptimizer():
    def __init__(self, targetAttractedSheet, fixedVars = [], renderMode = RenderMode.PYTHREEJS,
                 detActivationThreshold = 0.9, detActivationThresholdTubeTri = 0.5,
                 checkpointPath = None, originalDesignMesh = None, uninflatedDefoInit = None,
                 screenshotPath = None, sheetOutputDir = None, badEquilibriumDir = None,
                 fusingCurveSmoothnessConfig = None, wallStretchBarrier = [1.75, 2.25],
                 customConfigCallback = None):
        """
        PySheetOptimizer

        Arguments
        ---------
        targetAttractedSheet
            The sheet to optimize.
        fixedVars
            Variables to hold fixed in the simulation of the sheet's inflation
            (e.g., boundary vertices).
        renderMode
            Whether/how to visualize the current design/equilibrium.
        detActivationThreshold
            Area shrinkage factor below which the collapse barrier is activated.
        detActivationThresholdTubeTri
            A different area shrinkage factor to use for tube triangles
            (as opposed to wall triangles). If set to `None`,
            `detActivationThreshold` is used for all triangles.
        checkpointPath
            Path to write gzipped pickled state to.
        originalDesignMesh
            The original sheet design with respect to which collapse and smoothing energies are measured.
            (Defaults to the current design mesh in `targetAttractedSheet`.)
        uninflatedDefoInit
            The initial top sheet deformation to be used as the starting point for reinflation.
            (Used only by `reinflatedSheet`.)
        screenshotPath
            Directory or .mp4 path for writing iterate renderings.
        sheetOutputDir
            Directory for writing each iterate's `targetAttractedSheet`.
        badEquilibriumDir
            Directory for targetAttractedSheet's two equilibria when a "bad
            equilibrium" is encountered (when the currently tracked equilibrium
            becomes unstable).
        fusingCurveSmoothnessConfig
            Class whose `apply` method will configure the fusing curve smoothness weights.
        wallStretchBarrier = [activationLevel, limit]
            If not None, start penalizing stretches when they exceed `activationLevel`, ramping up to infinite penalty at `limit`
        customConfigCallback
            A callback function that allows additional configuration to be performed on the new instance before the initial design is committed.
        """
        opts = py_newton_optimizer.NewtonOptimizerOptions()
        opts.useIdentityMetric = True
        opts.beta = 1e-4
        opts.gradTol = 1e-6
        opts.niter = 200
        opts.stdoutFlushInterval = 20 # Reduce disk i/o
        opts.useNegativeCurvatureDirection = False
        self.opts = opts

        self.solver = Solver.SCIPY
        self.minimize = None # function pointer will be set based on self.solver by `optimize`

        self.uninflatedDefoInit = uninflatedDefoInit

        self.renderMode = renderMode

        self.cachedGradient = None
        self.report = OptimizationReport()

        initialVars = []

        self.targetAttractedSheet = targetAttractedSheet
        self.rso = inflation.ReducedSheetOptimizer(targetAttractedSheet, fixedVars, opts, detActivationThreshold, initialVars, originalDesignMesh=originalDesignMesh)

        if (detActivationThresholdTubeTri is not None):
            iwt = self.rso.sheet().isWallTri
            self.rso.collapseBarrier().setActivationThresholds(
                    [detActivationThreshold if iwt(i) else
                     detActivationThresholdTubeTri for i in range(self.rso.mesh().numTris())])

        if (wallStretchBarrier is not None):
            iwt = self.rso.sheet().isWallTri
            nt = self.rso.mesh().numTris()
            self.rso.collapseBarrier().setStretchBarrierActiviations([wallStretchBarrier[0]] * nt)
            self.rso.collapseBarrier().setStretchBarrierLimits      ([wallStretchBarrier[1]] * nt)
            self.rso.collapseBarrier().setApplyStretchBarriers      ([iwt(t) for t in range(nt)])

        if fusingCurveSmoothnessConfig is not None:
            fusingCurveSmoothnessConfig.apply(self.rso)

        self._generateViewers()

        self._screenshotPath = None
        self._screenshotDir  = None # differs from _screenshotPath if _screenshotPath is a .mp4
        self.sheetOutputDir      = utils.freshPath(sheetOutputDir)
        self.badEquilibriumDir   = badEquilibriumDir if badEquilibriumDir is not None else self.sheetOutputDir
        self.checkpointPath      = checkpointPath
        self.checkpointFrequency = 120 # every two minutes
        self.last_checkpoint     = datetime.now()

        if customConfigCallback is not None: customConfigCallback(self)

        self.committedVars = None # initialized in callback
        self.opt_callback(self.rso.getVars())

        # Set the screenshot paths/start recording after the callback so that we do not
        # render a frame before the user gets a chance to configure the camera
        self.screenshotPath = screenshotPath

    @property
    def screenshotPath(self): return self._screenshotPath

    @screenshotPath.setter
    def screenshotPath(self, path):
        """
        Record video in `path[:-4].deploy.mp4` and `path[:-4].flat.mp4` if `path` ends in `.mp4`;
        otherwise record screen shot images in `path/deploy_*.png` and `path/flat_*.png`.
        """
        if path is None:
            self._screenshotPath = None
            return
        if not self.visualizationEnabled(): raise Exception('Visualization disabled')
        if (path[-4:] == '.mp4'):
            outDir = os.path.dirname(path)
            if outDir != '': os.makedirs(outDir, exist_ok=True) # Create the enclosing directories, if necessary
            vidPathPrefix = utils.freshPath(path[:-4], suffix='.deploy.mp4', excludeSuffix=True)
            downscaled = [s // 2 for s in self.deploy_viewer.getSize()]
            self.deploy_viewer.recordStart(vidPathPrefix + '.deploy.mp4', streaming=True, outWidth=downscaled[0], outHeight=downscaled[1])
            self.  flat_viewer.recordStart(vidPathPrefix + '.flat.mp4',   streaming=True, outWidth=downscaled[0], outHeight=downscaled[1])
            self._screenshotDir = outDir
        else:
            self._screenshotDir = utils.freshPath(path)
            os.makedirs(self._screenshotDir, exist_ok=False)
        self._screenshotPath = path

    def visualizationEnabled(self):
        return self.renderMode != RenderMode.NONE

    def recordingVideo(self):
        if not self.visualizationEnabled(): return False
        drec = self.deploy_viewer.isRecording()
        frec = self.  flat_viewer.isRecording()
        if (drec != frec): raise Exception('Inconsistent recording state')
        return drec

    def _generateViewers(self):
        if not self.visualizationEnabled(): return
        import visualization
        if self.renderMode == RenderMode.PYTHREEJS:
            offscreen = False
            width, height = 640, 512
        elif self.renderMode == RenderMode.OFFSCREEN:
            offscreen = True
            width, height = 5120, 4096
        else: raise Exception('Unexpected RenderMode')
        isheet = self.targetAttractedSheet.sheet()
        self.flat_viewer, self.deploy_viewer = visualization.optimizationViewers(isheet, width, height, offscreen)
        self.deploy_viewer.scalarFieldGetter = visualization.ISheetScalarField.TGT_DIST(isheet, utils.getTargetSurf(self.targetAttractedSheet)) # visualize distance to the target surf

    def viewer(self):
        if (self.renderMode != RenderMode.PYTHREEJS):
            raise Exception('Interactive viewer not enabled')
        from ipywidgets import HBox
        return HBox([self.deploy_viewer.show(), self.flat_viewer.show()])

    def writeScreenshot(self):
        """
        Write screenshots of the deployed and flat state to `_screenshotDir`. This
        can be done even when a video is being recorded (in which case the screenshots
        will be placed in the directory containing the recording).
        """
        if (self._screenshotDir is None) or (not self.visualizationEnabled()): return
        if not hasattr(self, '_screenshotDirCreated'):
            os.makedirs(self._screenshotDir, exist_ok=True)
            self._screenshotDirCreated = True
        self.deploy_viewer.writeScreenshot(f'{self._screenshotDir}/deploy_{self.iterateNum()}.png')
        self.  flat_viewer.writeScreenshot(f'{self._screenshotDir}/flat_{self.iterateNum()}.png')

    def updateViewer(self):
        if self.visualizationEnabled():
            self.deploy_viewer.update(scalarField=self.deploy_viewer.scalarFieldGetter(self.rso.sheet()))
            self.  flat_viewer.update(scalarField=self.  flat_viewer.scalarField)

    def energy(self, x):
        print(f'Energy eval called at dist: {norm(x - self.committedVars)}', flush=True)
        # sys.stdout.flush()
        eCurr = self.report.energies[-1]['Full']
        bailed = not self.rso.setVars(x, bailEarlyOnCollapse=True, bailEnergyThreshold=1.0001 * eCurr)
        e = self.rso.energy()

        print(f'\tEvaluated energy {e}; (bail = {bailed})', flush=True)
        # SciPy's l-bfgs-b doesn't seem to like infinite values. Return a significantly larger objective value than the committed design iterate's.
        if (np.isinf(e) or bailed):
            if (not bailed): print("Bail criterion failed...")
            if (self.solver == Solver.SCIPY):
                # gbad = self.rso.gradient()
                # print(f'Gradient norm in bailed config is: ', np.linalg.norm(gbad)) # this can be `nan` or `inf` in bailed configurations!
                bailed2 = not self.rso.setVars(self.rso.getCommittedVars(), bailEarlyOnCollapse=True, bailEnergyThreshold=1.0001 * eCurr)
                if  bailed2: raise Exception('Bailed when setting committed variables')
                e = 2.0 * self.rso.energy()
                # Lie about the gradient too, since a gradient with `inf` or `nan` values confuses Scipy's linesearch
                self.cachedGradient = -self.rso.gradient()
                self.varsForCachedGradient = x.copy()
                print(f'Returning energy {e}')
                return e
            else: return np.inf

        # Cache the gradient since the following check for bad equilibria will lose
        # the currently computed equilibrium, making the subsequent gradient call
        # expensive.
        self.varsForCachedGradient = x
        self.cachedGradient = self.rso.gradient()

        # Check for an increase in energy due to settling down into a more stable/lower
        # energy equilibrium.
        if (len(self.report.energies) > 1) and (e > self.report.energies[-1]['Full']):
            perturbedEquilibrium = self.targetAttractedSheet.getVars()
            self.rso.setVars(self.committedVars) # revert to commmited design/equilibrium
            committedEquilibriumEnergy = self.targetAttractedSheet.energy()
            # Is the energy at the perturbedEquilibrium actually lower than at our commited equilibrium?
            self.targetAttractedSheet.setVars(perturbedEquilibrium)
            print(f'Perturbed configuration energy {self.targetAttractedSheet.energy()} vs committed equilibrium energy {committedEquilibriumEnergy}', flush=True)
            if (self.targetAttractedSheet.energy() < committedEquilibriumEnergy):
                # If so, re-apply the the design change and signal the committed equilibrium was bad.
                self.rso.setVars(x)
                raise BadCommittedEquilibrium()
        return e

    def gradient(self, x):
        print(f'Gradient eval called at dist: {norm(x - self.committedVars)}', flush=True)
        if (self.cachedGradient is not None) and (norm(x - self.varsForCachedGradient) < 1e-16):
            return self.cachedGradient
        self.rso.setVars(x, bailEarlyOnCollapse=True)
        return self.rso.gradient()

    def save(self, path):
        # Our offscreen renderer doesn't currently pickle...
        visEnabled = self.visualizationEnabled()
        if visEnabled:
            deploy_viewer = self.deploy_viewer
            flat_viewer   = self.flat_viewer
            self.deploy_viewer = None
            self.flat_viewer   = None

        # IPopt minimizer cannot be pickled
        minimize = self.minimize
        self.minimize = None

        # It's redundant to pickle targetAttractedSheet since this is pickled as part of self.rso
        tas = self.targetAttractedSheet
        self.targetAttractedSheet = None

        if visEnabled:
            if hasattr(deploy_viewer.renderer, 'matModel'): # Only the offscreen renderer has this...
                self.viewer_pickle_state = (deploy_viewer.getCameraParams(), deploy_viewer.renderer.matModel,
                                              flat_viewer.getCameraParams(),   flat_viewer.renderer.matModel)
        pickle.dump(self, gzip.open(path, 'wb'))

        if visEnabled:
            self.deploy_viewer = deploy_viewer
            self.flat_viewer   = flat_viewer

        self.minimize = minimize

        self.targetAttractedSheet = tas

    @staticmethod
    def load(path):
        return load(path)

    def checkpoint(self):
        if (not self.checkpointPath): return
        now = datetime.now()
        if (now - self.last_checkpoint).seconds >= self.checkpointFrequency:
            sheet = self.rso.sheet()
            # Minimize risk of corruption/truncation of critical checkpoint
            # data in case the job is killed during checkpointing...
            self.save(self.checkpointPath + '.tmp')
            os.rename(self.checkpointPath + '.tmp', self.checkpointPath)
            self.last_checkpoint = now

    def iterateNum(self):
        return len(self.report.energies) - 1

    def writeSheet(self):
        if self.sheetOutputDir is None: return
        # Do not enforce a fresh directory here so that resuming an
        # optimization continues writing sheets where we left off.
        os.makedirs(self.sheetOutputDir, exist_ok=True)
        pickle.dump(self.targetAttractedSheet, gzip.open(self.sheetOutputDir + f'/tas_{self.iterateNum()}.pkl.gz', 'wb'))

    def opt_callback(self, x):
        rso = self.rso
        self.updateViewer()
        self.report.add(rso)
        rso.commitDesign()
        print("Committed design with energies:\n", self.report.energies[-1])
        print("\tgradient norm:", self.report.gradNorms[-1], flush=True)
        self.committedVars = rso.getVars()

        if (self._screenshotDir is not None) and (self.visualizationEnabled()):
            if not self.recordingVideo(): self.writeScreenshot() # This frame is already written if we are recording a video
        self.writeSheet()

        self.checkpoint()

        return False

    def setSolver(self, solver, maxIters=1000):
        if (solver == Solver.SCIPY):
            import scipy
            import scipy.optimize
            self.minimize = lambda x0: scipy.optimize.minimize(self.energy, x0, jac=self.gradient, method='L-BFGS-B', callback=self.opt_callback, tol=0, options=dict(maxiter=maxIters, gtol=1e-16)).message
        elif (solver == Solver.IPOPT):
            import ipopt
            class IPOptWrapper(ipopt.problem):
                def __init__(self, sheet_opt):
                    numConstraints = 0
                    super().__init__(n=sheet_opt.rso.numVars(), m=numConstraints, problem_obj=self, lb=None, ub=None, cl=None, cu=None)
                    self.sheet_opt = sheet_opt
                def   objective(self, x): return self.sheet_opt.energy(x)
                def    gradient(self, x): return self.sheet_opt.gradient(x)
                def constraints(self, x): return np.empty()
                def    jacobian(self, x): return np.empty()
                def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
                    if (iter_count > 0): # Ipopt actually calls this callback before doing any optimization for the initial iterate which has already been committed...
                        self.sheet_opt.opt_callback(None)
                def solve(self, x0):
                    self.addOption('mu_strategy',           'adaptive')
                    self.addOption('max_iter',              maxIters)
                    self.addOption('tol',                   1e-8)
                    self.addOption('hessian_approximation', 'limited-memory') # use l-bfgs
                    self.addOption('max_soc',               0) # disable second-order corrections: we don't have constraints, and the corrections seem to cause the linesearch to repeatedly evaluate around bad trial points before backtracking
                    #self.addOption('print_level',           0) # be quiet
                    x, info = super().solve(x0)
                    return info['status_msg']
            self.minimize = lambda x0: IPOptWrapper(self).solve(x0)
        else: raise Exception('Unknown solver')
        self.solver = solver

    def optimize(self, doBenchmark = True):
        if self.minimize is None: self.setSolver(Solver.SCIPY) # use SCIPY if no solver is specified.
        self.last_checkpoint = datetime.now()

        # Write a screenshot of the initial design (i.e., if we aren't
        # resuming the optimization.)
        # We must do this here since the user may have changed
        # the camera parameters after `opt_callback` was called...
        if self.iterateNum() == 0:
            self.updateViewer() # rerender to ensure all applied changes appear in the first frame
            if self.recordingVideo():
                self.deploy_viewer.recorder.writeFrame()
                self.  flat_viewer.recorder.writeFrame()
            else: self.writeScreenshot()

        max_num_bad_equilibria = 8
        if doBenchmark: benchmark.reset()
        for i in range(max_num_bad_equilibria):
            try:
                res = self.minimize(self.rso.getVars())
                print(f'Optimizer terminated with message: {res}')
                break
            except BadCommittedEquilibrium:
                # Force the optimizer to accept the higher-objective linesearch
                # iterate whose equilibrium configuration actually attains a
                # lower potential energy for the committed design than the
                # committed equilibrium
                print("Bad equilibrium detected; forcing a jump to a lower potential energy (but higher objective) design/equilibrium.", flush=True)
                if self.badEquilibriumDir is not None:
                    os.makedirs(self.badEquilibriumDir, exist_ok=True)
                    path = utils.freshPath(self.badEquilibriumDir + '/badEq', suffix='.pkl.gz', excludeSuffix=False)
                    badTAI = copy.deepcopy(self.rso.targetAttractedInflation())
                    badTAI.setRestVertexPositions(self.rso.getCommittedRestPositions())
                    badTAI.setVars(self.rso.getCommittedEquilibrium())
                    pickle.dump({'goodEqTAI': self.rso.targetAttractedInflation(),
                                 'badEqTAI': badTAI}, gzip.open(path, 'wb'))
                self.opt_callback(None)
                res = CustomOptimizerResult('Bad committed equilibrium')
        if doBenchmark: benchmark.report()
        return res, self.report

    def finish(self):
        # Ensure videos are fully written out
        # (in `try` in case recording viewers don't exist)
        try:
            self.flat_viewer.  recordStop()
            self.deploy_viewer.recordStop()
        except: pass

    def reinflatedSheet(self, configCallback=None, reinflationVideoPath=None):
        """
        Reinflate the current design from `self.uninflatedDefoInit` or the
        identity deformation if it is not present. A valid `self.uninflatedDefoInit`
        must exist if the equilibrium problem has fixed variables.

        Parameters
        ----------
        configCallback
                `configCallback(reinflateTAI)` is called before inflation,
                allowing the calling code to further configure the sheet (e.g.,
                disable tension field theory or set a different target attraction weight).
        reinflationVideoPath
                Path to which a video of the reinflation is recorded.
        """
        reinflateTAI = copy.deepcopy(self.rso.targetAttractedInflation())
        fv = self.rso.fixedEquilibriumVars()
        if self.uninflatedDefoInit is None:
            if len(fv) != 0: raise Exception('An initialization (uninflatedDefoInit) is needed to satisfy the fixed var constraints')
            reinflateTAI.sheet().setIdentityDeformation()
        else:
            reinflateTAI.sheet().setUninflatedDeformation(self.uninflatedDefoInit.T, prepareRigidMotionPinConstraints=False)
        reinflateTAI.setVars(reinflateTAI.getVars()) # Trigger closest point recomputation

        reinflateTAI.fittingWeight = 1e-10 # Use a very low weight, attempting to find the true equilibrium.
        if configCallback is not None:
            configCallback(reinflateTAI)

        iterCB = None
        rview = None
        if reinflationVideoPath is not None:
            import tri_mesh_viewer
            deploy_colors = self.deploy_viewer.scalarField
            rview = tri_mesh_viewer.OffscreenTriMeshViewer(reinflateTAI.sheet(), scalarField=deploy_colors, width=1920, height=1536)
            rview.setCameraParams(self.deploy_viewer.getCameraParams())
            rview.recordStart(reinflationVideoPath, writeFirstFrame=True)
            iterCB = lambda it: rview.update(scalarField=deploy_colors)

        inflation.inflation_newton(reinflateTAI, fv, self.opts, callback=iterCB)
        return reinflateTAI

    def restartFromEquilibrium(self, eqVars):
        """
        Restart the optimization tracking a potentially new equilibrium reached
        from initial guess `eqVars`.
        This can be useful if the current equilibrium is nearly unstable and the
        user wants to manually perturb to a more stable equilibrium.
        """
        oldEqVars = self.rso.targetAttractedInflation().getVars()
        self.rso.targetAttractedInflation().setVars(eqVars)
        success = self.rso.forceEquilibriumUpdate()
        if success == False:
            self.rso.targetAttractedInflation().setVars(oldEqVars) # roll back
            raise Exception('Reinflation failed')
        self.optimize()

    def reinflateAndRestart(self, configCallback=None, reinflationVideoPath=None):
        """
        Restart the optimization, tracking the equilibrium found by reinflating
        the current optimal design from scratch.
        """
        reinflateTAI = self.reinflatedSheet(configCallback=configCallback, reinflationVideoPath=reinflationVideoPath)
        restartFromEquilibrium(reinflateTAI.sheet().getvars())

    def replaceRSO(self, rso, screenshotPath = None):
        """
        (Hack) swap out this PySheetOptimizer's reduced sheet optimizer and reset its optimization state.
        TODO: add a cloning constructor that can accept a new RSO and copy over the other settings.
        """
        self.rso = rso
        self.targetAttractedSheet = rso.targetAttractedInflation()
        self.report = OptimizationReport()
        self.cachedGradient = None
        if screenshotPath is not None:
            self.renderMode = RenderMode.OFFSCREEN
            self._generateViewers()
        self.screenshotPath = None
        self._screenshotDir = None

        self.opt_callback(self.rso.getVars())
        # Set the screenshot paths/start recording after the callback so that we do not
        # render a frame before the user gets a chance to configure the camera
        self.screenshotPath = screenshotPath

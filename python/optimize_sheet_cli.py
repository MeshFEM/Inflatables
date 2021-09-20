import hpc_optimization_job
import sys, os, time, itertools
import pickle, gzip, subprocess

# WARNING: We avoid importing any inflatables/MeshFEM/numpy/matplotlib packages at the top level to make
# sure that the environment variables set in hpc_optimization_job before calling
# `run` can take effect.

def load(path):
    """
    load a pickled gzip object
    """
    return pickle.load(gzip.open(path, 'rb'))

def save(obj, path):
    """
    save an object to a pickled gzip
    """
    pickle.dump(obj, gzip.open(path, 'wb'))

def optimizer_options():
    import py_newton_optimizer
    opts = py_newton_optimizer.NewtonOptimizerOptions()
    opts.useIdentityMetric = True
    opts.beta = 1e-4
    opts.gradTol = 1e-6
    opts.niter = 5000
    opts.nbacktrack_iter = 18
    opts.stdoutFlushInterval = 20 # Reduce disk i/o
    opts.useNegativeCurvatureDirection = False
    return opts

def getNextUniquePath(path):
    if not os.path.exists(path):
        return path
    name, ext = os.path.splitext(path)

    for i in itertools.count():
        numberedPath = f'{name}.{i}{ext}'
        if (not os.path.exists(numberedPath)):
            return numberedPath

from enum import Enum
class Status(Enum):
    NOT_RUN                      = 0
    EQUILIBRIUM_INCOMPLETE       = 1
    FIRST_OPT_INCOMPLETE         = 2
    LOWFIT_OPT_INCOMPLETE        = 3
    NOHCP_OPT_INCOMPLETE         = 4
    NOFIT_EQUILIBRIUM_INCOMPLETE = 5
    FINAL_EQUILIBRIUM_INCOMPLETE = 6
    SUCCESS                      = 7

def check(jobs, job_index):
    jobdir = jobs.directoryForJob(job_index)
    params = jobs.parametersForJob(job_index)
    if not os.path.isdir(jobdir):                                                                              status = Status.NOT_RUN
    elif not os.path.exists(f'{jobdir}/optimizer.pkl.gz'):                                                     status = Status.EQUILIBRIUM_INCOMPLETE
    elif not os.path.exists(f'{jobdir}/tas_post_opt.pkl.gz'):                                                  status = Status.FIRST_OPT_INCOMPLETE
    elif not os.path.exists(f'{jobdir}/tas_post_lowfit_opt.pkl.gz'):                                           status = Status.LOWFIT_OPT_INCOMPLETE
    elif (params['holdCPFixed'] == 2) and (not os.path.exists(f'{jobdir}/tas_post_lowfit_nohcp_opt.pkl.gz')):  status = Status.NOHCP_OPT_INCOMPLETE
    elif not os.path.exists(f'{jobdir}/tas_nofit.pkl.gz'):                                                     status = Status.NOFIT_EQUILIBRIUM_INCOMPLETE
    elif not os.path.exists(f'{jobdir}/tas_reinflate.pkl.gz'):                                                 status = Status.FINAL_EQUILIBRIUM_INCOMPLETE
    else: status = Status.SUCCESS

    resumable = (status != Status.EQUILIBRIUM_INCOMPLETE) \
            and (status != Status.NOFIT_EQUILIBRIUM_INCOMPLETE) \
            and (status != Status.FINAL_EQUILIBRIUM_INCOMPLETE)
    return status, resumable

def run(jobs, job_index, resuming = False):
    import sheet_optimizer, inflation, benchmark, utils
    import numpy as np

    outdir = jobs.directoryForJob(job_index)
    os.makedirs(outdir, exist_ok=True)

    config = jobs.config

    if config.redirect_io:
        stdoutPath = getNextUniquePath(f'{outdir}/stdout.txt')
        stderrPath = getNextUniquePath(f'{outdir}/stderr.txt')
        print(f"Redirecting stdout to '{stdoutPath}'")
        print(f"Redirecting stderr to '{stderrPath}'")
        sys.stdout = open(stdoutPath, 'w', 4096) # Reduce file system i/o
        sys.stderr = open(stderrPath, 'w', 4096)

    params = jobs.parametersForJob(job_index)
    if config.debug: config.design_opt_iters = 5

    print(params)

    status, resumable = check(jobs, job_index)
    if resuming == False: status = Status.NOT_RUN
    else:
        print(f'resuming job with status {status}')
        if status == Status.SUCCESS:
            print("Nothing to resume")
            return
        elif not resumable:
            raise Exception("Cannot resume from this state--rerun job")

    # Default to scipy's l-bfgs if no solver is specified
    solver = config.solver if hasattr(config, 'solver') else config.Solver.SCIPY

    with inflation.ostream_redirect():
        sheet_opt, sheet_opt_lowfit, sheet_opt_lowfit_nohcp = None, None, None
        report, lowfit_report = None, None
        # Start from scratch
        if status == Status.NOT_RUN:
            targetAttractedSheet = load(params['input']['targetAttractedSheet'])
            isheet = targetAttractedSheet.sheet()
            isheet.setRelaxedStiffnessEpsilon(config.relaxedStiffnessEpsilon)

            fixedVars = params['input']['fixedVars'].get(isheet)

            isheet.setUseTensionFieldEnergy(True)
            isheet.setUseHessianProjectedEnergy(False)
            # TODO: tension field theory in the fused regions?

            # The loaded sheet should be in equilibrium under its applied
            # fitting weight/closest point projection settings, but changing
            # these requires a reinflation...
            targetAttractedSheet.fittingWeight = params['fittingWeight']
            targetAttractedSheet.targetSurfaceFitter().holdClosestPointsFixed = params['holdCPFixed'] > 0
            targetAttractedSheet.setVars(targetAttractedSheet.getVars())
            benchmark.reset()
            inflation.inflation_newton(targetAttractedSheet, fixedVars, optimizer_options())
            benchmark.report()

            print(utils.allEnergies(targetAttractedSheet))
            print(utils.allEnergies(isheet),"\n")

            sheet_opt = sheet_optimizer.PySheetOptimizer(targetAttractedSheet, fixedVars, renderMode=sheet_optimizer.RenderMode.OFFSCREEN, detActivationThreshold=params['cbThreshold'],
                                                         originalDesignMesh=isheet.mesh().copy(), checkpointPath=f'{outdir}/optimizer.pkl.gz', screenshotPath = f'{outdir}/opt.mp4',
                                                         fusingCurveSmoothnessConfig=params['fusingCurveSmoothness'])

            sheet_opt.setSolver(solver, config.design_opt_iters)
            if (config.debug): sheet_opt.checkpointFrequency = 0 # write out a checkpoint at every step!
            cparam = jobs.cameraParamsForJob(job_index)
            if cparam:
                sheet_opt.deploy_viewer.setCameraParams(cparam['deploy'])
                sheet_opt.  flat_viewer.setCameraParams(cparam[  'flat'])
            res, report = sheet_opt.optimize()

        # Resume first optimization round
        if status == Status.FIRST_OPT_INCOMPLETE:
            sheet_opt = sheet_optimizer.load(f'{outdir}/optimizer.pkl.gz')
            sheet_opt.targetAttractedSheet.sheet().setRelaxedStiffnessEpsilon(config.relaxedStiffnessEpsilon)
            sheet_opt.setSolver(solver, config.design_opt_iters)
            res, report = sheet_opt.optimize()

            # Extract the expected local variables from the loaded optimizer
            targetAttractedSheet = sheet_opt.targetAttractedSheet
            fixedVars = sheet_opt.rso.fixedEquilibriumVars()

        if sheet_opt is not None:
            sheet_opt.save(f'{outdir}/optimizer.pkl.gz') # overwrite checkpoint with final optimizer state
            save(targetAttractedSheet, f'{outdir}/tas_post_opt.pkl.gz')
            if report is not None: save(report, f'{outdir}/opt_report.pkl.gz')

        ################################################################################
        # Low fit optimization
        ################################################################################
        if status.value <= Status.LOWFIT_OPT_INCOMPLETE.value:
            print('Running fine-tuning optimization with lower fitting weight')
            lowfitOptCheckpointPath = f'{outdir}/lowfit_optimizer.pkl.gz'

            # If we just finished the first optimization round, set up a new lowfit optimization round
            resumingLowFit = True
            if status.value < Status.LOWFIT_OPT_INCOMPLETE.value:
                # Inflate with a lower target attraction weight to prepare for lowfit optimization
                targetAttractedSheet.fittingWeight = 1e-6
                inflation.inflation_newton(targetAttractedSheet, fixedVars, optimizer_options())
                save(targetAttractedSheet, f'{outdir}/tas_pre_lowfit_opt.pkl.gz')
                resumingLowFit = False
            elif not os.path.exists(lowfitOptCheckpointPath):
                # Recover from a failed sheet_opt_lowfit construction/missing checkpoint...
                # (where `tas_pre_lowfit_opt.pkl.gz` already exists)
                targetAttractedSheet = load(f'{outdir}/tas_pre_lowfit_opt.pkl.gz')
                targetAttractedSheet.sheet().setRelaxedStiffnessEpsilon(config.relaxedStiffnessEpsilon)
                fixedVars = params['input']['fixedVars'].get(targetAttractedSheet.sheet())
                resumingLowFit = False

            if not resumingLowFit:
                del sheet_opt # needed to prevent OSMesa render context from ballooning
                sheet_opt_lowfit = sheet_optimizer.PySheetOptimizer(targetAttractedSheet, fixedVars, renderMode=sheet_optimizer.RenderMode.OFFSCREEN, detActivationThreshold=params['cbThreshold'],
                                                                    originalDesignMesh=load(params['input']['targetAttractedSheet']).sheet().mesh(),
                                                                    uninflatedDefoInit=np.loadtxt(params['input']['uninflatedDefoInit']), # needed for reinflation below...
                                                                    checkpointPath=lowfitOptCheckpointPath, screenshotPath = f'{outdir}/lowfit_opt.mp4',
                                                                    fusingCurveSmoothnessConfig=params['fusingCurveSmoothness'])
                if (config.debug): sheet_opt_lowfit.checkpointFrequency = 0 # write out a checkpoint at every step!
                cparam = jobs.cameraParamsForJob(job_index)
                if cparam:
                    sheet_opt_lowfit.deploy_viewer.setCameraParams(cparam['deploy'])
                    sheet_opt_lowfit.  flat_viewer.setCameraParams(cparam[  'flat'])
            else:
                sheet_opt_lowfit = sheet_optimizer.load(lowfitOptCheckpointPath)
                sheet_opt_lowfit.targetAttractedSheet.sheet().setRelaxedStiffnessEpsilon(config.relaxedStiffnessEpsilon)

            sheet_opt_lowfit.setSolver(solver, config.design_opt_iters)
            res, lowfit_report = sheet_opt_lowfit.optimize()

            # Extract the potentially missing local variables from the optimizer
            targetAttractedSheet = sheet_opt_lowfit.targetAttractedSheet
            fixedVars = sheet_opt_lowfit.rso.fixedEquilibriumVars()

            sheet_opt_lowfit.save(f'{outdir}/optimizer.pkl.gz') # overwrite checkpoint with final optimizer state
            save(targetAttractedSheet, f'{outdir}/tas_post_lowfit_opt.pkl.gz')
            if lowfit_report is not None: save(lowfit_report, f'{outdir}/lowfit_opt_report.pkl.gz')

            final_sheet_opt = sheet_opt_lowfit

        ################################################################################
        # Freed closest point optimization
        ################################################################################
        noHCPRoundRequested = params['holdCPFixed'] == 2
        if noHCPRoundRequested and (status.value <= Status.NOHCP_OPT_INCOMPLETE.value):
            print('Running low fitting weight optimization with closest points freed ')
            nohcpOptCheckpointPath = f'{outdir}/lowfit_nohcp_optimizer.pkl.gz'

            # If we just finished the lowfit optimization round, set up a new nohcp optimization round
            if not os.path.exists(nohcpOptCheckpointPath):
                targetAttractedSheet.targetSurfaceFitter().holdClosestPointsFixed = False
                inflation.inflation_newton(targetAttractedSheet, fixedVars, optimizer_options())
                save(targetAttractedSheet, f'{outdir}/tas_pre_lowfit_nohcp_opt.pkl.gz')

                del sheet_opt_lowfit # needed to prevent OSMesa render context from ballooning
                sheet_opt_lowfit_nohcp = sheet_optimizer.PySheetOptimizer(targetAttractedSheet, fixedVars, renderMode=sheet_optimizer.RenderMode.OFFSCREEN, detActivationThreshold=params['cbThreshold'],
                                                                          originalDesignMesh=load(params['input']['targetAttractedSheet']).sheet().mesh(),
                                                                          uninflatedDefoInit=np.loadtxt(params['input']['uninflatedDefoInit']), # needed for reinflation below...
                                                                          checkpointPath=nohcpOptCheckpointPath, screenshotPath = f'{outdir}/lowfit_nohcp_opt.mp4',
                                                                          fusingCurveSmoothnessConfig=params['fusingCurveSmoothness'])
                cparam = jobs.cameraParamsForJob(job_index)
                if cparam:
                    sheet_opt_lowfit_nohcp.deploy_viewer.setCameraParams(cparam['deploy'])
                    sheet_opt_lowfit_nohcp.  flat_viewer.setCameraParams(cparam[  'flat'])
            else:
                # Resume the nohpc optimization round
                sheet_opt_lowfit_nohcp = sheet_optimizer.load(nohcpOptCheckpointPath)
                sheet_opt_lowfit_nohcp.targetAttractedSheet.sheet().setRelaxedStiffnessEpsilon(config.relaxedStiffnessEpsilon)

            sheet_opt_lowfit_nohcp.setSolver(solver, config.design_opt_iters)
            if (config.debug): sheet_opt_lowfit_nohcp.checkpointFrequency = 0 # write out a checkpoint at every step!
            res, lowfit_nohcp_report = sheet_opt_lowfit_nohcp.optimize()

            save(targetAttractedSheet, f'{outdir}/tas_post_lowfit_nohcp_opt.pkl.gz')
            if lowfit_nohcp_report is not None: save(lowfit_nohcp_report, f'{outdir}/lowfit_nohcp_opt_report.pkl.gz')

            # Extract the potentially missing local variables from the optimizer
            targetAttractedSheet = sheet_opt_lowfit_nohcp.targetAttractedSheet
            fixedVars = sheet_opt_lowfit_nohcp.rso.fixedEquilibriumVars()

            final_sheet_opt = sheet_opt_lowfit_nohcp

        ################################################################################
        # No-fit inflation/reinflation
        ################################################################################
        targetAttractedSheet.fittingWeight = 1e-10
        inflation.inflation_newton(targetAttractedSheet, fixedVars, optimizer_options())
        save(targetAttractedSheet, f'{outdir}/tas_nofit.pkl.gz')

        print('reinflate')
        tas_reinflate = final_sheet_opt.reinflatedSheet(reinflationVideoPath=f'{outdir}/reinflate.mp4')
        save(tas_reinflate, f'{outdir}/tas_reinflate.pkl.gz')

        print('Non-TF reinflated equilibrium solve')
        targetAttractedSheet.sheet().setUseTensionFieldEnergy(False)
        targetAttractedSheet.sheet().setUseHessianProjectedEnergy(True)
        inflation.inflation_newton(targetAttractedSheet, fixedVars, optimizer_options())
        save(targetAttractedSheet, f'{outdir}/tas_reinflate_notf.pkl.gz')

def resume(jobs, job_index):
    run(jobs, job_index, resuming=True)

################################################################################
# Result analysis
################################################################################
def getViewers(isheet, cparam = None):
    import visualization
    if 'flat_viewer' not in getViewers.__dict__:
        getViewers.flat_viewer, getViewers.deploy_viewer = visualization.optimizationViewers(isheet, width=3840, height=3072, offscreen=True)
    else:
        getViewers.deploy_viewer.update(mesh=isheet,        scalarField=getViewers.deploy_viewer.scalarField)
        getViewers.  flat_viewer.update(mesh=isheet.mesh(), scalarField=getViewers.  flat_viewer.scalarField)
    if cparam:
        getViewers.deploy_viewer.setCameraParams(cparam['deploy'])
        getViewers.  flat_viewer.setCameraParams(cparam[  'flat'])
    return getViewers.flat_viewer, getViewers.deploy_viewer

def generateFlipperData(jobs, ji, analysisDir):
    import mesh, utils
    from matplotlib import pyplot as plt
    print('analyzing ', ji)

    if check(jobs, ji)[0] != Status.SUCCESS: return
    flipDataDir = f'{analysisDir}/{jobs.jobName(ji, ignoreParams=jobs.choicelessParams())}'
    os.makedirs(flipDataDir, exist_ok=True)

    ################################################################################
    # Generate convergence plots
    ################################################################################
    fig = plt.figure(figsize=(16, 8))
    t = 'Convergence Plots - High Fitting Weight (Top) and Low Fitting Weight (Bottom)'
    reports = [load(f'{jobs.directoryForJob(ji)}/{prefix}opt_report.pkl.gz') for prefix in ['', 'lowfit_']]
    for r, report in enumerate(reports):
        st = fig.suptitle(t)
        plt.subplot(2, 5, 1 + 5 * r); plt.plot([e['Full']            for e in report.energies]); plt.grid()
        plt.yscale('log')
        plt.title('Full')
        plt.subplot(2, 5, 2 + 5 * r); plt.plot([e['Fitting']         for e in report.energies]); plt.grid()
        plt.title('Fitting')
        plt.yscale('log')
        plt.subplot(2, 5, 3 + 5 * r); plt.plot([e['CollapseBarrier'] for e in report.energies]); plt.grid()
        plt.title('Collapse Barrier')
        plt.subplot(2, 5, 4 + 5 * r); plt.plot([e['Smoothing'] for e in report.energies]); plt.grid()
        plt.title('Smoothing')
        plt.subplot(2, 5, 5 + 5 * r); plt.plot(report.gradNorms); plt.grid()
        plt.title('Gradient Norm')
        plt.yscale('log')
        plt.tight_layout()

    fig.subplots_adjust(top=0.92)
    plt.savefig(flipDataDir + '/convergence_plots.png', dpi=72)
    plt.close()

    jobDir = jobs.directoryForJob(ji)
    tas = load(f'{jobDir}/tas_reinflate.pkl.gz')
    tsf = tas.targetSurfaceFitter()
    isheet = tas.sheet()
    m = tas.sheet().mesh()

    params = jobs.parametersForJob(ji)
    # Visualize the optimized design, but first initialize the viewer with the
    # input design to set the same model matrix for all runs.
    flatview, deployview = getViewers(load(params['input']['targetAttractedSheet']).sheet())
    flatview, deployview = getViewers(isheet, jobs.cameraParamsForJob(ji))
    flatview  .writeScreenshot(flipDataDir + '/opt_design.png')
    deployview.writeScreenshot(flipDataDir + '/opt_deploy.png')
    subprocess.call(['convert', flipDataDir + '/opt_design.png', '-resize', '50%', flipDataDir + '/opt_design.jpg'])
    subprocess.call(['convert', flipDataDir + '/opt_deploy.png', '-resize', '50%', flipDataDir + '/opt_deploy.jpg'])

    tgtSurfV, tgtSurfF = tsf.targetSurfaceV, tsf.targetSurfaceF

    equilibrium_surf = isheet.visualizationMesh()
    xf = utils.renderingNormalization(tgtSurfV, placeAtopFloor = True)
    eqiMesh = flipDataDir + '/xf_equilibrium.obj'
    tgtMesh = flipDataDir + '/target_surf_xf.obj'
    mesh.save(eqiMesh, jobs.config.blenderVisXF(xf(equilibrium_surf.vertices())), equilibrium_surf.triangles())
    mesh.save(tgtMesh, jobs.config.blenderVisXF(xf(tgtSurfV)), tgtSurfF)

    # subprocess.call(['blender', '-b', '../rendering/inflatable_render_blank.blend', '--python', '../rendering/inflatable_render.py', '--', tgtMesh, eqiMesh, flipDataDir + '/equilibrium_with_target.png'])
    # subprocess.call(['blender', '-b', '../rendering/inflatable_render_blank.blend', '--python', '../rendering/inflatable_render.py', '--',          eqiMesh, flipDataDir + '/equilibrium.png'])
    # subprocess.call(['convert', flipDataDir + '/equilibrium_with_target.png', flipDataDir + '/equilibrium_with_target.jpg'])
    # subprocess.call(['convert', flipDataDir + '/equilibrium.png',             flipDataDir + '/equilibrium.jpg'])

    os.unlink(eqiMesh)
    os.unlink(tgtMesh)
    os.unlink(flipDataDir + '/opt_design.png')
    os.unlink(flipDataDir + '/opt_deploy.png')
    # os.unlink(flipDataDir + '/equilibrium_with_target.png')
    # os.unlink(flipDataDir + '/equilibrium.png')

def analyze(jobs, job_index, analysisPath):
    generateFlipperData(jobs, job_index, analysisPath)

if __name__ == '__main__':
    hpc_optimization_job.do_cli(run, analyze, check, resume)

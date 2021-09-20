import inflation, remesh, mesh, os, py_newton_optimizer
import numpy as np
from matplotlib import pyplot as plt

def inflateToPressure(p, isheet, opts = py_newton_optimizer.NewtonOptimizerOptions(), fixedVars = None):
    isheet.pressure = p
    opts.niter = 1000
    if (fixedVars is None):
        fixedVars = isheet.rigidMotionPinVars
    inflation.inflation_newton(isheet, fixedVars, opts)

def inflationPressureAnalysis(isheet, pressures, opts = py_newton_optimizer.NewtonOptimizerOptions(), outDir = None, fixedVars = None):
    integratedGaussCurvatures = []
    tensionStates = []
    maxEigenvalues = []
    contractions = []
    for i, p in enumerate(pressures):
        print(f'{i + 1}/{len(pressures)}: {p}', flush=True)
        inflateToPressure(p, isheet, opts, fixedVars)
        isa = inflation.InflatedSurfaceAnalysis(isheet)
        metric = isa.metric()
        contractions.append(metric.sigma_2)
        tensionStates.append(isheet.tensionStateHistogram())
        maxEigenvalues.append([ted.eigSensitivities().Lambda()[0] for ted in isheet.triEnergyDensities()])
        # print("Remeshing", flush=True)
        isurf = remesh.remesh(isa.inflatedSurface())
        # print("Computing angle deficits", flush=True)
        integratedGaussCurvatures.append(isurf.angleDeficits().sum())
        # print("Constructing CurvatureInfo", flush=True)
        # isurf.save('debug.msh')
        ci = inflation.CurvatureInfo(isurf.vertices(), isurf.elements())
        if (outDir is not None):
            os.makedirs(outDir, exist_ok=True)
            writer = mesh.MSHFieldWriter(f'{outDir}/surf_{p}.msh', isurf.vertices(), isurf.elements())
            writer.addField('gaussCurvature', ci.gaussianCurvature())
            vm = isheet.visualizationMesh()
            writer = mesh.MSHFieldWriter(f'{outDir}/infl_{p}.msh', vm.vertices(), vm.elements())
            writer.addField('maxEig', maxEigenvalues[-1])
    return {'pressures':                  pressures,
            'integratedGaussCurvatures' : integratedGaussCurvatures,
            'tensionStates'             : tensionStates,
            'maxEigenvalues'            : maxEigenvalues,
            'contractions'              : contractions}

def plot(analysisResult):
    plt.figure(figsize=(10, 8))
    pressures = analysisResult['pressures']

    plt.subplot(2, 2, 1)
    plt.plot(pressures, analysisResult['integratedGaussCurvatures'])
    plt.grid()
    plt.title('Integrated curvature')

    plt.subplot(2, 2, 2)
    plt.plot(pressures, [tsh[2] / np.sum(tsh) for tsh in analysisResult['tensionStates']])
    plt.grid()
    plt.title('elements in full tension')

    for i, name in enumerate(['material stretch', 'metric contraction']):
        plt.subplot(2, 2, i + 3)
        for pct in [100, 95, 90, 80, 50]:
            data = [1.0 / c for c in analysisResult['contractions']] if i == 1 else analysisResult['maxEigenvalues']
            plt.plot(pressures, [np.percentile(d, pct) for d in data], label=f'{pct}th percentile')
        if (i == 1):
            plt.axhline(np.pi / 2, color='black', linewidth=1, linestyle=(0, (5, 8)), label='pi / 2')
            plt.ylim(0.9, 1.8)
        else:
            plt.ylim(1.0, 3.0)
        plt.grid()
        plt.legend()
        plt.title(name)
        plt.xlabel('pressure')
    plt.tight_layout(pad=1.08)
    plt.show()

def run(isheet, pressures, opts = py_newton_optimizer.NewtonOptimizerOptions(), outDir = None, fixedVars = None):
    result = inflationPressureAnalysis(isheet, pressures, opts, outDir, fixedVars)
    plot(result)

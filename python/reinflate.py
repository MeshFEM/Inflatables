import MeshFEM, mesh, inflation, parametrization, mesh_utilities, py_newton_optimizer
import utils, os
import boundaries

def _reinflate_impl(output_callback, targetAttractedInflation, uninflatedDefo, fixBoundary=True):
    isheet = targetAttractedInflation.sheet()

    fixedVars = boundaries.getBoundaryVars(isheet) if fixBoundary else []
    isheet.setUninflatedDeformation(uninflatedDefo.T, prepareRigidMotionPinConstraints=False)
    opts = py_newton_optimizer.NewtonOptimizerOptions()
    opts.niter = 5000

    output_callback(-1) # Write out the initial deformation
    return inflation.inflation_newton(targetAttractedInflation, fixedVars, opts, callback=output_callback)

def reinflate(frameOutDir, targetAttractedInflation, uninflatedDefo, fixBoundary=True, iterationsPerOutput = 5):
    os.makedirs(frameOutDir, exist_ok=True)

    frame = 0
    def cb(it):
        nonlocal frame
        if (((it + iterationsPerOutput) % iterationsPerOutput) == (iterationsPerOutput - 1)):
            targetAttractedInflation.sheet().visualizationMesh().save(f'{frameOutDir}/step_{frame}.obj')
            frame = frame + 1

    return _reinflate_impl(cb, targetAttractedInflation, uninflatedDefo, fixBoundary=fixBoundary)

def reinflate_render(frameOutDir, offscreenViewer, targetAttractedInflation, uninflatedDefo, fixBoundary=True, iterationsPerOutput = 5, scalarField=None):
    offscreenViewer.recordStart(frameOutDir)
    frame = 0
    if scalarField is None: scalarField = visualization.ISheetScalarField.NONE
    def cb(it):
        nonlocal frame
        if (((it + iterationsPerOutput) % iterationsPerOutput) == (iterationsPerOutput - 1)):
            offscreenViewer.update(scalarField=scalarField(targetAttractedInflation.sheet()))
            frame = frame + 1

    cr = _reinflate_impl(cb, targetAttractedInflation, uninflatedDefo, fixBoundary=fixBoundary)
    offscreenViewer.recordStop()
    return cr

import registration
def register_frames(outDir, isheet):
    step_0 = mesh.Mesh(f'{outDir}/step_0.obj')
    registrationIndices = step_0.boundaryVertices()
    # top sheet only; we have spurious boundary vertices on the boundary sheet due to deduplication
    registrationIndices = registrationIndices[registrationIndices < isheet.mesh().numVertices()]
    firstBV = step_0.vertices()[registrationIndices]

    import glob
    for path in glob.glob(f'{outDir}/step_*.obj'):
        if (path.find('step_0.obj') >=0): continue
        m = mesh.Mesh(path)
        BV = m.vertices()[registrationIndices]
        R, t = registration.register_points(firstBV, BV)
        m.setVertices(m.vertices() @ R.T + t)
        m.save(path)

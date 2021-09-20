import MeshFEM, mesh, registration, inflation, mesh_operations
import numpy as np
from glob import iglob, glob
import sheet_meshing
import os

# Rigidly transform the geometry of each frame of an inflation sequence (frameDir/step_{}.msh)
# to best match the target surface (represented by liftedSheetPositions).
# We determine the best rigid transformation by only looking at the boundary points.
def registerFrames(frameDir, sheetMesh, iwv, liftedSheetPositions, placeAtopFloor=False, registrationVtxIdxs = None, omitOverlappingTris = True):
    isheet = inflation.InflatableSheet(sheetMesh, iwv)
    if registrationVtxIdxs is None:
        registrationVtxIdxs = sheetMesh.boundaryVertices()

    nc, wallLabels = sheet_meshing.wallMeshComponents(isheet)
    numTopSheetTris = sheetMesh.numTris()
    # Include all triangles from the top sheet
    includeTri = np.ones(2 * numTopSheetTris, dtype=np.bool)
    if (omitOverlappingTris):
        # Omit wall triangles from the bottom sheet
        includeTri[numTopSheetTris:] = wallLabels == -1

    isheet.setUninflatedDeformation(liftedSheetPositions.T, prepareRigidMotionPinConstraints=False)
    tgt_pts = isheet.visualizationMesh().vertices()[registrationVtxIdxs]

    isRegistrationVtx = np.zeros(sheetMesh.numVertices(), dtype=np.bool)
    isRegistrationVtx[registrationVtxIdxs] = True
    BE = sheetMesh.boundaryElements()
    registrationBE = np.array([be for be in BE if isRegistrationVtx[be].all()])

    for meshPath in [f'{frameDir}/lifted.msh'] + glob(f'{frameDir}/step_*.msh'):
        vm = mesh.Mesh(meshPath, embeddingDimension=3)
        V = vm.vertices()

        R, t = registration.register_points(tgt_pts, V[registrationVtxIdxs])
        Vxf = V @ R.T + t
        if (placeAtopFloor):
            Vxf[:, 2] -= Vxf[:, 2].min()

        name = os.path.basename(meshPath)
        #mesh.save(f'{frameDir}/registrationBE_{name}', *mesh_operations.removeDanglingVertices(V, registrationBE))
        outPath=f'{frameDir}/registered_{name}'
        mesh.save(outPath, *mesh_operations.removeDanglingVertices(Vxf, vm.triangles()[includeTri]))

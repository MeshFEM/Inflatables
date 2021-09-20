import MeshFEM, mesh, triangulation
import utils, sheet_meshing, mesh_operations
import field_sampler, wall_generation, filters
import inflation, mesh_utilities
import sheet_optimizer, opt_config
import numpy as np
import copy

def perm_inv(p):
    """
    Compute the inverse of a permutation represented as an index array `p`.
    """
    result = np.empty_like(p)
    result[p] = np.arange(len(p))
    return result

def permuteWallVerticesToMatch(remeshedVF, remeshedIWV, origSheet):
    """
    Permute the wall vertices of the remeshed sheet so that they correspond to
    the original sheet (and thus the design variable definition for both will
    match). While we're at it, we move the wall vertices to the beginning of
    the vertex set, which we expect to slightly speed up variable-fixing in the
    equilibrium solve.

    Parameters
    ----------
    remeshedVF
        (V, F) tuple of the remeshed sheet with incorrect wall vertex ordering.
    remeshedIWV
        Boolean array indicating which entries of V are wall vertices.
    origSheet
        The original sheet whose wall vertex ordering we wish to match.

    Returns
    -------
        Permuted mesh's (V, F) tuple and IWV array
    """
    remeshedWallVertices = remeshedVF[0][remeshedIWV]
    origWallVertices = origSheet.mesh().vertices()[origSheet.wallVertices()]
    mismatch = Exception('Wall vertex mismatch')
    nwv = len(remeshedWallVertices)
    if nwv != len(origWallVertices): raise mismatch

    # Permutations to bring each vertex set into (identical) sorted order.
    rperm = np.lexsort(np.rot90(remeshedWallVertices))
    operm = np.lexsort(np.rot90(origWallVertices))
    # permuting remeshedWallVertices into sorted order followed by inverse of
    # sorting permutation of origWallVertices brings remeshedWallVertices to origWallVertices
    wvPerm = rperm[perm_inv(operm)]

    if not np.all(remeshedWallVertices[wvPerm] == origWallVertices): raise mismatch

    # We reorder all vertices to bring the reordered wall vertices to the top,
    # with all the non-wall vertices appearing below in their original orders
    remeshedWVIndices    = np.flatnonzero( remeshedIWV)
    remeshedWVComplement = np.flatnonzero(~remeshedIWV)
    reorder = np.empty(len(remeshedIWV), dtype=np.int)
    reorder[0:nwv] = remeshedWVIndices[wvPerm]
    reorder[nwv:]  = remeshedWVComplement

    # We want V[reorder][Fnew] == V[F] ==> Fnew = perm_inv(reorder)[F]
    return (remeshedVF[0][reorder], perm_inv(reorder)[remeshedVF[1]]), remeshedIWV[reorder]

def remeshStretchedTubes(isheet, triArea, areaRelThreshold = 1.5):
    """
    After the optimizer has made some large alterations to the sheet design,
    some tubes may be stretched to the point that they are no longer simulated
    accurately. We remesh these tubes with a high-quality triangulation at the
    specified target area.

    Differs from `remeshSheetTubes` below in that only a subset of the tubes
    are remeshed.

    WARNING: currently ignores/removes 1D fusing curves internal to the tube.

    Parameters
    ----------
    isheet
        sheet to remesh
    triArea
        Maximum triangle area for the new mesh
    areaRelThreshold
        Area threshold (relative to triArea) that will trigger a remesh
        of an entire tube.

    Returns
    -------
    Newly remeshed `(mesh, isWallVtx)` pair if remeshing actually occurred,
    otherwise `None`.
    """
    nwmc, wmc = sheet_meshing.wallMeshComponents(isheet, distinctTubeComponents=True) # get indicator fields for the different tubes
    numTubes = -wmc.min()

    areaThreshold = areaRelThreshold * triArea

    # Remesh each tube with triangle areas exceeding `areaThreshold` (without altering the bounding curves).
    m = isheet.mesh()
    V, F = m.vertices(), m.triangles()
    alteredAreas = m.elementVolumes()
    remeshedRegion = np.zeros(m.numTris(), dtype=np.bool)

    getIsBoundaryVtx = lambda m: utils.maskForIndexList(m.boundaryVertices(), m.numVertices())

    newTubeMeshes = []
    for tubeIdx in range(numTubes):
        currTubeTris = (wmc == -(1 + tubeIdx))
        if alteredAreas[currTubeTris].max() < areaThreshold: continue
        remeshedRegion[currTubeTris] = True

        # Retriangulate tube keeping its boundary vertices unchanged
        oldTubeMesh = mesh.Mesh(*mesh_operations.submesh(V, F, currTubeTris))
        bdryV, bdryE = mesh_operations.removeDanglingVertices(oldTubeMesh.vertices(), oldTubeMesh.boundaryElements())
        # Create a quality mesh (but without inserting Steiner points on the boundary)
        # Two "Y"s are needed since the lack of hole points means walls tubes with wall islands look like interior segments
        # to triangle (even though they are actually boundary segments of the tube region).
        remeshedTubes, _, _ = wall_generation.triangulate_channel_walls(bdryV[:, 0:2], bdryE,
                                        holePoints=[], triArea=triArea, omitQualityFlag=False, flags="YY")

        if len(oldTubeMesh.boundaryLoops()) > 1:
            # There are internal holes in the original mesh; we must remove the
            # hole triangles from remeshedTube.
            fs = field_sampler.FieldSampler(oldTubeMesh)
            keep = fs.contains(remeshedTube.barycenters())
            remeshedTube = mesh.Mesh(*mesh_operations.submesh(remeshedTube.vertices(), remeshedTube.triangles(), keep))

        # ... and retriangulate again where necessary to avoid spurious fused triangles in the tubes
        fuseMarkers = getIsBoundaryVtx(remeshedTube)
        fuseSegments = remeshedTube.boundaryElements()
        remeshedTube = mesh_utilities.tubeRemesh(remeshedTube, fuseMarkers, fuseSegments, minRelEdgeLen=0.3)
        newTubeMeshes.append(remeshedTube)

    if len(newTubeMeshes) == 0: return None # No remeshing was needed...
    print(f'Remeshed {len(newTubeMeshes)} tubes')

    # Merge all remeshed tubes into the original mesh, replacing the remeshed regions.
    # The fused markers for the combined mesh are taken as simply the
    # transferred tube markers from the original mesh (we currently assume
    # there are no fused markers in the interior of the tube regions.)
    nonRemeshedV, nonRemeshedF, nonRemeshedFusedMarkers = mesh_operations.submesh(V, F, ~remeshedRegion, vtxData=[isheet.isWallVtx(i) for i in range(m.numVertices())])
    fusedMarkers = [nonRemeshedFusedMarkers] + [getIsBoundaryVtx(tm) for tm in newTubeMeshes]
    merged = mesh_operations.mergedMesh([(nonRemeshedV, nonRemeshedF)] + newTubeMeshes, fusedMarkers)
    #merged = mesh_operations.mergedMesh(newTubeMeshes, fusedMarkers[1:]) # for debugging
    permuted = permuteWallVerticesToMatch(merged[0:2], merged[2], isheet)
    return mesh.Mesh(*permuted[0]), permuted[1]

def remeshSheetTubes(isheet, triArea, minArea = None):
    """
    After the optimizer has made some large alterations to the sheet design,
    some tubes may be stretched to the point that they are no longer simulated
    accurately. We remesh all tubes with a high-quality triangulation
    but prevent over-simplifying shrunk regions by refining down to approximately
    the original area.

    Differs from `remeshStretchedTubes` above in that all tubes are remeshed.

    WARNING: currently ignores/removes 1D fusing curves internal to the tube.

    Parameters
    ----------
    isheet
        sheet to remesh
    triArea
        Maximum triangle area for the new mesh
    minArea
        Lower bound on the area to refine to (defaults to triArea / 10).
        If this equals triArea, no post-retriangulation refinement is run.

    Returns
    -------
    Newly remeshed `(mesh, isWallVtx)` pair.
    """
    if minArea is None: minArea = triArea / 10

    # Remesh each tube with triangle areas exceeding `areaThreshold` (without altering the bounding curves).
    m = isheet.mesh()
    V, F = m.vertices(), m.triangles()

    nwmc, wmc = sheet_meshing.wallMeshComponents(isheet)
    isTube = wmc < 0

    getIsBoundaryVtx = lambda m: utils.maskForIndexList(m.boundaryVertices(), m.numVertices())

    # Retriangulate tube keeping its boundary vertices unchanged
    oldTubeMesh = mesh.Mesh(*mesh_operations.submesh(V, F, isTube))
    bdryV, bdryE = mesh_operations.removeDanglingVertices(oldTubeMesh.vertices(), oldTubeMesh.boundaryElements())
    # Create a quality mesh (but without inserting Steiner points on the boundary)
    # Two "Y"s are needed since the lack of hole points means walls tubes with wall islands look like interior segments
    # to triangle (even though they are actually boundary segments of the tube region).
    remeshedTubes, _, _ = wall_generation.triangulate_channel_walls(bdryV[:, 0:2], bdryE, holePoints=[], triArea=triArea, omitQualityFlag=False, flags="YY")

    # There are almost certainly internal holes in the original tube mesh; we must remove the
    # hole triangles from remeshedTubes.
    fs = field_sampler.FieldSampler(oldTubeMesh)
    keep = fs.contains(remeshedTubes.barycenters())
    remeshedTubes = mesh.Mesh(*mesh_operations.submesh(remeshedTubes.vertices(), remeshedTubes.triangles(), keep))

    if minArea < triArea:
        # Clamp the old tube mesh areas to the desired range and transfer them
        # to the new mesh. For smoother results, we run some umbrella smoothing
        # iterations, transfer to the vertices of the new mesh and average onto
        # the faces.
        smoothedAreas = filters.smooth_per_element_field(oldTubeMesh, oldTubeMesh.elementVolumes(), 30)
        # smoothedAreas gives *target* areas, not maximum areas. We determined empirically (comparing
        # the median target and true sizes) that using it as an upper bound makes `triangle`
        # generate triangles about 1.5x smaller than the target.
        # We therefore scale the sizing field by 1.5.
        smoothedAreas *= 1.5
        vtxTgtAreas = fs.sample(remeshedTubes.vertices(), np.clip(smoothedAreas, minArea, triArea))
        triTgtAreas = vtxTgtAreas[remeshedTubes.triangles()].mean(axis=1)
        #utils.save((vtxTgtAreas, remeshedTubes, oldTubeMesh), 'refinement_guide_field.pkl.gz')

        # Refine mesh to triTgtAreas, again preventing inserting new vertices on the boundary
        refV, refF = triangulation.refineTriangulation(remeshedTubes.vertices()[:, 0:2], remeshedTubes.triangles(), triArea, triTgtAreas, additionalFlags="YY")
        remeshedTubes = mesh.Mesh(refV, refF)
        #utils.save(remeshedTubes, 'remeshedTubes.pkl.gz')

    # ... and retriangulate again where necessary to avoid spurious fused triangles in the tubes
    fuseMarkers = getIsBoundaryVtx(remeshedTubes)
    fuseSegments = remeshedTubes.boundaryElements()
    remeshedTubes = mesh_utilities.tubeRemesh(remeshedTubes, fuseMarkers, fuseSegments, minRelEdgeLen=0.3)

    # Merge the remeshed tubes into the original mesh, replacing the remeshed regions.
    # The fused markers for the combined mesh are taken as simply the
    # transferred tube markers from the original mesh (we currently assume
    # there are no fused markers in the interior of the tube regions.)
    nonRemeshedV, nonRemeshedF, nonRemeshedFusedMarkers = mesh_operations.submesh(V, F, ~isTube, vtxData=[isheet.isWallVtx(i) for i in range(m.numVertices())])
    merged = mesh_operations.mergedMesh([remeshedTubes, (nonRemeshedV, nonRemeshedF)],
                                         [getIsBoundaryVtx(remeshedTubes), nonRemeshedFusedMarkers])
    permuted = permuteWallVerticesToMatch(merged[0:2], merged[2], isheet)
    return mesh.Mesh(*permuted[0]), permuted[1]

def transferDeformation(srcSheet, tgtSheet):
    """
    Transfer the deformation from `srcSheet` to another sheet `tgtSheet` with
    the same design (fusing curves) but a different meshing.
    """
    fs = field_sampler.FieldSampler(srcSheet.mesh())
    tgtV = tgtSheet.mesh().vertices()
    # The vertices of the visualization mesh have the deformed sheet 0 positions followed by the deformed sheet 1 positions
    srcVisMeshV = srcSheet.visualizationMesh().vertices()
    deformedSrcSheets = [fs.sample(tgtV, srcVisMeshV[0:srcSheet.mesh().numVertices() , :]),
                         fs.sample(tgtV, srcVisMeshV[  srcSheet.mesh().numVertices():, :])]

    transferredVars = np.empty(tgtSheet.numVars())
    for vi in range(len(tgtV)):
        var0 = tgtSheet.varIdx(0, vi, 0)
        var1 = tgtSheet.varIdx(1, vi, 0)
        x0 = deformedSrcSheets[0][vi]
        x1 = deformedSrcSheets[1][vi]
        # Check that the transferred deformation respects the fusing
        if (var0 == var1) and (x0 != x1).any():
            raise Exception('Fusing violation')
        transferredVars[var0:var0 + 3] = x0
        transferredVars[var1:var1 + 3] = x1
    tgtSheet.setVars(transferredVars)

def transferProperties(srcSheet, tgtSheet):
    """
    Transfer the basic material/pressure properties from the source sheet to
    the target sheet. Note: not all properties can be transferred (e.g.,
    per-triangle configurations that were manually set in srcSheet)
    """
    tgtSheet.pressure        = srcSheet.pressure
    tgtSheet.youngModulus    = srcSheet.youngModulus
    tgtSheet.thickness       = srcSheet.thickness
    tgtSheet.referenceVolume = srcSheet.referenceVolume

    tgtSheet.setRelaxedStiffnessEpsilon(srcSheet.triEnergyDensities()[0].relaxedStiffnessEpsilon)

def remeshSheet(isheet, triArea, minArea = None):
    """
    Remesh a sheet and transfer the equilibrium deformation/properties to it.
    """
    isheet_remeshed = inflation.InflatableSheet(*remeshSheetTubes(isheet, triArea, minArea=minArea))
    transferDeformation(isheet, isheet_remeshed)
    transferProperties(isheet, isheet_remeshed)
    return isheet_remeshed

def remeshTargetAttractedSheet(tai, triArea, minArea = None):
    """
    Construct a clone of the TargetAttractedInflation object `tai` with a
    remeshed InflatableSheet.
    """
    isheet_remeshed = remeshSheet(tai.sheet(), triArea, minArea=minArea)
    return tai.cloneForRemeshedSheet(isheet_remeshed)

def remeshedOriginalDesign(origTAI, originalDesignMesh, remeshedTAI):
    """
    Remesh the original design mesh in the same way as `remeshedTAI` so that
    the collapse barrier term is approximately preserved when constructing a
    `PySheetOptimizer` with it.
    """
    origDesignWallPos = origTAI.sheet().wallVertexPositionsFromMesh(originalDesignMesh)

    wpi = inflation.WallPositionInterpolator(remeshedTAI.sheet())
    remeshedOrigDesignV2d = wpi.interpolate(origDesignWallPos[:, 0:2])
    return mesh.Mesh(np.pad(remeshedOrigDesignV2d, [(0, 0), (0, 1)]),
                     remeshedTAI.mesh().triangles())

def remeshedSheetOptimizer(sheet_opt, triArea, minArea = None, fixedVars = None):
    """
    Create a new `PySheetOptimizer` by remeshing the design
    being optimized by sheet_opt. An attempt is made to copy all optimization
    settings over to the remeshed optimizer, but this will fail if the
    user has, e.g., manually modified the per-triangle collapse barrier
    settings from their defaults.

    If `fixedVars` is `None` we attempt to translate the fixed variable
    indices used in `sheet_opt`. This only works if (a subset of) the wall
    vertices are used as fixed vertices.
    """
    orig_tai = sheet_opt.rso.targetAttractedInflation()

    remeshed_tai = remeshTargetAttractedSheet(orig_tai, triArea, minArea=minArea)
    remeshed_orig_design = remeshedOriginalDesign(orig_tai, sheet_opt.rso.originalMesh(), remeshed_tai)
    remeshed_isheet = remeshed_tai.sheet()

    # Attempt to transfer the fixedVars. We are guaranteed to have the same number
    # of wall vertices in the remeshed sheet *in the same order* as in the original
    # mesh, so we can easily determine correspondences between fixed variables
    # associated with wall vertices.
    orig_fv = sheet_opt.rso.fixedEquilibriumVars()
    orig_isheet = orig_tai.sheet()
    orig_wv = orig_isheet.wallVertices()
    remeshed_wv = remeshed_isheet.wallVertices()

    remeshed_fv = []
    for var in sheet_opt.rso.fixedEquilibriumVars():
        vtx = orig_isheet.vtxForVar(var)
        if vtx.sheet != 3: raise Exception('Only fixed variables corresponding to fused vertices can be transferred')
        remeshed_fv.append(remeshed_isheet.varIdx(0, remeshed_wv[orig_wv.index(vtx.vi)], var % 3))

    # Infer the collapse barrier settings from the triangles
    iwt_orig = np.array([orig_tai.sheet().isWallTri(ti) for ti in range(orig_tai.mesh().numTris())])
    orig_cb = sheet_opt.rso.collapseBarrier()

    penaltySettings = { 'applyStretchBarrierTube': False } # Only the walls have a stretch barrier applied by default
    def recordUniqueSetting(key, val):
        if key in penaltySettings and penaltySettings[key] != val: raise Exception('Collapse barrier/compression penalty settings could not be inferred')
        penaltySettings[key] = val

    orig_cp = sheet_opt.rso.compressionPenalty

    orig_nt = orig_tai.mesh().numTris()
    for ti in range(orig_nt):
        cpe = orig_cb.collapsePreventionEnergy(ti)
        suffix = 'Wall' if orig_tai.sheet().isWallTri(ti) else 'Tube'
        recordUniqueSetting('detActivationThreshold'   + suffix, cpe.activationThreshold)
        recordUniqueSetting('applyStretchBarrier'      + suffix, cpe.applyStretchBarrier)
        recordUniqueSetting('stretchBarrierActivation' + suffix, cpe.stretchBarrierActivation)
        recordUniqueSetting('stretchBarrierLimit'      + suffix, cpe.stretchBarrierLimit)
        recordUniqueSetting('compressionPenaltyActive' + suffix, orig_cp.includeSheetTri[ti])
        recordUniqueSetting('compressionPenaltyActive' + suffix, orig_cp.includeSheetTri[orig_nt + ti])
    print(penaltySettings)

    wallStretchBarrier = [penaltySettings['stretchBarrierActivationWall'],
                          penaltySettings['stretchBarrierLimitWall']] if penaltySettings['applyStretchBarrierWall'] else None

    fcs = sheet_opt.rso.fusingCurveSmoothness()
    fcs_config = opt_config.FusingCurveSmoothnessParams(fcs.dirichletWeight,
                                                        fcs.laplacianWeight,
                                                        fcs.lengthScaleSmoothingWeight,
                                                        fcs.curvatureWeight)

    def configCallback(remeshed_sheet_opt):
        remeshed_rso = remeshed_sheet_opt.rso
        # Configure the compression penalty
        remeshed_rso.compressionPenaltyWeight = sheet_opt.rso.compressionPenaltyWeight
        remeshed_cp = remeshed_rso.compressionPenalty
        wallActive = penaltySettings['compressionPenaltyActiveWall']
        tubeActive = penaltySettings['compressionPenaltyActiveTube']
        remeshed_cp.includeSheetTri = 2 * [wallActive if remeshed_tai.sheet().isWallTri(ti) else tubeActive for ti in range(remeshed_tai.mesh().numTris())]
        remeshed_cp.Etft_weight = orig_cp.Etft_weight
        remeshed_cp.modulation  = copy.deepcopy(orig_cp.modulation)

        # Configure the target surface fitter
        remeshed_sheet_opt.targetAttractedSheet.targetSurfaceFitter().holdClosestPointsFixed = \
                 sheet_opt.targetAttractedSheet.targetSurfaceFitter().holdClosestPointsFixed

        # Configure the fusing curve smoothness
        remeshed_fcs = remeshed_rso.fusingCurveSmoothness()
        remeshed_fcs.interiorWeight = fcs.interiorWeight
        remeshed_fcs.boundaryWeight = fcs.boundaryWeight

    return sheet_optimizer.PySheetOptimizer(remeshed_tai, remeshed_fv, renderMode=sheet_opt.renderMode,
                                            detActivationThreshold=penaltySettings['detActivationThresholdWall'],
                                            detActivationThresholdTubeTri=penaltySettings['detActivationThresholdTube'],
                                            wallStretchBarrier=wallStretchBarrier,
                                            originalDesignMesh=remeshed_orig_design,
                                            fusingCurveSmoothnessConfig=fcs_config,
                                            checkpointPath=sheet_opt.checkpointPath,
                                            screenshotPath=sheet_opt._screenshotPath,
                                            sheetOutputDir=sheet_opt.sheetOutputDir,
                                            customConfigCallback=configCallback)

################################################################################
# Debugging
################################################################################
def canonicallyOrientedFusingCurve(fc):
    """
    For fusing curves incident the boundary, we flip into a canonical
    orientation to enable comparison. For closed fusing curves, the orientation is guaranteed to
    match, but we must deduplicate the start point and then cyclically permute
    all points so that they start at a canonical position.
    """
    pa = tuple(fc[0])
    pb = tuple(fc[-1])
    if pa < pb: return fc
    if pb > pa: return fc[::-1]

    minPtIdx = min(range(len(fc) - 1), key=lambda i: tuple(fc[i]))
    return np.roll(fc[:-1], -minPtIdx, axis=0)

def matchFusingCurves(curves_a, curves_b):
    """
    Pair up matches in the fusing curve collections `curves_a` and `curves_b`.

    Parameters
    ----------
        curves_a, curves_b lists of fusing curve polylines (list of point arrays).
        For closed polylines, the first and last point should coincide.

    Returns
    -------
        [(ca, cb) for matching ca, cb in (curves_a, curves_b)]
        along with a list of unmatched curves from each collection.
    """
    canonical_curves_a = [canonicallyOrientedFusingCurve(ca) for ca in curves_a]
    canonical_curves_b = [canonicallyOrientedFusingCurve(cb) for cb in curves_b]
    matches = []
    unmatched_a = []
    for ca in canonical_curves_a:
        try:
            j = [np.array_equal(ca, cb) for cb in canonical_curves_b].index(True)
            matches.append((ca, canonical_curves_b.pop(j)))
        except:
            unmatched_a.append(ca)
    return matches, unmatched_a, canonical_curves_b

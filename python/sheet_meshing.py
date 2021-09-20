import scipy
import scipy.sparse.csgraph
import wall_generation, mesh
import mesh_operations
import utils
import triangulation, filters
from mesh_utilities import SurfaceSampler, tubeRemesh
import numpy as np

def meshComponents(m, cutEdges):
    """
    Get the connected components of triangles of a mesh cut along the edges `cutEdges`.

    Parameters
    ----------
    m
        The mesh to split
    cutEdges
        The edges (vertex index pairs) splitting the mesh up into disconnected regions

    Returns
    -------
    ncomponents
        Number of connected components
    components
        The component index for each mesh triangle.
    """
    cutEdgeSet = set([(min(fs), max(fs)) for fs in cutEdges])
    tri_sets = [set(t) for t in m.triangles()]

    # Build dual graph, excluding dual edges that cross the fused segments.
    def includeDualEdge(u, v):
        common_vertices = tri_sets[u] & tri_sets[v]
        return (len(common_vertices) == 2) and ((min(common_vertices), max(common_vertices)) not in cutEdgeSet)

    dual_edges = [(u, v) for u in range(len(tri_sets))
                         for v in m.trisAdjTri(u)
                         if includeDualEdge(u, v)]

    adj = scipy.sparse.coo_matrix((np.ones(len(dual_edges)), np.transpose(dual_edges))).tocsc()

    return scipy.sparse.csgraph.connected_components(adj)

def wallMeshComponents(sheet, distinctTubeComponents = False):
    """
    Get the connected wall components of a sheet's mesh (assigning the tubes "component" -1 by default or components -1, -2, ... if distinctTubeComponents is True).
    """
    m = sheet.mesh()
    nt = m.numTris()
    iwt = np.array([sheet.isWallTri(ti) for ti in range(nt)], dtype=np.bool)
    dual_edges = [(u, v) for u in range(nt)
                         for v in m.trisAdjTri(u)
                         if iwt[u] == iwt[v]]
    adj = scipy.sparse.coo_matrix((np.ones(len(dual_edges)), np.transpose(dual_edges))).tocsc()
    numComponents, components = scipy.sparse.csgraph.connected_components(adj)
    wallLabels = components[iwt].copy()

    renumber = np.empty(numComponents, dtype=np.int)
    renumber[:] = -1 # This assigns all non-wall triangles the "component" -1
    uniqueWallLabels = np.unique(wallLabels)
    numWallComponents = len(uniqueWallLabels)
    renumber[uniqueWallLabels] = np.arange(numWallComponents, dtype=np.int)

    if distinctTubeComponents:
        uniqueTubeLabels = np.unique(components[~iwt])
        renumber[uniqueTubeLabels] = -1 - np.arange(len(uniqueTubeLabels), dtype=np.int)

    components = renumber[components]
    return numWallComponents, components

def remeshWallRegions(m, fuseMarkers, fuseSegments, pointSetLiesInWall, permitWallInteriorVertices = False, pointsLieInHole = None):
    """
    Take an initial mesh of the sheet and determine the connected triangle components
    that correspond to fused regions. Also remesh these regions so as not to have
    interior wall vertices (if requested).

    Parameters
    ----------
    m
        The initial sheet mesh
    fuseMarkers
        Fused vertices of the original sheet mesh
    fuseSegments
        Fused edges of the original sheet mesh
    pointSetLiesInWall
        Function for testing whether a given point set lies within a wall region
    permitWallInteriorVertices
        Whether to permit Triangle to add vertices inside the wall regions.
        (This should usually be `False`, since these vertices permit the walls to crumple.)
    pointsLieInHole
        If provided, this function is used to test whether a given mesh
        component is actually a hole.

    Returns
    -------
    remeshedSheet
        A `MeshFEM` triangle mesh of the top sheet ready for inflation simulation.
    isWallVtx
        Per-vertex boolean array specifying whether each vertex is part of the wall region
        (interior or boundary).
    isWallBdryVtx
        Per-vertex boolean array specifying whether each vertex is part of the wall boundary.
        If `permitWallInteriorVertices` is `False`, then this is the same as `isWallVtx`.
    """
    ncomponents, components = meshComponents(m, fuseSegments)

    ############################################################################
    # Determine which connected components are walls.
    ############################################################################
    triCenters = m.vertices()[m.triangles()].mean(axis=1)

    numWalls = 0
    wallLabels = -np.ones(m.numTris(), dtype=np.int)
    for c in range(ncomponents):
        component_tris = np.flatnonzero(components == c)
        centers = triCenters[component_tris]

        # Discard the entire component if it is actually a hole of the flattened input mesh.
        if (pointsLieInHole is not None):
            if (pointsLieInHole(centers)):
                wallLabels[component_tris] = -2 # only labels -1 (tube) and >= 0 (wall) are kept
        if (pointSetLiesInWall(centers)):
            wallLabels[component_tris] = numWalls
            numWalls = numWalls + 1

    ############################################################################
    # Separately remesh each wall sub-mesh, preserving the non-wall component.
    ############################################################################
    origV = m.vertices()
    origF = m.triangles()
    meshes = [(origV, origF[wallLabels == -1])]
    remeshFlags = 'Y' + ('S0' if not permitWallInteriorVertices else '')
    for wall in range(numWalls):
        # Extract the original wall mesh.
        wallMesh = mesh.Mesh(*mesh_operations.submesh(origV, origF, wallLabels == wall))

        # Perform the initial remeshing of the wall's boundary segments.
        wallMesh, wmFuseMarkers, wmFusedEdges = wall_generation.triangulate_channel_walls(*mesh_operations.removeDanglingVertices(wallMesh.vertices()[:, 0:2], wallMesh.boundaryElements()), triArea=float('inf'), flags=remeshFlags)

        # Note: if the wall mesh encloses holes, Triangle will also have triangulated the holes;
        # we must detect these and remove them from the output.
        # We decompose the remeshed wall into connected components (after
        # cutting away the original wall boundary segments) and keep only the one
        # inside the wall region. Exactly one component should remain after this
        # process.
        nc, wallMeshComponents = meshComponents(wallMesh, wmFusedEdges)
        if (nc != 1):
            wmV = wallMesh.vertices()
            wmF = wallMesh.triangles()
            triCenters = wmV[wmF].mean(axis=1)

            keepTri = np.zeros(wallMesh.numTris(), dtype=np.bool)
            keptComponents = 0
            for c in range(nc):
                component_tris = np.flatnonzero(wallMeshComponents == c)
                if (pointSetLiesInWall(triCenters[component_tris])):
                    keepTri[component_tris] = True
                    keptComponents += 1

            if (keptComponents != 1): raise Exception('Should have kept exactly one component of the remeshed wall')
            # Extract only the kept component.
            wallMesh = mesh.Mesh(*mesh_operations.removeDanglingVertices(wmV, wmF[keepTri]))

        meshes.append(wallMesh)

    mergedV, mergedF = mesh_operations.mergedMesh(meshes)
    remeshedSheet = mesh.Mesh(mergedV, mergedF)

    ############################################################################
    # Determine wall vertices and wall boundary vertices.
    ############################################################################
    wallVertices = set()
    wallBoundaryVertices = set()
    def addVertices(vtxSet, V):
        for v in V: vtxSet.add(tuple(v))

    rsV = remeshedSheet.vertices()
    addVertices(wallBoundaryVertices, rsV[remeshedSheet.boundaryVertices()])

    for wallmesh in meshes[1:]:
        wmV = wallmesh.vertices()
        addVertices(wallBoundaryVertices, wmV[wallmesh.boundaryVertices()])
        addVertices(wallVertices, wmV)
    wallVertices = wallVertices | wallBoundaryVertices

    isWallVtx     = np.array([tuple(v) in wallVertices         for v in rsV])
    isWallBdryVtx = np.array([tuple(v) in wallBoundaryVertices for v in rsV])

    return remeshedSheet, isWallVtx, isWallBdryVtx

# Consider a point set to lie within a hole if over half of its points are within a hole (to a given tolerance)
class HoleTester:
    def __init__(self, meshVertices, meshSampler):
        self.V = meshVertices
        self.sampler = meshSampler
        self.eps = 1e-12 * utils.bbox_dims(meshVertices).max()
    def dist(self, X):
        """
        Compute the distance of each point in "X" to the sampler mesh.
        """
        # This could be more efficient if SurfaceSampler had a method to get a
        # distance to the sampled mesh...
        if (X.shape[1] == 2):
            X = np.pad(X, [(0, 0), (0, 1)])
        closestPts = self.sampler.sample(X, self.V)
        #print(closestPts)
        return np.linalg.norm(X - closestPts, axis=1)
    def pointWithinHole(self, X):
        """
        Check whether each point in X individually lies with a hole.
        """
        return self.dist(X) > self.eps
    def __call__(self, X):
        """
        Check whether a point set generally lies within a hole (i.e. if more
        than half of its points are within a hole).
        """
        return np.count_nonzero(self.pointWithinHole(X)) >= (X.shape[0] / 2)

# Note: if `targetEdgeSpacing` is set too low relative to `triArea`, `triangle`
# will insert new boundary points that fall in the interior of the flattened
# target surface (in strictly convex regions) when refining the triangulation.
#
# If the parametrization is subsequently used to lift the boundary points to
# 3D, these lifted points will not lie on the target surface's boundary. E.g.,
# they may lift off the ground plane even if all boundary vertices of the
# target surface lie on the ground plane.
def generateSheetMesh(sdfVertices, sdfTris, sdf, triArea, permitWallInteriorVertices = False, targetEdgeSpacing = 0.5, minContourLen = 0.75):
    """
    Extract the channel walls described by a signed distance function and use
    them generate a high-quality triangle mesh of the inflatable sheet.

    Parameters
    ----------
    sdfVertices
        Vertices for the wall SDF domain mesh.
    sdfTris
        Triangles for the wall SDF domain mesh.
    sdf
        Per-vertex signed distances to the channel walls
    triArea
        Maximum triangle area (passed to Triangle)
    permitWallInteriorVertices
        Whether to permit Triangle to add vertices inside the wall regions.
        (This should usually be `False`, since these vertices permit the walls to crumple.)
    targetEdgeSpacing
        The approximate resolution at which the extracted contours of the SDF are resampled
        to generate the wall boundary curves.
    minContourLen
        The length threshold below which extracted contours are discarded.

    Returns
    -------
    remeshedSheet
        A `MeshFEM` triangle mesh of the top sheet ready for inflation simulation.
    isWallVtx
        Per-vertex boolean array specifying whether each vertex is part of the wall region
        (interior or boundary).
    isWallBdryVtx
        Per-vertex boolean array specifying whether each vertex is part of the wall boundary.
        If `permitWallInteriorVertices` is `False`, then this is the same as `isWallVtx`.
    """
    pts, edges = wall_generation.extract_contours(sdfVertices, sdfTris, sdf,
                                                  targetEdgeSpacing=targetEdgeSpacing,
                                                  minContourLen=minContourLen)
    m, fuseMarkers, fuseSegments = wall_generation.triangulate_channel_walls(pts[:,0:2], edges, triArea)

    sdfSampler = SurfaceSampler(sdfVertices, sdfTris)
    # Note: the inside/outside test for some sample points of a connected component may disagree due to the limited
    # precision at which we extracted the contours (and the contour resampling), so we take a vote.
    pointsAreInWall = lambda X: np.mean(sdfSampler.sample(X, sdf)) < 0
    pointsLieInHole = HoleTester(sdfVertices, sdfSampler)

    return remeshWallRegions(m, fuseMarkers, fuseSegments, pointsAreInWall, permitWallInteriorVertices, pointsLieInHole=pointsLieInHole)

def generateSheetMeshCustomEdges(sdfVertices, sdfTris, sdf, customPts, customEdges, triArea, permitWallInteriorVertices = False):
    m, fuseMarkers, fuseSegments = wall_generation.triangulate_channel_walls(customPts[:,0:2], customEdges, triArea)

    sdfSampler = SurfaceSampler(sdfVertices, sdfTris)
    # Note: the inside/outside test for some sample points of a connected component may disagree due to the limited
    # precision at which we extracted the contours (and the contour resampling), so we take a vote.
    pointsAreInWall = lambda X: np.mean(sdfSampler.sample(X, sdf)) < 0

    pointsLieInHole = HoleTester(sdfVertices, sdfSampler)

    return remeshWallRegions(m, fuseMarkers, fuseSegments, pointsAreInWall, permitWallInteriorVertices, pointsLieInHole=pointsLieInHole)

def meshWallsAndTubes(fusing_V, fusing_E, m, isWallTri, holePoints, tubePoints, wallPoints, triArea, permitWallInteriorVertices, avoidSpuriousFusedTriangles):
    """
    Create a high quality mesh of the wall and tube regions enclosed by given fusing curves.

    Parameters
    ----------
    fusing_V, fusing_E
        PSLG to be triangulated representing the fusing curves.
    m
        An initial mesh of the sheet region (with any hole triangles removed!) used to obtain the intersection of the wall regions with the sheet boundary.
    isWallTri
        Boolean array holding whether each wall of `m` is a wall triangle
    holePoints, tubePoints, wallPoints
        Lists of points within the hole, tube, and wall regions.
    triArea
        Maximum triangle area for the triangulation
    permitWallInteriorVertices
        Whether wall regions get interior vertices.

    Returns
    -------
    remeshedSheet, isWallVtx, isWallBdryVtx
    """
    ############################################################################
    # 1. Create a quality mesh of the air tubes.
    ############################################################################
    # print(f"fusing_V.shape: {fusing_V.shape}")
    # print(f"fusing_E.shape: {fusing_E.shape}")
    # print(f"wallPoints: {np.array(wallPoints).shape}")
    # print(f"holePoints: {np.array(holePoints).shape}")
    mTubes, fuseMarkers, fuseSegments = wall_generation.triangulate_channel_walls(fusing_V[:, 0:2], fusing_E, holePoints=wallPoints + holePoints, triArea=triArea, omitQualityFlag=False, flags="j") # jettison vertices that got eaten by holes...
    #utils.save((mTubes, fuseMarkers, fuseSegments), 'tubes_and_markers.pkl.gz')
    if avoidSpuriousFusedTriangles:
        try:
            mTubes = tubeRemesh(mTubes, fuseMarkers, fuseSegments, minRelEdgeLen=0.3) # retriangulate where necessary to avoid spurious fused triangles in the tubes
        except:
            utils.save((mTubes, fuseMarkers, fuseSegments), utils.freshPath('tubeRemeshFailure', suffix='.pkl.gz'))
            raise

    fuseMarkers += [0 for i in range(mTubes.numVertices() - len(fuseMarkers))]
    #mTubes.save('remeshedTubes.msh')

    # For meshes without finite-thickness wall regions, we are done
    # (and all vertices in fusing_V are wall/wall boundary vertices.)
    if not np.any(isWallTri):
        isWallVtx = np.array(fuseMarkers, dtype=np.bool)
        return mTubes, isWallVtx, isWallVtx

    ############################################################################
    # 2. Triangulate the wall meshes without inserting any Steiner points.
    ############################################################################
    # We need to triangulate the new collection of boundary segments, which consists of the
    # boundary segments from the new tube mesh along with the original mesh boundary segments
    # that border wall regions.
    boundarySegments = [(mTubes.vertices(), mTubes.boundaryElements())]
    #print(boundarySegments)
    # mesh.save("tube_bdry.obj", *boundarySegments[0])
    wallBoundaryElements = m.boundaryElements()[isWallTri[m.elementsAdjacentBoundary()]]
    if len(wallBoundaryElements) > 0:
        boundarySegments.append((m.vertices(), wallBoundaryElements))
    # mesh.save("sheet_bdry_intersect_walls.obj", *boundarySegments[1])
    newPts, newEdges = mesh_operations.mergedMesh(boundarySegments)
    # mesh.save("new_contour.obj", newPts, newEdges)

    wallmeshFlags = 'Y' + ('S0' if not permitWallInteriorVertices else '')
    mWall, _, _ = wall_generation.triangulate_channel_walls(newPts[:,0:2], newEdges, holePoints=tubePoints + holePoints, triArea=triArea if permitWallInteriorVertices else float('inf'), omitQualityFlag=False, flags="j" + wallmeshFlags) # jettison vertices that got eaten by holes...

    # mWall.save("walls.obj")

    ############################################################################
    # 3. Merge the tube and wall meshes
    ############################################################################
    mFinal = mesh.Mesh(*mesh_operations.mergedMesh([mTubes, mWall]), embeddingDimension=3)
    # mFinal.save("final.obj")

    ############################################################################
    # 4. Determine wall vertices and wall boundary vertices.
    ############################################################################
    wallVertices = set()
    wallBoundaryVertices = set()
    def addVertices(vtxSet, V):
        for v in V: vtxSet.add(tuple(v))

    finalV = mFinal.vertices()
    addVertices(wallBoundaryVertices, finalV[mFinal.boundaryVertices()])

    wmV = mWall.vertices()
    addVertices(wallBoundaryVertices, wmV[mWall.boundaryVertices()])
    addVertices(wallVertices, wmV)

    # Also include fused vertices marked inside the tube mesh (i.e., those
    # fused by zero-width curves)
    addVertices(wallBoundaryVertices, mTubes.vertices()[np.array(fuseMarkers, dtype=np.bool)])

    wallVertices = wallVertices | wallBoundaryVertices

    isWallVtx     = np.array([tuple(v) in wallVertices         for v in finalV])
    isWallBdryVtx = np.array([tuple(v) in wallBoundaryVertices for v in finalV])

    return mFinal, isWallVtx, isWallBdryVtx

def newMeshingAlgorithm(sdfVertices, sdfTris, sdf, customPts, customEdges, triArea, permitWallInteriorVertices = False, avoidSpuriousFusedTriangles = True):
    ############################################################################
    # 1. Perform an initial, low quality triangulation used only to segment the
    #    design domain into tube and wall regions.
    ############################################################################
    m, fuseMarkers, fuseSegments = wall_generation.triangulate_channel_walls(customPts[:, 0:2], customEdges, triArea=float('inf'), omitQualityFlag=True, flags="YY")
    # m.save('initial_triangulation.msh')

    ############################################################################
    # 2. Determine the wall components/hole points.
    ############################################################################
    triCenters = m.vertices()[m.triangles()].mean(axis=1)

    sdfSampler = SurfaceSampler(sdfVertices, sdfTris)

    numWalls = 0
    wallPoints = []
    tubePoints = []
    holePoints = []

    ncomponents, components = meshComponents(m, fuseSegments)

    # First detect and remove holes
    pointsLieInHole = HoleTester(sdfVertices, sdfSampler)
    for c in range(ncomponents):
        component_tris = components == c
        centers = triCenters[component_tris]
        p = centers[0, 0:2] # Todo: pick center of largest area triangle?

        if (pointsLieInHole(centers)):
            components[component_tris] = -1 #mark for deletion
            holePoints.append(p)
            continue

    if len(holePoints) > 0:
        print(f'Detected {len(holePoints)} holes')
        # Note: there shouldn't be any dangling vertices since no new vertices
        # are inserted inside the holes.
        m = mesh.Mesh(m.vertices(), m.elements()[components >= 0])
        ncomponents, components = meshComponents(m, fuseSegments)
        triCenters = m.vertices()[m.triangles()].mean(axis=1)
    # m.save('without_holes.msh')

    wallLabels = -np.ones(m.numTris(), dtype=np.int) # assign -1 to tubes

    # Next, pick a point within each air tube
    for c in range(ncomponents):
        component_tris = components == c
        centers = triCenters[component_tris]
        p = centers[0, 0:2] # Todo: pick center of largest area triangle?

        # Note: the inside/outside test for some sample points of a connected component may disagree due to the limited
        # precision at which we extracted the contours (and the contour resampling), so we take a vote.
        triCentersAreInWall = np.mean(sdfSampler.sample(centers, sdf)) < 0

        if (triCentersAreInWall):
            wallPoints.append(p)
            wallLabels[component_tris] = numWalls
            numWalls = numWalls + 1
        else:
            tubePoints.append(p)
    return meshWallsAndTubes(customPts, customEdges, m, wallLabels >= 0, holePoints, tubePoints, wallPoints, triArea, permitWallInteriorVertices, avoidSpuriousFusedTriangles)

def generateSheetMeshNewAlgorithm(sdfVertices, sdfTris, sdf, triArea, permitWallInteriorVertices = False, targetEdgeSpacing = 0.5, minContourLen = 0.75, avoidSpuriousFusedTriangles=True):
    pts, edges = wall_generation.extract_contours(sdfVertices, sdfTris, sdf,
                                                  targetEdgeSpacing=targetEdgeSpacing,
                                                  minContourLen=minContourLen)
    return newMeshingAlgorithm(sdfVertices, sdfTris, sdf, pts, edges, triArea, permitWallInteriorVertices, avoidSpuriousFusedTriangles)

def remeshSheet(isheet, triArea, permitWallInteriorVertices = False, omitWallsContainingPoints=[]):
    """
    Remesh an inflatable sheet design with a high quality triangulation
    (leaving the fusing curves unchanged).
    We can omit certain walls by passing a nonempty point set for the `omitWallsContainingPoints` argument.

    Returns
    -------
    remeshedSheet, isWallVtx, isWallBdryVtx
    """
    nwmc, wmc = wallMeshComponents(isheet, distinctTubeComponents=True)
    im = isheet.mesh()
    imV = im.vertices()
    imF = im.triangles()

    # Convert walls specified by `omitWallsContainingPoints` into tube regions.
    ssampler = SurfaceSampler(imV, imF)
    omittedWallComponents = []
    if (len(omitWallsContainingPoints) > 0):
        tris, _ = ssampler.closestTriAndBaryCoords(np.array(omitWallsContainingPoints))
        omittedWallComponents = np.unique(wmc[tris])
        if (np.any(omittedWallComponents < 0)): raise Exception("omitWallsContainingPoints contains non-wall points.")
        wmc[tris] = -1 # Reassign omitted walls to the first tube region.

    # Generate tube and wall points (one in each tube/wall component)
    tubePoints = []
    wallPoints = []
    triCenters = imV[:, 0:2][imF].mean(axis=1)
    #print(np.unique(wmc))
    for c in range(np.min(wmc), nwmc):
        #print(f'Component: {c}')
        if c in omittedWallComponents: continue
        p = triCenters[np.where(wmc == c)[0][0]]
        if (c < 0): tubePoints.append(p)
        else      : wallPoints.append(p)

    # Generate hole points inside each internal boundary loop; this
    # requires a low-quality triangulation of the boundary loops.
    sheetBoundary = mesh_operations.removeDanglingVertices(imV, im.boundaryElements())
    mHoleDetect, mFuseMarkers, mFuseSegments = wall_generation.triangulate_channel_walls(sheetBoundary[0][:, 0:2], sheetBoundary[1], triArea=float('inf'), omitQualityFlag=True, flags="YY")
    nholeDetectComponents, holeDetectComponents = meshComponents(mHoleDetect, mFuseSegments)

    holeTest = HoleTester(imV, ssampler)
    holePoints = np.array([triCenters[np.where(holeDetectComponents == c)[0][0]] for c in range(nholeDetectComponents)])
    holePoints = list(holePoints[holeTest.pointWithinHole(holePoints)])

    # Get all design curves for remeshing. These consist of the union of two disjoint sets of curves:
    #   - Boundaries of the wall regions.
    #   - The intersection of the tube and sheet boundaries.
    wm = mesh.Mesh(*mesh_operations.mergedMesh([(imV, imF[wmc == i]) for i in range(nwmc)])) # Mesh of walls only.
    sheetBoundaryIntersectTubes = mesh_operations.removeDanglingVertices(imV, im.boundaryElements()[wmc[im.elementsAdjacentBoundary()] < 0])

    fusing_V, fusing_E = mesh_operations.mergedMesh([(wm.vertices(), wm.boundaryElements()), sheetBoundaryIntersectTubes])

    isWallTri = (wmc >= 0)
    return meshWallsAndTubes(fusing_V, fusing_E, im, isWallTri, holePoints, tubePoints, wallPoints, triArea, permitWallInteriorVertices, avoidSpuriousFusedTriangles=True)

import triangulation, field_sampler
def forward_design_mesh(V, E, fusedPts, holePts, triArea):
    """
    Create an inflatable sheet mesh from a collection of curves and points indicating
    whether the closed curve containing them should be considered a wall or a hole
    (instead of a tube/pillow).
    """
    sdfV, sdfF, pointMarkers, edgeMarkers = triangulation.triangulate(V[:, 0:2], E, holePts=holePts, triArea=1e8, omitQualityFlag=True, outputPointMarkers=True, outputEdgeMarkers=True)   
    minit = mesh.Mesh(sdfV[:, 0:2], sdfF)

    # Create a SDF field indicating the wall regions (the data needed by newMeshingAlgorithm)
    nc, c = meshComponents(minit, edgeMarkers)
    sdf = c
    fs = field_sampler.FieldSampler(minit)
    if len(fusedPts) > 0:
        fusedComponents = np.array(np.unique(fs.sample(fusedPts, c)), dtype=np.int)
        sdf[c == fusedComponents] = -1

    return newMeshingAlgorithm(sdfV, sdfF, sdf, V, E, triArea=triArea)

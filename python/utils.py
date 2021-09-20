import numpy as np
import MeshFEM, mesh
import registration
import os
import pickle, gzip

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

def sheetTrisForVar(sheet, varIdx):
    """
    Get indices of triangles influencing a particular equilibrium variable of the sheet.
    (indices < sheet.mesh().numTrix() refer to triangles in the top sheet, the
    rest to triangles in the bottom sheet.)
    """
    v = sheet.vtxForVar(varIdx)
    result = []
    m = sheet.mesh()
    if v.sheet & 1: result.extend(np.where(m.triangles() == v.vi)[0])
    if v.sheet & 2: result.extend(np.where(m.triangles() == v.vi)[0] + m.numTris())
    return result

def maskForIndexList(indices, size):
    mask = np.zeros(size, dtype=np.bool)
    mask[indices] = True
    return mask

def freshPath(path, suffix='', excludeSuffix = False):
    if path is None: return
    if not os.path.exists(path + suffix): return path if excludeSuffix else path + suffix
    i = 0
    candidatePath = lambda i: f'{path}.{i}{suffix}'
    while os.path.exists(candidatePath(i)): i += 1
    print(f'Requested path exists; using fresh path {candidatePath(i)}')
    return f'{path}.{i}' if excludeSuffix else candidatePath(i)

def allEnergies(obj):
    return {name: obj.energy(etype) for name, etype in obj.EnergyType.__members__.items()}

def allGradientNorms(obj, freeVariables = None):
    if freeVariables is None:
        freeVariables = np.arange(obj.numVars(), dtype=np.int)
    return {name: np.linalg.norm(obj.gradient(etype)[freeVariables]) for name, etype in obj.EnergyType.__members__.items()}

def loadObj(path):
    V, F = [], []
    for l in open(path, 'r'):
        comps = l.strip().split(' ')
        specifier = comps[0].lower()
        if (specifier == 'v'): V.append([float(c) for c in comps[1:]])
        if (specifier == 'l' or specifier == 'f'): F.append([int(i) - 1 for i in comps[1:]])
    return np.array(V), np.array(F)

def normalizedParamEnergies(obj):
    ET = obj.EnergyType
    return [obj.energy(et) / reg if reg != 0 else obj.energy(et)
            for (et, reg) in [(ET.Fitting,               1.0),
                              (ET.AlphaRegularization,   obj.alphaRegW),
                              (ET.PhiRegularization,     obj.phiRegW),
                              (ET.BendingRegularization, obj.bendRegW)]]

def bbox(P):
    return np.min(P, axis=0), np.max(P, axis=0)

def bbox_dims(P):
    bb = bbox(P)
    return bb[1] - bb[0]

def getClosestPointDistances(P):
    """
    Gets the distance of each point in a point collection P to its closest other point in P.
    """
    closestDist = []
    for p in P:
        closestDist.append(np.partition(np.linalg.norm(p - P, axis=1), 1)[1])
    return closestDist

def prototypeScaleNormalization(P, placeAtopFloor = False, objectScale = 750, reorient = False):
    if reorient: P = registration.align_points_with_axes(P)
    bb = bbox(P)
    c = (bb[0] + bb[1]) / 2 # use center of bounding box rather than center of mass
    t = -c
    if (placeAtopFloor): t[2] = -bb[0][2]
    return (P + t) * (objectScale / np.max(bb[1] - bb[0]))

def renderingNormalization(P, placeAtopFloor = False):
    """
    Return the transformation function that maps the points `P` in a standard
    configuration for rendering.
    """
    c = np.mean(P, axis=0)
    bb = bbox(P)
    t = -c
    if placeAtopFloor:
        t[2] = -bb[0][2]
    s = 1.0 / np.max(bb[1] - bb[0])
    return lambda x: s * (x + t)

def isWallTri(sheet_mesh, is_wall_vtx):
    """
    Determine which triangles are part of a wall (triangles made of three wall vertices).
    """
    return is_wall_vtx[sheet_mesh.triangles()].all(axis=1)

def pad2DTo3D(P):
    if P.shape[1] == 3: return P
    return np.pad(P, [(0, 0), (0, 1)], mode='constant')

import itertools
def nth_choice(n, *args):
    return next(itertools.islice(itertools.product(*args), n, None))

def writeFields(path, m, name1, field1, *args):
    mfw = mesh.MSHFieldWriter(path, m.vertices(), m.triangles())
    data = [name1, field1] + list(args)
    for name, field in zip(data[0::2], data[1::2]):
        mfw.addField(name, field)
    del mfw

import mesh_utilities
def getLiftedSheetPositions(origSheetMesh, uv, target_surf):
    paramSampler = mesh_utilities.SurfaceSampler(pad2DTo3D(uv), target_surf.triangles())
    return paramSampler.sample(origSheetMesh.vertices(), target_surf.vertices())

import parametrization
def getSquashedLiftedPositionsFromLiftedPos(optSheetMesh, liftedPos, liftFrac = 0.2, freeBoundary = False):
    flatPos = None
    if freeBoundary:
        # Note: we assume the design sheet has already been registered with the target boundary...
        flatPos = optSheetMesh.vertices()
    else:
        # If we're fixing the boundary, the flattened state must perfectly match the target surface's boundary.
        # Do this by mapping the design sheet to the interior of the target surface's boundary harmonically.
        bv = optSheetMesh.boundaryVertices()
        flatPos = parametrization.harmonic(optSheetMesh, liftedPos[bv])
    return flatPos + liftFrac * (liftedPos - flatPos)

def getSquashedLiftedPositions(optSheetMesh, origSheetMesh, uv, target_surf, liftFrac = 0.2):
    liftedPos = getLiftedSheetPositions(origSheetMesh, uv, target_surf)
    return getSquashedLiftedPositionsFromLiftedPos(optSheetMesh, liftedPos, liftFrac)

import mesh, glob
def getBoundingBox(framesDir):
    minCorner = [ np.inf,  np.inf,  np.inf]
    maxCorner = [-np.inf, -np.inf, -np.inf]
    for i in glob.glob(f'{framesDir}/step_*.msh'):
        V = mesh.Mesh(i, embeddingDimension=3).vertices()
        minCorner = np.min([minCorner, V.min(axis=0)], axis=0)
        maxCorner = np.max([maxCorner, V.max(axis=0)], axis=0)
    return np.array([minCorner, maxCorner])

def printBoundingBox(framesDir):
    print('{', ', '.join(map(str, getBoundingBox(framesDir).ravel(order='F'))), '}')

def getTargetSurf(tas):
    tsf = tas.targetSurfaceFitter()
    return mesh.Mesh(tsf.targetSurfaceV, tsf.targetSurfaceF)

################################################################################
# Strain analysis
################################################################################
def getStrains(isheet):
    getStrain = lambda ted: ted.principalBiotStrains() if hasattr(ted, 'principalBiotStrains') else (np.sqrt(ted.eigSensitivities().Lambda()) - 1)
    return np.array([getStrain(ted) for ted in isheet.triEnergyDensities()])

def tensionStates(isheet):
    return [ted.tensionState() for ted in isheet.triEnergyDensities()]

# Get the amount by which each element is compressed. This is
# zero for elements in complete tension or the increase in
# strain needed to put the element in tension.
def compressionMagnitudes(isheet):
    def cm(ted):
        l = ted.eigSensitivities().Lambda()
        if (l[0] < 1): return 1 - np.sqrt(l[0]) # full compression case
        return np.max([np.sqrt(1 / np.sqrt(l[0])) - np.sqrt(l[1]), 0]) # partial compression or full tension case.
    return np.array([cm(ted) for ted in isheet.triEnergyDensities()])

# Get the amount by which each element is "fully compressed" (nonzero
# only for elements in full compression rather than partial tension).
def fullCompressionMagnitudes(isheet):
    return np.clip(1.0 - np.sqrt(np.array([ted.eigSensitivities().Lambda()[0] for ted in isheet.triEnergyDensities()])), 0.0, None)

def writeStrainFields(path, isheet):
    vm = isheet.visualizationMesh()
    strains = getStrains(isheet)
    mfw = mesh.MSHFieldWriter(path, vm.vertices(), vm.elements())
    mfw.addField("tensionState",         tensionStates(isheet))
    mfw.addField("compressionMagnitude", compressionMagnitudes(isheet))
    mfw.addField("lambda_0",             strains[:, 0])
    mfw.addField("lambda_1",             strains[:, 1])

def strainHistogram(isheet):
    from matplotlib import pyplot as plt
    strains = getStrains(isheet)
    plt.hist(strains[:, 0], bins=500, range=(-0.4,0.1), label='$\lambda_0$');
    plt.hist(strains[:, 1], bins=500, range=(-0.4,0.1), label='$\lambda_1$');
    plt.legend()
    plt.grid()
    plt.title('Principal strains');

def cumulativeArcLen(loopPts):
    numPts, numComp = loopPts.shape
    arcLen = np.empty(numPts)
    arcLen[0] = 0.0
    for i in range(1, numPts):
        arcLen[i] = arcLen[i - 1] + np.linalg.norm(loopPts[i] - loopPts[i - 1])
    return arcLen

################################################################################
# Curve operations
################################################################################
def samplePointsOnLoop(loopPts, numSamples, offset):
    """
    Sample `numSamples` evenly spaced along the arlength of a closed polyline "loopPts"
    This closed loop is represented by a list of points, with the first and
    last point coinciding.
    The first sample point is placed at `offset`, a relative arclength position along the  curve in [0, 1].
    If `offset` is a list of `n` floats (instead of just a float), then we generate n * numSamples points
    at the specified offsets (with the sampled points for each offset value interleaved).
    """
    assert(np.linalg.norm(loopPts[-1] - loopPts[0]) == 0)
    numPts, numComp = loopPts.shape
    arcLen = cumulativeArcLen(loopPts)
    arcLen /= arcLen[-1] # normalize arc lengths to [0, 1]

    # Arc length position of the sample points
    if (not isinstance(offset, list)):
        offset = [offset]
    s = np.vstack([np.fmod(np.linspace(0, 1, numSamples, endpoint=False) + o, 1.0) for o in offset]).ravel(order='F')

    samples = np.empty((len(s), numComp))
    for c in range(numComp):
        samples[:, c] = np.interp(s, arcLen, loopPts[:, c])
    return samples

import shapely
import shapely.ops
import shapely.geometry as shp

def normalOffset(polygon, dist):
    """
    Offset points on the planar curve or shp.Polygon "polygon" in the normal
    direction by "dist". This curve should lie in a "z = const" plane or the
    result will be distorted.

    Returns a **list** of the resulting polygon(s) (shp.Polygon instances),
    as an inward offset can divide the input polygon into multiple pieces.
    """
    if not isinstance(polygon, shp.Polygon):
        polygon = shp.Polygon(polygon[:, 0:2])
    offsetResult = polygon.buffer(dist)
    # Note: the result could be a Polygon or a MultiPolygon...
    if (isinstance(offsetResult, shp.Polygon)):
        return [offsetResult]
    elif (isinstance(offsetResult, shp.MultiPolygon)):
        return list(offsetResult)
    else: raise Exception('Unexpected polygon offset result type')

def getBoundary(polygon, getAll = False):
    """
    Get the boundary of a shapely polygon.
    If `getAll` is true, we return a list with all boundary polylines sorted by descending length;
    if false, we return the largest one and print a warning.
    """
    result = polygon.boundary
    if result.geom_type == 'LineString':
        if getAll: return [np.array(result)]
        return np.array(result)
    if result.geom_type == 'MultiLineString':
        allBoundaries = sorted([np.array(r) for r in result], key=lambda a: -len(a))
        if getAll: return allBoundaries
        print('WARNING: union boundary has multiple components; returning the largest one')
        return allBoundaries[0]
    raise Exception('Unexpected boundary result type')

def unionPolygons(polygons):
    """
    Union two or more polygons [ptsA, ptsB, ...] described by point lists `ptsA` and `ptsB`.
    (For each of these lists, the first and last points must agree)
    """
    return shapely.ops.unary_union([shp.Polygon(p) for p in polygons])

import os
def get_nonexistant_path(fname_path):
    """
    Get the path to a filename which does not exist by incrementing path.
    From https://stackoverflow.com/a/43167607/122710
    """
    if not os.path.exists(fname_path):
        return fname_path
    filename, file_extension = os.path.splitext(fname_path)
    i = 1
    new_fname = "{}-{}{}".format(filename, i, file_extension)
    while os.path.exists(new_fname):
        i += 1
        new_fname = "{}-{}{}".format(filename, i, file_extension)
    return new_fname

import scipy
import scipy.sparse
def reconnectPolygons(polygons, originatingPolygon, minGap = 0):
    """
    Add the line segments of the minimal length necessary to connect the entries of
    polygon list `polygons`, only allowing line segments that lie within the
    originating polygon (using a minimum spanning tree).
    This is meant to address the problem where eroding a polygon can separate it
    into a bunch of small polygons that we want to connect at the seam width.

    Unfortunately, we can have two polygons whose ground-truth connection line
    (indicated by * below) exceeds the distance of their nearest points (a and b)
                a---  *  --------+
                                 |
                b----------------+
    (here the "--" lines represent thin polygons). This will result in a reconnection
    failure. It could be mitigated by splitting up large polygons with some threshold,
    but we instead opt for the reconnectPolygons2 algorithm below.
    """
    #pickle.dump(polygons, open(get_nonexistant_path('polygons.pkl'), 'wb'))
    #pickle.dump(originatingPolygon, open(get_nonexistant_path('originatingPolygon.pkl'), 'wb'))
    inputPolygons = polygons
    polygons = [shp.Polygon(p) for p in polygons]
    originatingPolygon = shp.Polygon(originatingPolygon)
    n = len(polygons)
    dists = np.full((n, n), np.inf)
    closestPoints = np.empty((n, n), dtype='O')
    for i, pi in enumerate(polygons):
        for j, pj in enumerate(polygons):
            if (i >= j): continue; # only compute upper triangle
            cp = np.vstack([np.array(o.coords) for o in shapely.ops.nearest_points(pi, pj)])
            connectionDist = np.linalg.norm(np.subtract(*cp))
            distToOrig = shp.Point(cp.mean(axis=0)).distance(originatingPolygon)
            if (distToOrig > 0.25 * connectionDist): continue # If the candidate connecting line strays too far outside the originating polygon, it is probably invalid
            dists        [i, j] = connectionDist
            closestPoints[i, j] = cp

    outputPolylines = inputPolygons.copy()
    for mst_edge in zip(*scipy.sparse.csgraph.minimum_spanning_tree(dists).nonzero()):
        i, j = sorted(mst_edge)
        if (dists[i, j] < minGap): continue # no connection needed
        outputPolylines.append(closestPoints[i, j])
    return outputPolylines

import scipy.spatial
def reconnectPolygons2(inputPolygons, originatingPolygon, fuseWidth, includeExtensions=False):
    """
    Hopefully superior algorithm for inserting line segments to reconnect the
    distinct polygons that arose from an erosion operation on originatingPolygon.
    This one works by detecting "bridges"--regions of `originatingPolygon \ inputPolygons`
    that connect two distinct polygons of inputPolygons--and then joining the
    closest points of these input polygons (after intersecting with a
    neighborhood of the bridge).
    """
    eps = 1e-6
    polygons = [shp.Polygon(p).buffer(fuseWidth / 2 + eps) for p in inputPolygons]
    originatingPolygon = shp.Polygon(originatingPolygon)
    bridges = [p for p in originatingPolygon.difference(shapely.ops.unary_union(polygons)) if p.boundary.length > 3 * fuseWidth]
    outputPolylines = inputPolygons.copy()
    for b in bridges:
        distances = np.array([b.distance(p) for p in polygons])
        # If "b" actually bridges between two polygons, connect these
        # polygons' closest points (restricted to a neighborhood of the bridge)
        closest = np.argsort(distances)
        if (distances[closest[1]] < fuseWidth / 2):
            bridgeRegion = b.buffer(2 * fuseWidth)
            p0, p1 = shapely.ops.nearest_points(bridgeRegion.intersection(polygons[closest[0]]),
                                                bridgeRegion.intersection(polygons[closest[1]]))
            outputPolylines.append(np.array([np.asarray(p0), np.asarray(p1)]))
        elif includeExtensions:
            if (b.boundary.length > 4 * fuseWidth):
                bdryPts = np.array(b.boundary)
                _, p0 = shapely.ops.nearest_points(b, polygons[closest[0]])
                b_to_p0 = scipy.spatial.distance.cdist([np.asarray(p0)], bdryPts[:, 0:2])[0]
                farthest = np.argmax(b_to_p0)
                if (b_to_p0[farthest] > 4 * fuseWidth):
                    p1, _ = shapely.ops.nearest_points(polygons[closest[0]], shp.Point(bdryPts[farthest, 0:2]))
                    outputPolylines.append(np.array([np.asarray(p1), bdryPts[farthest, 0:2]]))
    return outputPolylines

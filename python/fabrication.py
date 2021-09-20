import os
import MeshFEM, mesh, inflation, sheet_meshing, mesh_operations, visualization, registration
from mesh_operations import VertexMerger
import parametrization, mesh_utilities
import numpy as np
import utils
import filters
import shapely
import shapely.geometry as shp

def mm_to_inch(mm):
    return np.array(mm) / 25.4

def appendOverlap(polyline, overlap):
    """
    Extend the "endpoint" of closed polygon `polyline` to wrap around the
    polygon by distance `overlap`.
    """
    p = polyline
    if overlap <= 0: return p
    if np.linalg.norm(p[-1] - p[0]) > 1e-10: return p # not a closed polyline
    edgeLens = np.linalg.norm(np.diff(p, axis=0), axis=1)
    additionalPoints = []
    i = 0
    numEdges = len(p) - 1
    while overlap > 0:
        if (edgeLens[i] <= overlap):
            # full edge
            overlap -= edgeLens[i]
            additionalPoints.append(p[i + 1]) # add edge endpoint
        else:
            # partial edge
            alpha = overlap / edgeLens[i]
            additionalPoints.append((1 - alpha) * p[i] + alpha * p[i + 1])
            break
        i = (i + 1) % numEdges
    return np.vstack([p, np.array(additionalPoints)])

# Remove axes and rescale so that the output file matches the design perfectly
def remove_axes_and_rescale():
    from visualization import plt as plt
    plt.axis('off')
    plt.margins(0, 0)
    w = -np.subtract(*plt.gca().get_xbound())
    h = -np.subtract(*plt.gca().get_ybound())
    plt.subplots_adjust(0, 0, 1, 1, wspace=0.0, hspace=0.0)
    plt.gcf().set_size_inches(*mm_to_inch([w, h]))

def getAllPoints2D(polylines):
    """
    Get all the points (rows) appearing in a list of {np array | list of np arrays}
    """
    flattened = []
    for p in polylines:
        if isinstance(p, list):
            for pp in p: flattened.append(pp[:, 0:2])
        else: flattened.append(p[:, 0:2])
    return np.vstack(flattened)

def mapAllPolylines(polylinesSoup, f):
    result = []
    for p in polylinesSoup:
        if isinstance(p, list):
            result.append([f(pp) for pp in p])
        else:
            result.append(f(p))
    return result

def transformAllPoints(polylinesSoup, xf):
    return mapAllPolylines(polylinesSoup, lambda p: xf(p[:, 0:2]))

def orientOptimally(polylines, aspectRatio):
    """
    Brute-force search for optimal orientation of polylines `polylines` to fit within work area of aspect ratio `aspectRatio`.
    This is the orientation for which the design can be scaled up the most without exceeding the work area.
    """
    print('Searching for optimal orientation')
    #return polylines
    P = getAllPoints2D(polylines)[:,0:2]
    rot2d = lambda theta: np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    def scale(theta):
        R = rot2d(theta)
        Pxf = P @ R.T
        bb = utils.bbox(Pxf)
        w, h = bb[1] - bb[0]
        return min(aspectRatio / w, 1 / h)
    thetas = np.linspace(0, np.pi / 2, 4096)
    #thetas = np.linspace(0, np.pi / 2, 12)
    R_opt = rot2d(thetas[np.argmax([scale(theta) for theta in thetas])]) # maximize the admissible scale factor
    Pxf = P @ R_opt.T
    bb = utils.bbox(Pxf)
    t = -np.mean(bb, axis=0)
    print('Optimal bbox size: ', -np.subtract(*bb))
    return transformAllPoints(polylines, lambda p: p[:, 0:2] @ R_opt.T + t)

def write_polylines(outPath, polylines, physicalLineWidth=1.0, highlightMultipolygons=False, overlap=0, aspectRatio=1374/1325):
    print('appending overlap: ', overlap)
    visualization.plot_polylines(orientOptimally(mapAllPolylines(polylines, lambda p: appendOverlap(p, overlap)), aspectRatio), physicalLineWidth=physicalLineWidth, highlightMultipolygons=highlightMultipolygons)
    remove_axes_and_rescale()
    if outPath is not None:
        visualization.plt.savefig(outPath, transparent=True)
        visualization.plt.close()

def listwrap(maybeList): return (maybeList if isinstance(maybeList, list) else [maybeList])
def flatten(inlist):
    """
    Flatten a list containing lists (and potentially non-list items)
    into a list of items. Only applies one level of flattening...
    [[a, b], c, [[d, e]]] ==> [a, b, c, [d, e]]
    """
    return [item for sublist in inlist for item in listwrap(sublist)]

# Apply the following conversions to objects in a list:
#   - polyline (a np.array)    ==> polyline
#   - shapely polygon          ==> list of polylines
#   - list of polylines        ==> list of polylines
#   - list of shapely polygons ==> list of polylines (flattened)
def shapely_to_numpy_polylines(shapely_polygons):
    def deshapelyed(p):
        if isinstance(p,        list): return flatten(shapely_to_numpy_polylines(p)) # Flatten the result to properly handle the list-of-shapely-polygons case
        if isinstance(p, shp.Polygon): return [polyline for polyline in ([np.array(p.exterior)] + [np.array(h) for h in p.interiors])] # pure list of polylines
        if isinstance(p,  np.ndarray): return p
        raise Exception('Unexpected type')
    return [deshapelyed(l) for l in shapely_polygons]

def write_design_file(outPath, polylines_and_samples, scale, channelMargin, tabMargin,
                      tabWidth=3, tabHeight=4, inletScale=2, tabBlacklist=[],
                      basePlate=False, tabRounding=0.2, fuseSeamWidth=None,
                      reconnect=True, numBoundaryLoops=1, overlap=0,
                      smartOuterChannelForISheet=None):
    """
    Generate and write the polylines describing the final fusing/cutting geometry for
    a given design.

    Parameters
    ----------
    outPath
        Output SVG file path, or None to display in notebook.
    polylines_and_samples
        An array holding:
            polylines_and_samples[0:-2-numBoundaryLoops]: the inner fusing polygons
            polylines_and_samples[-2-numBoundaryLoops:-2]: the original sheet boundaries; only the last is used as an inlet/outer air channel.
            polylines_and_samples[-2]:   the inlet location edge
            polylines_and_samples[-1]:   the subsampled tab locations
        Note: entries of this array can either be plain polylines (numpy arrays) or instances of shp.Polygon.
    scale
        Factor by which to scale all geometry prior to export
    channelMargin
        The gap to leave around the wall fusing curves so that all channels are connected
    tabMargin
        The gap to leave around the boundary air channel for tab fusing material
    tabWidth
        The width of the tabs relative to tabMargin
    tabHeight
        The height of the tabs relative to tabMargin
    inletScale
        Factor by which to scale the inlet geometry (relative to channelMargin)
        If zero or None, the inlet is omitted.
    tabBlacklist
        List of tabs to skip (clockwise counting starting at inlet)
    basePlate
        Whether we are generating the base plate instead of a sheet pattern. In this case we use
        thinner, disconnected tab slits instead of tabs.
    tabRounding
        The amount by which to round corners of the tab geometry (should probably be in [0, 1])
    fuseSeamWidth
        The width of the fusing seam used for fabrication; if specified, the
        walls will be offset inward by half this value.
    numBoundaryLoops
    overlap
        How much to extend the end of each polyline to wrap around the polygon to counter fabrication errors (ultimately not needed)
    smartOuterChannelForISheet
        Instead of adding an outer air channel of constant width that will fight contraction, make a "smart" version that adds smaller air bypass channels
        only where needed. This needs access to the InflatableSheet instance.
    """
    if (fuseSeamWidth is not None):
        degeneratedCount, reconnectedCount = 0, 0
        fusingCurves = [utils.normalOffset(c, -0.5 * fuseSeamWidth / scale) for c in polylines_and_samples[0:-2-numBoundaryLoops]]
        for i, f in enumerate(fusingCurves):
            components = len(f)
            if components > 1:
                if (reconnect):
                    f[:] = utils.reconnectPolygons2(f, polylines_and_samples[i], fuseSeamWidth / scale, True)
                    if (len(f) > components):
                        reconnectedCount += 1
                degeneratedCount += 1
        if degeneratedCount > 0:
            print(f"WARNING: {degeneratedCount} fusing polygons degenerated; requested seam exceeds fusing region width");
        if reconnectedCount > 0:
            print(f"WARNING: {reconnectedCount} fusing polygons required reconnecting; requested seam exceeds fusing region width");
    else:
        fusingCurves = polylines_and_samples[0:-2-numBoundaryLoops]

    sheetRegion = shp.Polygon(polylines_and_samples[-3])
    # Interpret all additional boundary loops (if they exist) as holes
    for pi in range(1, numBoundaryLoops):
        sheetRegion = sheetRegion - shp.Polygon(polylines_and_samples[-3 - pi])
    #utils.save(polylines_and_samples, 'polylines_and_samples.pk.gz')
    #utils.save(sheetRegion, 'sheetRegion.pkl.gz')

    # Transform inlet/tab geometry to originate at the center of edge "e" and
    # orient the +x axis along e (e.g., so that inlet points along the normal)
    def mapToEdge(geometry, e, normalOffset):
        t = np.diff(e.T)[0:2, 0]
        t = t / np.linalg.norm(t)
        n = np.array([-t[1], t[0]])
        R = np.stack((t, n), axis=1)
        return geometry @ R.T + e.mean(axis=0)[0:2] + normalOffset * n

    if smartOuterChannelForISheet is not None:
        V = smartOuterChannelForISheet.mesh().vertices()
        polylines = smartOuterChannelForISheet.fusedRegionBooleanIntersectSheetBoundary()
        boundaryEdges = shp.MultiLineString([V[p] for p in polylines])
        #utils.save(boundaryEdges.buffer(channelMargin), 'test.pkl.gz')
        bypasses = boundaryEdges.buffer(channelMargin)
        if bypasses.geom_type == 'Polygon': bypasses = [bypasses] # we generally expect a multipolygon...
        outerAirChannelPolygons = [shapely.ops.unary_union([sheetRegion] + list(bypasses))]
    else:
        outerAirChannelPolygons = utils.normalOffset(sheetRegion, channelMargin)
    if (inletScale is not None) and (inletScale > 0):
        # Inlet geometry at the origin, oriented along the y axis
        # We extend it pretty far inside the sheet (y=-2) to prevent its union
        # with the "smartOuterChannel" from containing holes.
        inletGeometry = inletScale * channelMargin * np.array([[-0.5, -2], [0.5, -2], [0.5, 3], [1.5,4],[-1.5,4], [-0.5, 3], [-0.5, -2]])

        inletEdge = polylines_and_samples[-2]
        #utils.save(outerAirChannelPolygons, 'outerAirChannelPolygons.pkl.gz')
        sheetWithOuterChannel = utils.unionPolygons([*outerAirChannelPolygons,
                                                    mapToEdge(inletGeometry, inletEdge, channelMargin * 0.5)])
    else:
        sheetWithOuterChannel = utils.unionPolygons(outerAirChannelPolygons)
    fusingCurves.extend(utils.getBoundary(sheetWithOuterChannel, getAll=True))

    tabPoints = polylines_and_samples[-1]
    tabInwardExtension = 0 # We need to extend the tabs inward significantly to avoid creating holes when unioning with "smartOuterChannel"
    if (basePlate):
        tabHeight *= 0.05 # just make slits instead of tabs
        tabWidth *= 1.05
    else:
        # We will start the tabs at the channel boundary to avoid forming sharp
        # corners/gaps when we union with the tab margin.
        # We therefore need to add 1 tabMargin unit to the height to ensure
        # the effective tab height is approximately tabHeight.
        tabHeight += 1
        # Contract so that we obtain the target margin width after expanding...
        tabMargin *= 1 - tabRounding
        # The margin contraction will shrink the tabWidth/tabHeight proportionally, but
        # we instead want to contract by a constant distance. We need to readjust
        # tabWidth/tabHeight to achieve this constant inset.
        tabWidth  = (tabWidth  - 2 * tabRounding) / (1 - tabRounding)
        tabHeight = (tabHeight -     tabRounding) / (1 - tabRounding)
        tabInwardExtension = tabHeight
    tabGeometry = tabMargin * np.array([[-tabWidth / 2, -tabInwardExtension], [ tabWidth / 2, -tabInwardExtension],
                                        [ tabWidth / 2,           tabHeight], [-tabWidth / 2,           tabHeight],
                                        [-tabWidth / 2, -tabInwardExtension]])
    # Offset the outer border air channel boundary(s) outward to produce the outer cut boundary.
    sheetWithOuterChannelAndCutLine = sheetWithOuterChannel.buffer(tabMargin)
    tabPolygons = [mapToEdge(tabGeometry, np.array(tabEdge), channelMargin + tabMargin * (0.0 if not basePlate else 1.1))
                        for i, tabEdge in enumerate(zip(tabPoints[::2], tabPoints[1::2])) if i not in tabBlacklist]
    # For the sheet pattern: union the tabs with the design outline to generate the tab cut paths.
    # For the basePlate, we generate a bunch of disconnected slits in place of the tabs.
    if (not basePlate):
        try:
            outerBoundary = utils.unionPolygons([sheetWithOuterChannelAndCutLine] + tabPolygons)
            if (tabRounding > 0):
                outerBoundary = outerBoundary.buffer(tabRounding * (tabMargin / (1 - tabRounding)))
            laserCutCurves = utils.getBoundary(outerBoundary, getAll=True)
        except AttributeError as e:
            print("Exception while unioning tab geometry; is a tab inside the inlet region or floating outside a concavity? Check output and try changing tabOffset.")
    else:
        laserCutCurves = utils.getBoundary(sheetWithOuterChannelAndCutLine, getAll=True) + tabPolygons

    outputCurves = shapely_to_numpy_polylines(fusingCurves + laserCutCurves)
    scaledOutputCurves = [[scale * P for P in c] if isinstance(c, list) else scale * c for c in outputCurves]
    print(f'Writing {outPath}')
    bb = utils.bbox(np.vstack([P[:, 0:2] for P in flatten(scaledOutputCurves)]))
    print(f'bbox: ', bb, ' dimensions: ', bb[1] - bb[0])
    write_polylines(outPath, scaledOutputCurves, physicalLineWidth=fuseSeamWidth, overlap=overlap)

def writeFabricationData(outPath, origDesign, optDesign, iwv, targetSurf, uv,
                         scale=1.0, channelMargin=0.5, tabMargin=0.5,
                         inletOffset=0.0, tabOffset=None, numTabs=50,
                         tabWidth=3, tabHeight=4, inletScale=2, tabBlacklist = [], tabRounding=0.2,
                         fuseSeamWidth=None, reconnect=True, inletBoundaryIdx=0,
                         overlap=0, smartOuterChannel=None, flipped=True):
    """
    Write all the necessary fabrication data for a model

    Parameters
    ----------
    outPath
        Output directory or None to display in notebook
    origDesign, optDesign
        Original and optimized design meshes
    iwv
        Wall vertex markers
    targetSurf
        Target surface
    uv
        Parametrization of the target surface used to generate origDesign
    scale
        Factor by which to scale all output curves prior to export
    channelMargin
        The gap to leave around the wall fusing curves so that all channels are connected
    tabMargin
        The gap to leave around the boundary air channel for tab fusing material
    inletOffset, tabOffset
        Clockwise normalized arclength offset (in [0, 1]) for the inlet and the first tab.
        The inlet offset is relative to the sheet's first boundary vertex, and the
        tab offset is relative to the inlet position. By default the first and
        last tabs are centered around the inlet.
    numTabs
        Number of tabs to generate on the boundary loop inletBoundaryIdx;
        a proportional number of tabs will be placed on the other boundary loops.
    tabWidth, tabHeight
        The size of the tabs relative to tabMargin
    inletScale
        Factor by which to scale the inlet geometry (relative to channelMargin)
    tabBlacklist
        List of tabs to skip (clockwise counting starting at inlet)
    tabRounding
        The amount by which to round corners of the tab geometry (should probably be in [0, 1])
    fuseSeamWidth
        The width of the fusing seam used for fabrication; the walls will be offset inward by half this value.
    inletBoundaryIdx
        Which sheet boundary is used to form the outer air channel/inlet
    overlap
        Distance by which to extend the end of each closed polyline to overlap its beginning
        (so that the fusing curve  around the polygon slightly more than 1x).
    smartOuterChannel
        Instead of adding an outer air channel of constant width that will fight contraction, make a "smart" version that adds smaller air bypass channels
        only where needed.
    flipped
        whether the `uv` was computed in a way that the boundary loops flipped during flattening;
        in this case we need to undo this flip when generating the so that the boundary loops have the proper
        orientation for normal offsetting.
    """
    if outPath is not None: os.makedirs(outPath, exist_ok=True)
    if tabOffset is None:
        tabOffset = 0.0
        if (numTabs > 0):
            tabOffset = 0.5 / numTabs # Center tabs around the inlet by default.

    for name, designMesh in zip(['opt', 'orig'], [optDesign, origDesign]):
        isheet = inflation.InflatableSheet(designMesh, iwv)
        m = isheet.mesh()

        V, F = m.vertices(), m.elements()
        idxPolygons = filters.extract_component_polygons(m, -np.array(isheet.airChannelIndices()))
        bdryLoops = [shp.Polygon(V[i.exterior], holes=[V[h] for h in i.holes]) for i in idxPolygons]

        sheetBoundaries = mesh_operations.boundaryLoops(m)

        polysWithHoles = [i for i, p in enumerate(idxPolygons) if len(p.holes) > 0]
        if len(polysWithHoles) > 0:
            write_polylines(f'{outPath}/{name}.holes.svg', [scale * p for p in (flatten(shapely_to_numpy_polylines([bdryLoops[i] for i in polysWithHoles])) + sheetBoundaries)])

        # Move the "inlet boundary" to the end, where the rest of the code expects it...
        sheetBoundaries = sheetBoundaries[0:inletBoundaryIdx] + sheetBoundaries[inletBoundaryIdx + 1:] + [sheetBoundaries[inletBoundaryIdx]]
        bdryLoops.extend(sheetBoundaries)
        sheetBdry = bdryLoops[-1] # the user-specified boundary that we use to place the tabs/outer air channel/inlet.
        tangent_fd_eps = 1e-4
        bdryLoops.append(utils.samplePointsOnLoop(sheetBdry, 1, [inletOffset - tangent_fd_eps, inletOffset + tangent_fd_eps])) # generate "inlet edge"

        # Sample locations for "tab edges" on all sheet boundaries
        o = inletOffset + tabOffset
        tabDensity = numTabs / utils.cumulativeArcLen(sheetBoundaries[-1])[-1]
        bdryLoops.append(np.array([p for sb in sheetBoundaries
                                            for p in utils.samplePointsOnLoop(sb, round(tabDensity * utils.cumulativeArcLen(sb)[-1]), [o - tangent_fd_eps, o + tangent_fd_eps])]))

        wdf = lambda path, bloops, basePlate, fuseSeamWidth, isMappedBoundary: \
            write_design_file(path, bloops, scale, channelMargin, tabMargin,
                    tabWidth, tabHeight, inletScale, tabBlacklist=tabBlacklist,
                    basePlate=basePlate, tabRounding=tabRounding,
                    fuseSeamWidth=fuseSeamWidth, reconnect=reconnect,
                    numBoundaryLoops=len(sheetBoundaries), overlap=overlap,
                    smartOuterChannelForISheet=isheet if (smartOuterChannel and not isMappedBoundary) else None)
        wdf(f'{outPath}/{name}.wall_boundaries.svg' if outPath is not None else None, bdryLoops, basePlate=False, fuseSeamWidth=fuseSeamWidth, isMappedBoundary=False)

        # We have only one "wall matching map"; the one constructed for the original sheet would
        # be identical since the optimization simply moves the original vertices around (while preserving connectivity)
        if name == 'opt':
            # For matching the fabricated sheet boundaries to the target boundaries:
            # Create a map from the uv domain (i.e., the original design domain) to the interior of the target
            # boundary (found by flattening the target with a harmonic map).
            sampler = mesh_utilities.SurfaceSampler(np.pad(uv, [(0, 0), (0, 1)], 'constant'), targetSurf.triangles())
            squashedTarget = parametrization.harmonic(targetSurf, targetSurf.vertices()[targetSurf.boundaryVertices()])
            targetSquashedPosForSheetVtx = sampler.sample(origDesign.vertices(), squashedTarget)

            # Create flattened image for matching boundaries to target boundaries by
            # applying the map from design sheet => orig design sheet => squashed target mesh
            designSampler = mesh_utilities.SurfaceSampler(m.vertices(), designMesh.triangles())

            bdryLoops = shapely_to_numpy_polylines(bdryLoops)
            if flipped:
                mapPolyline = lambda p: designSampler.sample(p, targetSquashedPosForSheetVtx)[:, [1, 0, 2]]
            else:
                mapPolyline = lambda p: designSampler.sample(p, targetSquashedPosForSheetVtx)
            def mapLoopEntry(l):
                if (isinstance(l, list)): return [mapPolyline(p) for p in l]
                return mapPolyline(l)
            mappedBoundaryLoops = [mapLoopEntry(l) for l in bdryLoops]
            zHeight = np.abs(np.subtract(*utils.bbox(np.vstack(flatten(mappedBoundaryLoops))))[2])
            if zHeight > 1e-10:
                print("WARNING: target boundary does not lie on z=0; projected down to xy plane anyway...")
                print(f"(Height along z: {zHeight}")
            wdf(f'{outPath}/mapped_wall_boundaries.svg' if outPath is not None else None, mappedBoundaryLoops, basePlate=True, fuseSeamWidth=None, isMappedBoundary=True)

def getDeformedBoundary(sheet):
    bdryPts = VertexMerger()
    bdryElements = sheet.mesh().boundaryElements()
    mergedBdryElements = [[bdryPts.add(eqVars[varOffset:varOffset + 3])
                                for varOffset in [sheet.varIdx(0, v, 0) for v in be]]
                            for be in bdryElements]
    return bdryPts.vertices(), mergedBdryElements

import matplotlib
from matplotlib import pyplot as plt, tri as mtri
from matplotlib import collections as mc
from enum import Enum
import numpy as np

def hueFromHalfAngle(x):
    val = np.fmod(x / np.pi, 1.0)
    val += 1.0 * (val < 0)
    return val

def uvTriangulation(param):
    uv = param.uv()
    return mtri.Triangulation(uv[:, 0], uv[:, 1], param.mesh().triangles())

def visualizeStretchOrientations(param, quiver=True, width=14, height=14):
    uvTriang = uvTriangulation(param)
    plt.figure(figsize=(width, height))
    plt.axis('equal')
    if (quiver):
        uv = param.uv()
        triCenters = np.array([(1 / 3.0) * (uv[t[0], :] + uv[t[1], :] + uv[t[2], :])
                      for t in param.mesh().triangles()])
        stretchOrientation = param.leftStretchAngles()
        plt.quiver(triCenters[:, 0], triCenters[:, 1], np.cos(stretchOrientation), np.sin(stretchOrientation), pivot='mid', headaxislength=0)
        # plt.scatter(triCenters[:, 0], triCenters[:, 1])
    else:
        plt.tripcolor(uvTriang, hueFromHalfAngle(param.leftStretchAngles()), cmap=matplotlib.cm.hsv, vmin=0, vmax=1)
    #plt.triplot(uvTriang, 'k-', lw=0.25)
    plt.tight_layout()
    # plt.savefig('test.png')
    plt.show()

class QuiverVisualization(Enum):
    NONE = 0
    PER_TRI = 1
    PER_VTX = 2

def visualizeChannelOrientation(param, quiver=QuiverVisualization.PER_TRI, orientationHue=True, width=14, height=14):
    uvTriang = uvTriangulation(param)
    plt.figure(figsize=(width, height))
    plt.axis('equal')
    uv = param.uv()
    if (quiver == QuiverVisualization.PER_TRI):
        leftStretchAngle = param.leftStretchAngles()
        triCenters = np.array([(1 / 3.0) * (uv[t[0], :] + uv[t[1], :] + uv[t[2], :])
                      for t in param.mesh().triangles()])
        quiverPts = triCenters
        shading='flat'
    if (quiver == QuiverVisualization.PER_VTX):
        leftStretchAngle = param.perVertexLeftStretchAngles()
        quiverPts = uv
        shading='gouraud'
    channelOrientation = leftStretchAngle + np.pi / 2
    if orientationHue: plt.tripcolor(uvTriang, hueFromHalfAngle(channelOrientation), cmap=matplotlib.cm.hsv, vmin=0, vmax=1, shading=shading)
    plt.quiver(quiverPts[:, 0], quiverPts[:, 1],
               np.cos(channelOrientation), np.sin(channelOrientation),
               pivot='mid', scale=100.0,
               headaxislength=0, headlength=0, headwidth=0)
    #plt.triplot(uvTriang, 'k-', lw=0.25)
    plt.tight_layout()
    # plt.savefig('test.png')

def visualizeChannelOrientationSubsampled(param, orientationHue=True, scaleFactorColoring=False, width=14, height=14, numSamples = 1000, quiverScale=50.0):
    plt.figure(figsize=(width, height))
    plt.axis('equal')
    channelOrientation = param.leftStretchAngles() + np.pi / 2
    if orientationHue:
        uvTriang = uvTriangulation(param)
        plt.tripcolor(uvTriang, hueFromHalfAngle(channelOrientation), cmap=matplotlib.cm.hsv, vmin=0, vmax=1)
    if scaleFactorColoring:
        uvTriang = uvTriangulation(param)
        plt.tripcolor(uvTriang, param.getAlphas(), vmin=1, vmax=np.pi / 2, cmap=matplotlib.cm.viridis)
    uv = param.uv()
    uvPadded = np.pad(uv, [(0, 0), (0, 1)], mode='constant')

    F = param.mesh().triangles()
    # import igl
    # sampleUV = igl.uniformly_sample_two_manifold_internal(uvPadded, F, numSamples, 0)
    import point_cloud_utils as pcu
    sampleTris, _ = pcu.sample_mesh_poisson_disk(uvPadded, F, numSamples, use_geodesic_distance=False)

    import mesh_utilities

    F = F[sampleTris]
    channelOrientation = channelOrientation[sampleTris]
    triCenters = uv[F].mean(axis=1)
    plt.quiver(triCenters[:, 0], triCenters[:, 1],
               np.cos(channelOrientation), np.sin(channelOrientation),
               pivot='mid', scale=quiverScale,
               headaxislength=0, headlength=0, headwidth=0)
    plt.tight_layout()

def visualizeScaleFactors(param, width=14, height=14, colorbar=True):
    plt.figure(figsize=(width, height))
    plt.title('Stretch Factor')
    plt.axis('equal')
    uvt = uvTriangulation(param)
    #plt.triplot(uvt, 'k-', lw=0.25)
    plt.tripcolor(uvt, param.getAlphas(), vmin=1.0, vmax = np.pi / 2, cmap=matplotlib.cm.viridis)
    plt.tight_layout()
    if colorbar: plt.colorbar()

def quiverAnglePlot(mesh2D, vertexAngles, width=14, height=14, scale=50.0):
    plt.figure(figsize=(width, height))
    plt.axis('equal')

    V = mesh2D.vertices();
    plt.quiver(V[:, 0], V[:, 1],
               np.cos(vertexAngles), np.sin(vertexAngles),
               pivot='mid', scale=scale,
               headaxislength=0, headlength=0, headwidth=0)

    plt.tight_layout()

    # for i, tri in enumerate(mesh2D.triangles()):
    #     plt.text(np.mean(V[tri, 0]),
    #              np.mean(V[tri, 1]),
    #              "{}".format(i), fontsize=8)

    # plt.savefig('quiver.png')

    plt.show()

def stretchFactorPlot(param, width=16, height=7, vmin=1.0, vmax=np.pi / 2):
    plt.figure(figsize=(width, height))
    uvt = uvTriangulation(param)
    plt.title('Stretch Factor')
    plt.axis('equal')
    plt.triplot(uvt, 'k-', lw=0.25)
    plt.tripcolor(uvt, param.getAlphas(), vmin=vmin, vmax=vmax, cmap=matplotlib.cm.viridis)
    plt.tight_layout()
    plt.colorbar()

def customScalarFieldPlot(param, fieldValues, width=16, height=7):
    plt.figure(figsize=(width, height))
    uvt = uvTriangulation(param)
    plt.title('Stretch Factor')
    plt.axis('equal')
    plt.triplot(uvt, 'k-', lw=0.25)
    plt.tripcolor(uvt, fieldValues, vmin=1.0, vmax = np.pi / 2, cmap=matplotlib.cm.viridis)
    plt.tight_layout()
    plt.colorbar()

def visualize(param, width=16, height=7):
    plt.figure(figsize=(width, height))
    uvt = uvTriangulation(param)

    plt.subplot(1, 2, 1)
    plt.title('Stretch Orientation')
    plt.axis('equal')
    plt.triplot(uvt, 'k-', lw=0.25)
    plt.tripcolor(uvt, hueFromHalfAngle(param.leftStretchAngles()), cmap=matplotlib.cm.hsv, vmin=0, vmax=1)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title('Stretch Factor')
    plt.axis('equal')
    plt.triplot(uvt, 'k-', lw=0.25)
    plt.tripcolor(uvt, param.getAlphas(), vmin=1.0, vmax = np.pi / 2, cmap=matplotlib.cm.viridis)
    plt.tight_layout()
    plt.colorbar()

def visualize_vtx(param, width=16, height=7):
    plt.figure(figsize=(width, height))
    uvt = uvTriangulation(param)

    plt.subplot(1, 2, 1)
    plt.title('Stretch Orientation')
    plt.axis('equal')
    plt.triplot(uvt, 'k-', lw=0.25)
    plt.tripcolor(uvt, hueFromHalfAngle(param.perVertexLeftStretchAngles()), cmap=matplotlib.cm.hsv, vmin=0, vmax=1, shading='gouraud')
    plt.colorbar()

    # # Check if we get the same results by explicitly doing the angle ==> rgb conversion before interpolation (we do)
    # collection = matplotlib.collections.TriMesh(uvt)
    # collection.set_facecolor(matplotlib.cm.hsv(hueFromHalfAngle(param.perVertexLeftStretchAngles())))
    # plt.gca().add_collection(collection)
    # plt.gca().autoscale_view()

    plt.subplot(1, 2, 2)
    plt.title('Stretch Factor')
    plt.axis('equal')
    plt.triplot(uvt, 'k-', lw=0.25)
    plt.tripcolor(uvt, param.perVertexAlphas(), vmin=1.0, vmax = np.pi / 2, cmap=matplotlib.cm.viridis)
    plt.tight_layout()
    plt.colorbar()

def plot_vertices(param, pointList, width=7, height=7):
    plt.figure(figsize=(width, height))
    uvt = uvTriangulation(param)

    uv = param.uv()

    plt.axis('equal')
    plt.triplot(uvt, 'k-', lw=0.25)
    plt.scatter(uv[pointList, 0], uv[pointList, 1])

    plt.tight_layout()
    plt.show()

class ZProjectionParam:
    def __init__(self, m):
        self.my_uv = m.vertices()[:, 0:2]
        self.my_mesh = m
    def uv(self):   return self.my_uv
    def mesh(self): return self.my_mesh

def plot_2d_mesh(m, width=7, height=7, pointList=[], segmentList=[], additionalScatterPts = np.zeros((0, 1)), triEdgeWidth=0.25, bbox=None, markerSize=5**2):
    param = ZProjectionParam(m)

    plt.figure(figsize=(width, height))
    uvt = uvTriangulation(param)

    if (len(pointList) > 0):
        uv = param.uv()
        plt.scatter(uv[pointList, 0], uv[pointList, 1], s=markerSize, marker='.') # default s: 6**2
    if (additionalScatterPts.shape[0] > 0):
        plt.scatter(additionalScatterPts[:, 0], additionalScatterPts[:, 1])

    plt.axis('equal')
    plt.triplot(uvt, 'k-', lw=triEdgeWidth)

    if (len(segmentList) > 0):
        ax = plt.gca()
        ax.add_collection(mc.LineCollection(param.uv()[segmentList], linewidths=2))

    if (bbox is not None):
        ax = plt.gca()
        ax.set_xlim(*bbox[0])
        ax.set_ylim(*bbox[1])

    plt.tight_layout()
    #plt.savefig('2d_mesh.png')

# Hack to specify line width in physical units;
# from https://stackoverflow.com/a/42972469/122710
from matplotlib.lines import Line2D
class LineDataUnits(Line2D):
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop("linewidth", 1)
        if (_lw_data is None): raise Exception('_lw_data is None')
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72./self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data))-trans((0, 0)))*ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)

def getMatplotlibLineColors():
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = np.array([matplotlib.colors.to_rgb(c) for c in prop_cycle.by_key()['color']])
    colors = np.pad(colors, [(0, 0), (0, 1)], mode='constant', constant_values=1)
    return colors

def plot_polylines(polylines, width=7, height=7, bbox=None, physicalLineWidth=None, highlightMultipolygons=False, colorOverride=None, ax = None):
    """
    Plot a list of polylines, each represented as a list of consecutive points
    in an X by 2 matrix.
    We also support the case where individual entries of `polylines` are actually a list of
    distinct polylines (called "subpoylines" below) that are intended to be
    given the same color.
    """
    if ax is None:
        plt.figure(figsize=(width, height))
        ax = plt.gca()
        ax.set_aspect('equal')
    else:
        plt.sca(ax)

    colors = getMatplotlibLineColors()
    numColors = len(colors)
    for i, p in enumerate(polylines):
        subpolylines = p if isinstance(p, list) else [p]
        c  = colors[i % numColors]
        if len(subpolylines) > 1 and highlightMultipolygons:
            c = "black"
        if colorOverride is not None:
            c = colorOverride
        for pp in subpolylines:
            if len(pp) < 2: continue
            if (physicalLineWidth is None): l = Line2D       (pp[:, 0], pp[:, 1], color=c)
            else:                           l = LineDataUnits(pp[:, 0], pp[:, 1], color=c, linewidth=physicalLineWidth)
            ax.add_line(l)
        plt.autoscale()

    if (bbox is not None):
        ax = plt.gca()
        ax.set_xlim(*bbox[0])
        ax.set_ylim(*bbox[1])
    plt.tight_layout()

# Break into contiguous paths (point lists)
def break_into_polylines(pts, edges):
    paths = []
    prevPtIdx = edges[0][0]
    currPath = [pts[edges[0][0]]]
    for e in edges:
        if (e[0] == prevPtIdx):
            currPath.append(pts[e[1]])
        else:
            paths.append(np.array(currPath))
            currPath = [pts[e[0]], pts[e[1]]]
        prevPtIdx = e[1]
    paths.append(np.array(currPath))
    return paths

def plot_line_segments(pts, edges, width=7, height=7, bbox=None):
    plot_polylines(break_into_polylines(pts, edges),
                   width=width, height=height, bbox=None)

    # lc = mc.LineCollection([[pts[e[0]], pts[e[1]]] for e in edges], linewidths=2)
    # ax = plt.gca()
    # ax.add_collection(lc)
    # plt.axis('equal')
    # ax.autoscale()
    # ax.margins(0.1)
    # plt.show()

def channelAndQuiverPlot(channelPts, channelEdges, mesh2D, vertexQuiverAngles, width=14, height=14, scale=50.0, quiverAlpha=0.5):
    plot_line_segments(channelPts, channelEdges, width, height)

    V = mesh2D.vertices();
    plt.quiver(V[:, 0], V[:, 1],
               np.cos(vertexQuiverAngles), np.sin(vertexQuiverAngles),
               pivot='mid', scale=scale, alpha=quiverAlpha,
               headaxislength=0, headlength=0, headwidth=0)


def singularValueHistogram(param, width=7, height=7, bins=500):
    plt.figure(figsize=(width, height))
    plt.rc('text', usetex=False)
    plt.hist(np.concatenate((param.getMinSingularValues(),
                             param.getAlphas())), bins=bins)
    plt.title('Distribution of Singular Values σ0, σ1')

def scalarFieldPlotFast(vertices, tris, field, nlevels = 16, width=14, height=14, customMin = None, customMax = None, showEdges = False, cmap = matplotlib.cm.viridis):
    tri = mtri.Triangulation(vertices[:, 0], vertices[:, 1], tris)
    plt.figure(figsize=(width, height))
    plt.axis('equal')
    if (customMin is None): customMin = np.min(field)
    if (customMax is None): customMax = np.max(field)
    if showEdges: plt.triplot(tri, 'k-', lw=0.25)
    plt.tricontourf(tri, field, levels=np.linspace(customMin, customMax, nlevels), cmap=cmap)
    plt.tight_layout()
    # plt.savefig("lilium_stripe_field.png")

def plotPerTriScalarField(V, F, field, width=22, height=14, cmap = matplotlib.cm.jet, plotDomain = None):
    tri = mtri.Triangulation(V[:, 0], V[:, 1], F)
    plt.figure(figsize=(22, 14))
    plt.axis('equal')
    plt.tripcolor(tri, field, cmap=matplotlib.cm.jet)
    if plotDomain is not None:
        plt.xlim(plotDomain[:, 0])
        plt.ylim(plotDomain[:, 1])
    plt.tight_layout()

def display_svg(path):
    from IPython.display import HTML
    svg = open(path).read()
    svg = svg[svg.find('<svg'):] # trim off the xml header
    return HTML(svg)

def line_segments_texture(pts, edges, width=16, dpi=144):
    bb_min = np.min(pts[:, 0:2], axis=0)
    bb_max = np.max(pts[:, 0:2], axis=0)
    bb_dim = bb_max - bb_min
    aspect = bb_dim[0] / bb_dim[1]
    fig = plt.figure(figsize=(width, width / aspect))

    # Break into contiguous paths (point lists)
    paths = []
    prevPtIdx = edges[0][0]
    currPath = [pts[edges[0][0]]]
    for e in edges:
        if (e[0] == prevPtIdx):
            currPath.append(pts[e[1]])
        else:
            paths.append(np.array(currPath))
            currPath = [pts[e[0]], pts[e[1]]]
        prevPtIdx = e[1]
    paths.append(np.array(currPath))

    xvals = [p[:, 0] for p in paths]
    yvals = [p[:, 1] for p in paths]
    plt.axis('off')
    plt.plot(*(c for t in zip(xvals, yvals) for c in t))
    plt.xlim(bb_min[0], bb_max[0])
    plt.ylim(bb_min[1], bb_max[1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0) # make subplot fill the entire figure!
    # This works but gives lower quality than the PNG output for some reason:
    # fig.canvas.draw()
    # w, h = fig.canvas.get_width_height()
    # buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    # buf.shape = (h, w, 3)
    # plt.close()
    # return buf

    from PIL import Image
    from io import BytesIO
    imgBuffer = BytesIO()
    plt.savefig(imgBuffer, format='png', dpi=dpi)
    plt.close()
    imgBuffer.seek(0)
    im = Image.open(imgBuffer)
    return np.array(im)

def vectorFieldFromISheetVarField(isheet, varField):
    '''Convert a variable vector from an inflatable sheet into a per-visualization-mesh-vertex vector field'''
    # The visualization mesh vertices consist of the top sheet vertices (sheet_idx = 0) concatenated with the bottom sheet vertices (sheet_idx = 1)
    assert(varField.shape[0] == isheet.numVars())
    nv = isheet.visualizationMesh().numVertices()
    nvTopSheet = nv // 2
    result = np.empty((nv, 3))
    for sheet_idx in range(2):
        for vtx in range(nvTopSheet):
            vo = isheet.varIdx(sheet_idx, vtx)
            result[sheet_idx * nvTopSheet + vtx, :] = varField[vo:vo + 3]
    return result

# Wrapper for visualizing the equilibrium configuration of an inflatable sheet
class EquilibriumMesh:
    def __init__(self, isheet):
        self.isheet = isheet
    def visualizationGeometry(self):
        return self.isheet.visualizationMesh().visualizationGeometry()

import MeshFEM, tri_mesh_viewer
def forceViewer(isheet):
    return tri_mesh_viewer.TriMeshViewer(isheet.visualizationMesh(), vectorField=vectorFieldFromISheetVarField(isheet, isheet.gradient()))

################################################################################
# Functions for extracting/exporting render data for the paper.
################################################################################
import os, MeshFEM, mesh, inflation, sheet_meshing, mesh_operations

# Decompose the mesh into numColors distinct partitions of the wall mesh components
# (via face labels 0..numColors) and the tube meshes (label -1)
def getWallColorComponentsAndColors(isheet):
    nc, wallLabels = sheet_meshing.wallMeshComponents(isheet)

    np.random.seed(0)
    colorPermutation = np.random.permutation(nc)
    isWall = wallLabels >= 0
    wallLabels[isWall] = colorPermutation[wallLabels[isWall]]

    colors = getMatplotlibLineColors()
    numColors = len(colors)

    wallLabels[isWall] = wallLabels[isWall] % numColors
    return wallLabels, colors

def getWallIndicatorColors(isheet, tubeColor = [1, 1, 1, 0.5]):
    wallLabels, colors = getWallColorComponentsAndColors(isheet)

    FColors = colors[wallLabels]
    FColors[wallLabels == -1] = tubeColor
    return FColors

def explodeFaceColors(V, F, N, FColors):
    nRepV = 3 * F.shape[0]
    FRep = np.arange(np.prod(F.shape), dtype=np.int).reshape(-1, 3)
    VColors = np.empty((nRepV, FColors.shape[1]))
    VColors[FRep] = np.repeat(FColors[:, np.newaxis, :], 3, 1)
    return V[F].reshape(-1, 3), FRep, N[F].reshape(-1, 3), VColors
def explodeFaceColorsOfMesh(m, FColors):
    return explodeFaceColors(m.vertices(), m.triangles(), m.vertexNormals(), FColors)

def saveColoredMesh(pathPrefix, m, wallLabels, colors, tubeColor = [1, 1, 1, 0.5]):
    FColors = colors[wallLabels]
    FColors[wallLabels == -1] = tubeColor

    V, F, N, C = explodeFaceColorsOfMesh(m, FColors)
    mesh_operations.saveOBJWithNormals(pathPrefix + '.obj', V, F, N)
    np.savetxt(pathPrefix + '.colors.txt', C * 255, fmt='%i')

def saveColoredComponentMeshes(pathPrefix, m, wallLabels, colors, numTopSheetTris = None):
    V = m.vertices()
    F = m.triangles()
    tubeMesh = mesh.Mesh(*mesh_operations.removeDanglingVertices(V, F[wallLabels == -1]), embeddingDimension=3)
    # Note: the tube mesh may be non-manifold, so directly calling `tubeMesh.vertexNormals()` may hang.
    mesh_operations.saveOBJWithNormals(f'{pathPrefix}.tubes.obj', tubeMesh.vertices(), tubeMesh.triangles(), mesh_operations.getVertexNormals(tubeMesh))

    includeTri = np.zeros(len(wallLabels), dtype=np.bool)
    for i, c in enumerate(colors):
        filename = f'{pathPrefix}.wall_color_' + '_'.join(map(lambda x: str(int(x)), c[0:-1] * 255)) + '.obj'
        includeTri[:numTopSheetTris] = (wallLabels == i)[:numTopSheetTris] # discard the coinciding bottom sheet wall triangles (if numTopSheetTris is not None)
        wallMesh = mesh.Mesh(*mesh_operations.removeDanglingVertices(V, F[includeTri]), embeddingDimension=3)
        mesh_operations.saveOBJWithNormals(filename, wallMesh.vertices(), wallMesh.triangles(), mesh_operations.getVertexNormals(wallMesh))

    includeTri[:numTopSheetTris] = (wallLabels >= 0)[:numTopSheetTris]
    wallMesh = mesh.Mesh(*mesh_operations.removeDanglingVertices(V, F[includeTri]), embeddingDimension=3)
    # Like with the tube mesh, vertexNormals() can hang...
    mesh_operations.saveOBJWithNormals(f'{pathPrefix}.allWalls.obj', wallMesh.vertices(), wallMesh.triangles(), mesh_operations.getVertexNormals(wallMesh))

def writeVisDataForMesh(pathPrefix, m, numTopSheetTris, wallLabels, colors):
    topSheetMesh = m
    if (topSheetMesh.numTris() != numTopSheetTris):
        if (topSheetMesh.numTris() != 2 * numTopSheetTris): raise Exception(f'Triangle count mismatch: {topSheetMesh.numTris()} vs {2 * numTopSheetTris}')
        topSheetMesh = mesh.Mesh(*mesh_operations.removeDanglingVertices(m.vertices(), m.triangles()[0:numTopSheetTris]))
    for (i, loop) in enumerate(mesh_operations.boundaryLoops(topSheetMesh)):
        np.savetxt(f'{pathPrefix}.bdryloop{i}.txt', loop)

    #saveColoredMesh(pathPrefix, m, wallLabels, colors)
    saveColoredComponentMeshes(pathPrefix, m, wallLabels, colors, numTopSheetTris=numTopSheetTris)

def writeVisualizationDataForSheet(directoryPath, sheet, origDesign, optDesign = None, origInflated = None, optInflated = None):
    os.makedirs(directoryPath, exist_ok=True)
    numTopSheetTris = origDesign.numTris()
    wallLabels, colors = getWallColorComponentsAndColors(sheet)

    writeVisDataForMesh(f'{directoryPath}/orig', origDesign, numTopSheetTris, wallLabels, colors)
    if optDesign is not None:
        writeVisDataForMesh(f'{directoryPath}/opt', optDesign, numTopSheetTris, wallLabels, colors)
    if origInflated is not None:
        writeVisDataForMesh(f'{directoryPath}/orig.infl', origInflated, numTopSheetTris, np.concatenate((wallLabels, wallLabels)), colors)
    if optInflated is not None:
        writeVisDataForMesh(f'{directoryPath}/opt.infl', optInflated, numTopSheetTris, np.concatenate((wallLabels, wallLabels)), colors)

def writeVisualizationData(directoryPath, iwv, origDesign, optDesign = None, origInflated = None, optInflated = None):
    writeVisualizationDataForSheet(inflation.InflatableSheet(origDesign, iwv), origDesign, optDesign = None, origInflated = None, optInflated = None)

def saveTubeAndWallMeshes(pathPrefix, isheet, scale=1.0):
    m = isheet.visualizationMesh(duplicateFusedTris=True)
    numTopSheetTris = isheet.mesh().numTris()
    wallLabels, colors = getWallColorComponentsAndColors(isheet)
    wallLabels = np.concatenate((wallLabels, wallLabels))
    
    V = scale * m.vertices()
    F = m.triangles()
    tubeMesh = mesh.Mesh(*mesh_operations.removeDanglingVertices(V, F[wallLabels == -1]), embeddingDimension=3)
    # Note: the tube mesh may be non-manifold, so directly calling `tubeMesh.vertexNormals()` may hang.
    mesh_operations.saveOBJWithNormals(f'{pathPrefix}.tubes.obj', tubeMesh.vertices(), tubeMesh.triangles(), mesh_operations.getVertexNormals(tubeMesh))

    includeTri = np.zeros(len(wallLabels), dtype=np.bool)
    includeTri[:numTopSheetTris] = (wallLabels >= 0)[:numTopSheetTris]
    wallMesh = mesh.Mesh(*mesh_operations.removeDanglingVertices(V, F[includeTri]), embeddingDimension=3)
    # Like with the tube mesh, vertexNormals() can hang...
    mesh_operations.saveOBJWithNormals(f'{pathPrefix}.allWalls.obj', wallMesh.vertices(), wallMesh.triangles(), mesh_operations.getVertexNormals(wallMesh))

def writeParametrizationVisualizations(pathPrefix, rparam, paramBounds = [(-625, 700), (-650, 600)], histYlim = (0, 1000)):
    singularValueHistogram(rparam)
    plt.title('Local-Global Parametrization Singular Value $\sigma_0$, $\sigma_1$ Histogram')
    plt.ylim(histYlim)
    plt.xlim((0.9, 1.6))
    plt.tight_layout()
    plt.savefig(f'{pathPrefix}svhist.pdf')

    visualizeScaleFactors(rparam, width=10, height=10)
    plt.xlim(paramBounds[0])
    plt.ylim(paramBounds[1])
    plt.tight_layout()
    plt.savefig(f'{pathPrefix}scale_factors.pdf')

    visualizeChannelOrientationSubsampled(rparam, numSamples=1000, width=10, height=10)
    plt.xlim(paramBounds[0])
    plt.ylim(paramBounds[1])
    plt.tight_layout()
    plt.savefig(f'{pathPrefix}orientations_hue.pdf')

    visualizeChannelOrientationSubsampled(rparam, orientationHue=False, numSamples=1000, width=10, height=10)
    plt.xlim(paramBounds[0])
    plt.ylim(paramBounds[1])
    plt.tight_layout()
    plt.savefig(f'{pathPrefix}orientations_nohue.pdf')
    plt.close(), plt.close(); plt.close(); plt.close()

def getFlatViewer(isheet, width, height, offscreen):
    # Color each wall region using the matplotlib color scheme
    flat_colors = getWallIndicatorColors(isheet, tubeColor=[1.0, 1.0, 1.0, 1.0])[:, 0:3]
    Viewer = tri_mesh_viewer.OffscreenTriMeshViewer if offscreen else tri_mesh_viewer.TriMeshViewer
    flat_viewer= Viewer(isheet.mesh(), scalarField=flat_colors,   width=width, height=height)
    flat_viewer.renderer.transparentBackground = False
    flat_viewer.renderer.lineWidth = 0.75
    flat_viewer.renderer.lineColor = [0.5, 0.5, 0.5, 1.0]
    return flat_viewer

def optimizationViewers(isheet, width, height, offscreen):
    flat_viewer = getFlatViewer(isheet, width, height, offscreen)
    deploy_colors = flat_viewer.scalarField.copy() # TODO: we don't need to replicate here since visualizationField does that for us.

    Viewer = tri_mesh_viewer.OffscreenTriMeshViewer if offscreen else tri_mesh_viewer.TriMeshViewer
    deploy_viewer = Viewer(isheet, scalarField=deploy_colors, width=width, height=height)
    deploy_viewer.renderer.transparentBackground = False
    deploy_viewer.renderer.lineWidth = 0.75
    deploy_viewer.renderer.lineColor = [0.5, 0.5, 0.5, 1.0]
    deploy_viewer.scalarFieldGetter = ISheetScalarField.WALL_COLOR()
    deploy_viewer.normalCreaseAngle = np.pi / 6
    return (flat_viewer, deploy_viewer)

import vis
class ISheetScalarField:
    class NONE:
        def __call__(self, isheet):
            return None

    class WALL_COLOR:
        def __init__(self): self.colors = None
        def __call__(self, isheet):
            if self.colors is None:
                self.colors = getWallIndicatorColors(isheet, tubeColor=[1.0, 1.0, 1.0, 1.0])
            return self.colors

    class TGT_DIST:
        def __init__(self, isheet, targetSurf, vmin=None, vmax=None, cmap=matplotlib.cm.Spectral_r):
            import measurements
            self.dist = measurements.MidsurfaceDistMeasurer(isheet, targetSurf, relative=True)
            self.cmap = cmap
            initialDists = self.dist(isheet)
            if vmin is None: vmin = initialDists.min()
            if vmax is None: vmax = initialDists.max()
            self.vmin = vmin
            self.vmax = vmax

        def __call__(self, isheet):
            return vis.fields.ScalarField(isheet, self.dist(isheet), vmin=self.vmin, vmax=self.vmax, colormap=self.cmap)

def figImageData(fig, dpi = 72):
    fig.set_dpi(dpi)
    plt.tight_layout()
    fig.canvas.draw()
    return fig.canvas.buffer_rgba().tobytes()

def figToPIL(fig, dpi = 72):
    from PIL import Image
    return Image.frombytes('RGBA', fig.canvas.get_width_height(), figImageData(fig, dpi))

def plotEnergies(energies, width = 10, height = 8, iterate = None, keys = None, colors = getMatplotlibLineColors()):
    fig = plt.figure(figsize=(width, height))
    if keys is None:
        keys = list(energies[0].keys())
    for i, k in enumerate(keys):
        plt.plot([e[k] for e in energies], color=colors[i], label=k)
        if iterate is not None:
            plt.scatter(iterate, energies[iterate][k], s=4**2, color=colors[i])
    plt.grid()
    plt.legend()
    plt.xlabel("Iterate")
    plt.ylabel("Energy")
    plt.tight_layout()
    return fig

import pickle, gzip
import video_writer
def renderOptimizationIterateData(tas_in_path_fmt, out_path_prefix, opt_report, width, height, deployCam = None, flatCam = None, upscale=2, maxFrames = np.inf):
    """
    Render and plot optimization iterate data from pickled target-attracted sheet files
    `tas_in_path_fmt.format(i)` for i in 0, 1... and the optimization report 'opt_report'
    to videos `out_path_prefix + 'flat.mp4'`
              `out_path_prefix + 'deploy.mp4'`
              `out_path_prefix + 'energies.mp4'`
    """
    in_path  = lambda i: tas_in_path_fmt.format(i)
    if not os.path.exists(in_path(0)): raise Exception('Input data not found at ' + in_path(0))
    in_sheet = lambda i: pickle.load(gzip.open(in_path(i), 'rb')).sheet()
    sheet = in_sheet(0) # Must happen here so sheet/mesh doesn't go out of scope...
    flat_viewer, deploy_viewer = optimizationViewers(sheet, width * upscale, height * upscale, offscreen=True)
    plot_writer = None
    if opt_report is not None:
        fig = plotEnergies(opt_report.energies)
        plot_writer = video_writer.PlotVideoWriter(out_path_prefix + 'energies.mp4', fig)
        plt.close()

    if   flatCam is not None:   flat_viewer.setCameraParams(  flatCam)
    if deployCam is not None: deploy_viewer.setCameraParams(deployCam)

    flat_viewer  .recordStart(out_path_prefix +   'flat.mp4', outWidth=width, outHeight=height, streaming=True)
    deploy_viewer.recordStart(out_path_prefix + 'deploy.mp4', outWidth=width, outHeight=height, streaming=True)
    i = 0
    while os.path.exists(in_path(i)):
        if (i > maxFrames): break
        sheet = in_sheet(i)
        flat_viewer  .update(mesh=sheet.mesh(), scalarField=  flat_viewer.scalarField)
        deploy_viewer.update(mesh=sheet,        scalarField=deploy_viewer.scalarField)

        if plot_writer is not None:
            fig = plotEnergies(opt_report.energies, iterate=i)
            plot_writer.writeFrame(fig)
            plt.close()
        i = i + 1
    flat_viewer  .recordStop()
    deploy_viewer.recordStop()

def renderOptimizationIterates(sheet_opt, out_path_prefix, **customKWArgs):
    """
    Render and plot optimization iterate data for the optimizer `sheet_opt`
    to videos `out_path_prefix + 'flat.mp4'`
              `out_path_prefix + 'deploy.mp4'`
              `out_path_prefix + 'energies.mp4'`
    the arguments of `renderOptimizationIterateData` can be manually specified
    using `customKWArgs`.
    Note, this must be run from the same `cwd` as `sheet_opt` was created in
    (so that the relative paths in `sheet_opt` are correct).
    """
    kwArgs = customKWArgs.copy()
    # Default to argument values stored in sheet_opt if not overriden by customKWArgs
    width, height = sheet_opt.flat_viewer.getSize()
    kwArgs.setdefault('width',     width  // 2) # Upscaling was already applied...
    kwArgs.setdefault('height',    height // 2)
    kwArgs.setdefault('flatCam',   sheet_opt.flat_viewer  .getCameraParams())
    kwArgs.setdefault('deployCam', sheet_opt.deploy_viewer.getCameraParams())
    renderOptimizationIterateData(sheet_opt.sheetOutputDir + '/tas_{}.pkl.gz', out_path_prefix, sheet_opt.report, **kwArgs)

def strainHistograms(isheet):
    import utils
    strains = utils.getStrains(isheet)
    iwt = np.array([isheet.isWallTri(t) for t in range(isheet.mesh().numTris())] * 2)
    fig = plt.figure(figsize=(12,6))
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.hist(strains[ iwt, i], bins=1000, label='wall', alpha=0.75);
        plt.hist(strains[~iwt, i], bins=1000, label='tube', alpha=0.75);
        if (i == 0): plt.xlim(-0.7, 0.2);
        plt.yscale('log')
        plt.legend()

def designAlterationStretchHistograms(reducedSheetOptimizer):
    isheet = reducedSheetOptimizer.sheet()
    cb = reducedSheetOptimizer.collapseBarrier()
    design_modification_sigmas = np.array([cb.collapsePreventionEnergy(t).svd().Sigma() for t in range(isheet.mesh().numTris())])
    iwt = np.array([isheet.isWallTri(t) for t in range(isheet.mesh().numTris())])
    fig = plt.figure(figsize=(12,6))
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.hist(design_modification_sigmas[ iwt, i], bins=1000, label='wall', alpha=0.75);
        plt.hist(design_modification_sigmas[~iwt, i], bins=1000, label='tube', alpha=0.75);
        plt.yscale('log')
        plt.legend()

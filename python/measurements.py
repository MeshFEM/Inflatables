import inflation
import numpy as np

class DistMeasurer:
    def __init__(self, numPts, targetSurf, relative=False):
        self.tsf = inflation.TargetSurfaceFitter(targetSurf)
        self.tsf.setQueryPtWeights(np.ones(numPts))
        self.scale = 1.0 / np.linalg.norm(targetSurf.bbox[1] - targetSurf.bbox[0]) if relative else 1.0
        self.isBoundary = np.zeros(numPts, dtype=bool) # don't treat boundary vertices specially (let them project to interior points)

    def __call__(self, q):
        self.tsf.updateClosestPoints(q, self.isBoundary)
        return np.linalg.norm(self.tsf.closestSurfPts - q, axis=1) * self.scale

class MidsurfaceDistMeasurer(DistMeasurer):
    def __init__(self, isheet, targetSurf, relative=False):
        super().__init__(isheet.mesh().numVertices(), targetSurf, relative)

    def __call__(self, isheet):
        m = isheet.mesh()
        q = np.array([0.5 * (isheet.getDeformedVtxPosition(vi, 0) + isheet.getDeformedVtxPosition(vi, 1)) for vi in range(m.numVertices())])
        return super().__call__(q)

def getApproxMidsurfaceDistances(isheet, targetSurf, relative=False):
    """
    Gets a scalar field over the full visualization mesh measuring the distance
    from the approximate midsurface (computed by averaging top and bottom tube
    vertices) to the target surface.

    If `relative` is True, distances are reported relative to the target
    surface's bounding box diagonal
    """
    return MidsurfaceDistMeasurer(isheet, targetSurf, relative)(isheet)

def getWallDistances(isheet, targetSurf, relative=False):
    """
    Get the distances from wall vertices to `targetSurf`.
    If `relative` is True, distances are reported relative to the target
    surface's bounding box diagonal
    """
    q = isheet.deformedWallVertexPositions()
    return DistMeasurer(len(q), targetSurf, relative)(q)

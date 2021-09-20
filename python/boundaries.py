import numpy as np

def getBoundaryVars(isheet):
    return [isheet.varIdx(0, i, c) for i in isheet.mesh().boundaryVertices() for c in range(3)]

def getOuterBoundaryVars(isheet):
    """
    Variables for vertices on the *outer* boundary loop of the mesh (excluding inner hole boundaries).
    Uses a simple heuristic: we expect the outermost boundary loop to be the longest.
    This of course can break for particularly jagged inner boundaries.
    """
    m = isheet.mesh()
    V = m.vertices()
    BV = m.boundaryVertices()
    arclen = lambda l: np.linalg.norm(np.diff(V[BV[np.array(l)]], axis=0), axis=1).sum()
    outerLoopBdryVertices = max(m.boundaryLoops(), key=arclen)
    return [isheet.varIdx(0, BV[bvi], c) for bvi in outerLoopBdryVertices for c in range(3)]

def getBdryWallVars(isheet):
    """
    Variables for wall boundary vertices that are also mesh boundary vertices.
    """
    import numpy as np
    m = isheet.mesh()
    isTrueWallVertex = np.zeros(m.numVertices(), dtype=np.bool)
    isTrueWallVertex[isheet.trueWallVertices()] = True
    bv = m.boundaryVertices()
    bdryWallVertices = [v for v in bv if isTrueWallVertex[v]]
    return [isheet.varIdx(0, i, c) for i in bdryWallVertices for c in range(3)]

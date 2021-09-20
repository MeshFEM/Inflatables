import numpy as np
def squareWithVerticalChannels(channelHeightFrac, numChannels):
    # border
    pts = [[0, 0], [1, 0], [1, 1], [0, 1]]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    channelSpacing = 1.0 / numChannels
    ybot = (1.0 - channelHeightFrac) * 0.5
    ytop = 1.0 - ybot
    for wall in range(1, numChannels):
        x = wall * channelSpacing
        edges.append(tuple([len(pts), len(pts) + 1]))
        pts += [[x, ybot], [x, ytop]]
    return pts, edges

def parallelTubes(N, h, d, w, triArea):
    """
    Generate a rectangular sheet mesh with N parallel tubes of height h with
    tube width w and fusing curve width d.
    Unlike the other functions in this module, this creates an actual triangle mesh
    """
    tubeXCoords = [0, w] # tube 0
    nonemptyFuseRegion = d > 1e-10
    for i in range(1, N):
        l_old = tubeXCoords[-1]
        if nonemptyFuseRegion: tubeXCoords += [l_old + d, l_old + d + w] # fuse gap + tube i 
        else:                  tubeXCoords += [l_old + w] # just tube i
    pts = [[x, y] for y in [0, h] for x in tubeXCoords]
    numBottomPts = len(tubeXCoords)
    edges  = [(i, i + 1) for i in range(numBottomPts - 1)]                      # bottom edges
    edges += [(i, i + 1) for i in range(numBottomPts, 2 * numBottomPts - 1)]    # top edges
    edges += [(i, numBottomPts + i) for i in range(numBottomPts)]               # vertical edges

    import wall_generation
    m, fuseMarkers, fuseSegments = wall_generation.triangulate_channel_walls(pts, edges, triArea)

    if (nonemptyFuseRegion):
        import sheet_meshing
        fakeSignedDistance = lambda x: w - np.fmod(x, d + w) # will be negative inside the wall region
        pointSetLiesInWall = lambda X: np.mean(fakeSignedDistance(X[:, 0])) < 0.0
        m, iwv, iwbv = sheet_meshing.remeshWallRegions(m, fuseMarkers, fuseSegments, pointSetLiesInWall)
    else:
        iwv = iwbv = fuseMarkers

    return m, iwv, iwbv

def concentricCircles(numChannels, numSegments):
    pts, edges = [], []
    for wall in range(1, numChannels + 1): # last wall is actually the pillow boundary
        # The innermost circle's diameter should match the channel width, so its radius
        # is half the channel spacing (treat it like a half-channel).
        r = (0.5 + (wall - 1)) / (numChannels - 0.5)
        
        angles = np.linspace(0, 2 * np.pi, numSegments, endpoint = False)
        newPtIdxs = list(range(len(pts), len(pts) + len(angles)))
        pts += [[r * np.cos(t), r * np.sin(t)] for t in angles]
        edges += zip(newPtIdxs, newPtIdxs[1:] + [newPtIdxs[0]])
    return pts, edges

def circle(resolution):
    t = np.linspace(0, 2 * np.pi, resolution, endpoint = False)
    pts   = list(np.column_stack((np.cos(t), np.sin(t))))
    edges = list(zip(range(resolution), list(range(1, resolution)) + [0]))
    return pts, edges

def radialChannels(numSectors, numSectorSubdivisions, circleResolution):
    pts, edges = circle(circleResolution)

    def subdivide(nsubdiv, angleStart, angleEnd, rStart, rEnd):
        if (nsubdiv == 0): return
        angleMid = 0.5 * (angleStart + angleEnd)
        alpha = 0.5
        rMid     = (1 - alpha) * rStart + alpha * rEnd

        edges.append((len(pts), len(pts) + 1))
        pts.append([rMid * np.cos(angleMid), rMid * np.sin(angleMid)])
        pts.append([rEnd * np.cos(angleMid), rEnd * np.sin(angleMid)])

        subdivide(nsubdiv - 1, angleStart, angleMid, rMid, rEnd)
        subdivide(nsubdiv - 1, angleMid  , angleEnd, rMid, rEnd)

    for sector in range(numSectors):
        sectorBegin = (2 * np.pi * sector) / numSectors
        sectorEnd =   (2 * np.pi * (sector + 1)) / numSectors

        rStart = 0.06
        rEnd = 0.95

        edges.append((len(pts), len(pts) + 1))
        pts.append([rStart * np.cos(sectorBegin), rStart * np.sin(sectorBegin)])
        pts.append([rEnd   * np.cos(sectorBegin), rEnd   * np.sin(sectorBegin)])

        subdivide(numSectorSubdivisions, sectorBegin, sectorEnd, rStart, rEnd)

    return pts, edges

from matplotlib import pyplot as plt

# alpha: angle between spiral tangent and axis vector dtheta (not dr)
def logSpiralPlot(alpha = 70.0, radius = 1.0, minDist = 0.05, margin = 0.05, edgeLength = 0.02):
    alpha_rad = np.deg2rad(alpha)
    b = np.tan(alpha_rad)

    # Note: the logarithmic spiral is self-similar, so its scale is irrelevant.
    # We therefore use the unit-scaled logarithmic spiral, given in polar coordinates by:
    #   r = e^(b theta)
    # We evaluate the spiral at evenly spaced points along its arclength.
    sqrtTerm = np.sqrt(1 + 1 / (b * b))
    rForTheta      = lambda th: np.exp(b * th)
    thetaForR      = lambda r: np.log(r) / b
    thetaForArclen = lambda s: (1.0 / b) * (np.log(s + sqrtTerm) - np.log((sqrtTerm)))
    arclenForTheta = lambda th: sqrtTerm * (np.exp(b * th) - 1.0)

    def thetasForRadiusInterval(rmin, rmax):
        smin, smax = map(lambda r: arclenForTheta(thetaForR(r)), [rmin, rmax])
        nsubdiv = int(np.round((smax - smin) / edgeLength))
        return thetaForArclen(np.linspace(smin, smax, nsubdiv))

    pts, edges = circle(int(np.round(2 * radius * np.pi / edgeLength)))

    def generate_points(rs, thetas, rotation = 0):
        return np.column_stack((rs * np.cos(thetas + rotation), rs * np.sin(thetas + rotation)))

    numSectors = 1
    # Draw walls (spiral arms) dividing the circle into numSectors sectors
    # for numSectors in 2, 4, 8, ...
    while True:
        # We approximate the channel thickness by multiplying the channel's
        # sector angle by the normal velocity of spiral arm (wall) points as
        # the arms are rotated at unit angular velocity.
        #       thickness ~= (2 * pi / numSectors) * r * sin(alpha)
        # where sin(alpha) is the (constant) angle between the curve's normal
        # and the radial axis vector dr.
        # Then we can solve for the minimum radius such that the thickness is >= minDist:
        #       (2 * pi / numSectors) * r * sin(alpha) >= minDist   ==>
        #       r >= (minDist * numSectors) / (2 * pi * sin(alpha))
        rmin = (minDist * numSectors) / (2 * np.pi * np.sin(alpha_rad))
        # The following version reproduces the original Matlab behavior (but
        # leads to tightly spaced channels for small alpha)
        # rmin = max(minDist / 2, rmin) if numSectors > 2 else minDist / 2
        rmin = max(minDist / 2, rmin)
        rmax = radius - margin
        if (rmin > rmax - edgeLength): break # admissible channel walls have shrunk below the target edge length
        thetas = thetasForRadiusInterval(rmin, rmax)
        rvalues = rForTheta(thetas)

        for arm in range(numSectors):
            if ((numSectors > 1) and (arm % 2 == 0)): continue # even arms have already been drawn by previous passes
            newPts = list(generate_points(rvalues, thetas, arm * (2 * np.pi) / numSectors))
            if (len(newPts) >= 2): # At least two points must be added to form a segment
                ptOffset = len(pts)
                pts += newPts
                for i in range(len(newPts) - 1):
                    edges.append((ptOffset + i, ptOffset + i + 1))
        numSectors *= 2

    return pts, edges

def bentArc(length, width, curvature, numArcSegments, includeStart = True, includeEnd = True):
    r = 1.0 / curvature
    thetaLen = length / r
    
    thetaStart, thetaEnd = 0, thetaLen
    if (not includeStart): thetaStart += thetaLen / numArcSegments
    if (not includeEnd):   thetaEnd   -= thetaLen / numArcSegments
    numArcSegments -= includeStart + includeEnd
    
    thetas = np.linspace(thetaStart, thetaEnd, numArcSegments + 1)[:, np.newaxis]
    rOuter, rInner = r + width / 2, r - width / 2
    circlePts = np.concatenate((np.cos(thetas),
                                np.sin(thetas)), axis=1)
    pts = np.concatenate((rOuter * circlePts, rInner * circlePts[::-1]), axis=0)
    pts -= [r, 0]
    
    arcPtIdxs = np.arange(numArcSegments + 1)
    edges = list(zip(arcPtIdxs, arcPtIdxs[1:]))

    if includeEnd: edges.append((numArcSegments, numArcSegments + 1))
    innerPtOffset = numArcSegments + 1
    edges += zip(innerPtOffset + arcPtIdxs, innerPtOffset + arcPtIdxs[1:])
    if includeStart: edges.append((len(pts) - 1, 0))

    return pts, edges

# Get a vector field by sampling the contraction direction at "pts"
def bentArcContractionDirection(curvature, pts):
    r = 1.0 / curvature;
    pts = pts + [r, 0] # Undo translation
    return pts / np.linalg.norm(pts, axis=1)[:, np.newaxis]

# numLinks: number of links in the chain of circular arcs
# (if numLinks == 1, we get an s-shaped curve)
def bentArchChain(numLinks, length, width, curvature, numArcSegments):
    r = 1.0 / curvature
    thetaLen = length / r
    pts, edges = np.empty((0, 2)), []
    rotMat = lambda x: np.array([[np.cos(x), -np.sin(x)],[np.sin(x), np.cos(x)]])
    terminals = None
    for l in range(numLinks):
        # Transform the old points so that the new chain will connect at the origin
        pts = (pts + [r, 0]) @ rotMat(-thetaLen).transpose() - [r, 0]
        # Reflect the chain for every added link to curve in the opposite direction.
        # Note: we make no guarantees on the curve orientation, so we needn't reverse
        #       any edges!
        pts *= [-1, 1]
        link_pts, link_edges = bentArc(length, width, curvature, numArcSegments, l == 0, l == numLinks - 1)
        offset = pts.shape[0]
        pts = np.concatenate((pts, link_pts), axis=0)
        # Connect with the terminals of the previous chain link (if one exists)
        if (terminals is not None):
            edges += [(terminals[1], offset), (terminals[0], offset + len(link_pts) - 1)]

        edges += [(i + offset, j + offset) for i, j in link_edges]
        terminals = (offset + numArcSegments - 1 + (l > 0), offset + numArcSegments + (l > 0))
        
    return pts, edges

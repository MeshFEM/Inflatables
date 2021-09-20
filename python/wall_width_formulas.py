# Wall widths are canonically in the range [0, 2pi] to match the
# stripe distance field (which takes value 0 at the wall center and pi at the
# air channel center). However, based on the physical spacing between
# air channels, this canonical width corresponds to different physical wall
# widths.
import numpy as np

def wallWidthForCanonicalWidth(canonicalWidth, channelSpacing):
    return canonicalWidth * channelSpacing / (2 * np.pi)

def canonicalWallWidthForGeometry(wallWidth, channelSpacing):
    return wallWidth * 2 * np.pi / channelSpacing

def spacingForCanonicalAndPhysicalWallWidths(canonicalWidth, physicalWidths):
    return physicalWidths * 2 * np.pi / canonicalWidth

# Largest singular value of the mapping from 3D to the plane.
def stretchFactorForCanonicalWallWidth(w):
    # stretched width / unstretched width
    return 2 * np.pi / (w + (2 / np.pi) * (2 * np.pi - w))

# Inverse of stretchFactorForCanonicalWallWidth
def canonicalWallWidthForStretchFactor(s):
    return ((2 * np.pi / s) - 4) / (1 - 2 / np.pi)

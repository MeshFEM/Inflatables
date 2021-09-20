import os as _os
import boundaries
INFLATABLES_PYROOT = _os.path.dirname(__file__)
################################################################################
# Helper classes for specifying/applying job configurations
################################################################################
class FixedVarsBoundary:
    name = "bdry"
    @staticmethod
    def get(isheet):
        return boundaries.getBoundaryVars(isheet)

class FixedVarsBoundaryWall:
    name = "bdrywall"
    @staticmethod
    def get(isheet):
        return boundaries.getBoundaryVars(isheet)

class FixedVarsNone:
    name = "none"
    @staticmethod
    def get(isheet):
        return []

class FusingCurveSmoothnessParams:
    def __init__(self, dirichletWeight, laplacianWeight, lengthScaleSmoothingWeight, curvatureWeight):
        self.dirichletWeight            = dirichletWeight
        self.laplacianWeight            = laplacianWeight
        self.lengthScaleSmoothingWeight = lengthScaleSmoothingWeight
        self.curvatureWeight            = curvatureWeight

    def apply(self, rso):
        fcs = rso.fusingCurveSmoothness()
        fcs.dirichletWeight            = self.dirichletWeight
        fcs.laplacianWeight            = self.laplacianWeight
        fcs.lengthScaleSmoothingWeight = self.lengthScaleSmoothingWeight
        fcs.curvatureWeight            = self.curvatureWeight

    def __repr__(self):
        return f'dirichlet_{self.dirichletWeight:g}_laplacian_{self.laplacianWeight:g}_scalesmoothing_{self.lengthScaleSmoothingWeight:g}_curvature_{self.curvatureWeight:g}'

from enum import Enum
class Solver(Enum):
    SCIPY = 0
    IPOPT = 1

# Default settings that can be overriden
redirect_io = True
solver = Solver.IPOPT
design_opt_iters = 1000
blenderVisXF = lambda x: x # Additional transform to apply after the renderingNormalization transform
debug = False
relaxedStiffnessEpsilon = 1e-10

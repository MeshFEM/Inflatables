import numpy as np
from numpy.linalg import norm
from MeshFEM import sparse_matrices

def preamble(obj, xeval, perturb, etype, fixedVars = []):
    if (xeval   is None): xeval = obj.getVars()
    if (perturb is None): perturb = np.random.uniform(low=-1,high=1, size=obj.numVars())
    if (etype   is None): etype = obj.__class__.EnergyType.Full
    xold = obj.getVars()
    perturb = np.copy(perturb)
    perturb[fixedVars] = 0.0
    return (xold, xeval, perturb, etype)

def fdGrad(obj, fd_eps, xeval = None, perturb = None, etype = None, fixedVars = []):
    xold, xeval, perturb, etype = preamble(obj, xeval, perturb, etype, fixedVars)

    def evalAt(x):
        obj.setVars(x)
        val = obj.energy(etype)
        return val

    fd_delta_E = (evalAt(xeval + perturb * fd_eps) - evalAt(xeval - perturb * fd_eps)) / (2 * fd_eps)
    obj.setVars(xold)

    return fd_delta_E

def validateGrad(obj, fd_eps = 1e-6, xeval = None, perturb = None, etype = None, fixedVars = []):
    xold, xeval, perturb, etype = preamble(obj, xeval, perturb, etype, fixedVars)
    
    obj.setVars(xeval)
    g = obj.gradient(etype)
    analytic_delta_E = g.dot(perturb)

    fd_delta_E = fdGrad(obj, fd_eps, xeval, perturb, etype, fixedVars)

    return (fd_delta_E, analytic_delta_E)

def validateHessian(obj, fd_eps = 1e-6, xeval = None, perturb = None, etype = None, fixedVars = []):
    xold, xeval, perturb, etype = preamble(obj, xeval, perturb, etype, fixedVars)

    def gradAt(x):
        obj.setVars(x)
        val = obj.gradient(etype)
        return val

    obj.setVars(xeval)
    h = obj.hessian(etype)
    fd_delta_grad = (gradAt(xeval + perturb * fd_eps) - gradAt(xeval - perturb * fd_eps)) / (2 * fd_eps)
    analytic_delta_grad = h.apply(perturb)

    obj.setVars(xold)

    return (norm(analytic_delta_grad - fd_delta_grad) / norm(fd_delta_grad), fd_delta_grad, analytic_delta_grad)

def gradConvergence(obj, perturb=None, energyType=None, fixedVars = []):
    epsilons = np.logspace(-9, -3, 100)
    errors = []
    if (energyType is None): energyType = obj.EnergyType.Full
    if (perturb is None): perturb = np.random.uniform(-1, 1, size=obj.numVars())
    for eps in epsilons:
        fd, an = validateGrad(obj, etype=energyType, perturb=perturb, fd_eps=eps, fixedVars = fixedVars)
        err = np.abs(an - fd) / np.abs(an)
        errors.append(err)
    return (epsilons, errors, an)

def gradConvergencePlot(obj, perturb=None, energyType=None, fixedVars = []):
    from matplotlib import pyplot as plt
    eps, errors, ignore = gradConvergence(obj, perturb, energyType, fixedVars)
    plt.title('Directional derivative fd test for gradient')
    plt.ylabel('Relative error')
    plt.xlabel('Step size')
    plt.loglog(eps, errors)
    plt.grid()

def hessConvergence(obj, perturb=None, energyType=None, fixedVars = []):
    epsilons = np.logspace(-9, -3, 100)
    errors = []
    if (energyType is None): energyType = obj.EnergyType.Full
    if (perturb is None): perturb = np.random.uniform(-1, 1, size=obj.numVars())
    for eps in epsilons:
        err, fd, an = validateHessian(obj, etype=energyType, perturb=perturb, fd_eps=eps, fixedVars = fixedVars)
        errors.append(err)
    return (epsilons, errors, an)

def hessConvergencePlot(obj, perturb=None, energyType=None, fixedVars = []):
    from matplotlib import pyplot as plt
    eps, errors, ignore = hessConvergence(obj, perturb, energyType, fixedVars)
    plt.title('Directional derivative fd test for Hessian')
    plt.ylabel('Relative error')
    plt.xlabel('Step size')
    plt.loglog(eps, errors)
    plt.grid()

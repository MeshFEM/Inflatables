#ifndef PARAMETRIZATION_NEWTON_HH
#define PARAMETRIZATION_NEWTON_HH

#include <MeshFEM/newton_optimizer/newton_optimizer.hh>

namespace parametrization {

template<class RParam>
ConvergenceReport regularized_parametrization_newton(RParam &rparam, const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts);

}

#endif /* end of include guard: PARAMETRIZATION_NEWTON_HH */

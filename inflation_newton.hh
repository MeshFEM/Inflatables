#ifndef INFLATION_NEWTON_HH
#define INFLATION_NEWTON_HH

#include <MeshFEM/newton_optimizer/newton_optimizer.hh>
#include <memory>
#include <functional>

using CallbackFunction = std::function<void(size_t)>;

template<class ISheet>
std::unique_ptr<NewtonOptimizer> get_inflation_optimizer(ISheet &isheet, const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts = NewtonOptimizerOptions(), CallbackFunction = nullptr);

template<class ISheet>
ConvergenceReport inflation_newton(ISheet &isheet, const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts, CallbackFunction = nullptr);

#endif /* end of include guard: INFLATION_NEWTON_HH */

#ifndef FIT_METRIC_NEWTON_HH
#define FIT_METRIC_NEWTON_HH

#include <MeshFEM/newton_optimizer/newton_optimizer.hh>
#include "MetricFitter.hh"

using CallbackFunction = std::function<void(size_t)>;

ConvergenceReport fit_metric_newton(MetricFitter &mfit, const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts, CallbackFunction = nullptr);

#endif /* end of include guard: FIT_METRIC_NEWTON_HH */

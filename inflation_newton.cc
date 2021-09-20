#include "inflation_newton.hh"
#include <memory>

template<class ISheet>
struct InflationNewtonProblem : public NewtonProblem {
    InflationNewtonProblem(ISheet &isheet)
        : m_sheet(isheet), m_hessianSparsity(isheet.hessianSparsityPattern()) { }

    virtual void setVars(const Eigen::VectorXd &vars) override { m_sheet.setVars(vars.cast<typename ISheet::Real>()); }
    virtual const Eigen::VectorXd getVars() const override { return m_sheet.getVars().template cast<double>(); }
    virtual size_t numVars() const override { return m_sheet.numVars(); }

    virtual Real energy() const override { return m_sheet.energy(); }

    virtual Eigen::VectorXd gradient(bool /* freshIterate */ = false) const override {
        auto result = m_sheet.gradient();
        return result.template cast<double>();
    }

    void setCustomIterationCallback(const CallbackFunction &cb) { m_customCallback = cb; }

    virtual SuiteSparseMatrix hessianSparsityPattern() const override { /* m_hessianSparsity.fill(1.0); */ return m_hessianSparsity; }

protected:
    virtual void m_evalHessian(SuiteSparseMatrix &result, bool /* projectionMask */) const override {
        result.setZero();
        m_sheet.hessian(result);
    }
    virtual void m_evalMetric(SuiteSparseMatrix &result) const override {
        // TODO: mass matrix?
        result.setIdentity(true);
    }

    virtual void m_iterationCallback(size_t i) override { if (m_customCallback) m_customCallback(i); }

    CallbackFunction m_customCallback;

    ISheet &m_sheet;
    mutable SuiteSparseMatrix m_hessianSparsity;
};

template<class ISheet>
std::unique_ptr<NewtonOptimizer> get_inflation_optimizer(ISheet &isheet, const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts, CallbackFunction customCallback) {
    auto problem = std::make_unique<InflationNewtonProblem<ISheet>>(isheet);
    problem->addFixedVariables(fixedVars);
    problem->setCustomIterationCallback(customCallback);
    auto opt = std::make_unique<NewtonOptimizer>(std::move(problem));
    opt->options = opts;
    return opt;
}

template<class ISheet>
ConvergenceReport inflation_newton(ISheet &isheet, const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts, CallbackFunction customCallback) {
    return get_inflation_optimizer(isheet, fixedVars, opts, customCallback)->optimize();
}

// Explicit function template instantiations
#include "InflatableSheet.hh"
#include "TargetAttractedInflation.hh"
template ConvergenceReport inflation_newton<InflatableSheet         >(InflatableSheet &,          const std::vector<size_t> &, const NewtonOptimizerOptions &, CallbackFunction);
template ConvergenceReport inflation_newton<TargetAttractedInflation>(TargetAttractedInflation &, const std::vector<size_t> &, const NewtonOptimizerOptions &, CallbackFunction);

// The following shouldn't be necessary, but fix an undefined symbol error when loading the `inflation` Python module
template std::unique_ptr<NewtonOptimizer> get_inflation_optimizer<InflatableSheet         >(InflatableSheet          &, const std::vector<size_t> &, const NewtonOptimizerOptions &, CallbackFunction);
template std::unique_ptr<NewtonOptimizer> get_inflation_optimizer<TargetAttractedInflation>(TargetAttractedInflation &, const std::vector<size_t> &, const NewtonOptimizerOptions &, CallbackFunction);

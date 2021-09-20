#include "fit_metric_newton.hh"
#include <memory>

struct MetricFittingProblem : public NewtonProblem {
    MetricFittingProblem(MetricFitter &mfit)
        : m_mfit(mfit), m_hessianSparsity(mfit.hessianSparsityPattern()) { }

    virtual void setVars(const Eigen::VectorXd &vars) override { m_mfit.setVars(vars); }
    virtual const Eigen::VectorXd getVars() const override { return m_mfit.getVars(); }
    virtual size_t numVars() const override { return m_mfit.numVars(); }

    virtual Real energy() const override { return m_mfit.energy(); }

    virtual Eigen::VectorXd gradient(bool /* freshIterate */ = false) const override {
        auto result = m_mfit.gradient();
        return result;
    }

    virtual SuiteSparseMatrix hessianSparsityPattern() const override { /* m_hessianSparsity.fill(1.0); */ return m_hessianSparsity; }

    void setCustomIterationCallback(const CallbackFunction &cb) { m_customCallback = cb; }

protected:
    virtual void m_evalHessian(SuiteSparseMatrix &result, bool /* projectionMask */) const override {
        result.setZero();
        m_mfit.hessian(result);
    }
    virtual void m_evalMetric(SuiteSparseMatrix &result) const override {
        // TODO: mass matrix?
        result.setIdentity(true);
    }

    virtual void m_iterationCallback(size_t i) override { if (m_customCallback) m_customCallback(i); }

    CallbackFunction m_customCallback;

    MetricFitter &m_mfit;
    mutable SuiteSparseMatrix m_hessianSparsity;
};

ConvergenceReport fit_metric_newton(MetricFitter &mfit, const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts, CallbackFunction customCallback) {
    auto problem = std::make_unique<MetricFittingProblem>(mfit);
    problem->addFixedVariables(fixedVars);
    problem->setCustomIterationCallback(customCallback);
    NewtonOptimizer opt(std::move(problem));
    opt.options = opts;
    return opt.optimize();
}

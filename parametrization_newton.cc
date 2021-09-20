#include "parametrization_newton.hh"
#include <memory>

#include "parametrization.hh"

namespace parametrization {

void applyBoundConstraints(const RegularizedParametrizer &rparam, std::vector<NewtonProblem::BoundConstraint> &bc) {
    // Set bounds on alpha variables
    bc.reserve(rparam.numAlphaVars());
    const size_t nvar = rparam.numVars();
    const auto &vars = rparam.getVars();
    for (size_t i = rparam.alphaOffset(); i < nvar; ++i) {
        if ((vars[i] < rparam.alphaMin()) || (vars[i] > rparam.alphaMax()))
            throw std::runtime_error("Alpha bound violated " + std::to_string(i - rparam.alphaOffset()));
        bc.emplace_back(i, rparam.alphaMin(), NewtonProblem::BoundConstraint::Type::LOWER);
        bc.emplace_back(i, rparam.alphaMax(), NewtonProblem::BoundConstraint::Type::UPPER);
    }
}

void applyBoundConstraints(const RegularizedParametrizerSVD &/* rparam */, NewtonProblem::BoundConstraint &/* bc */) {
    // No bound constraints...
}

template<typename ParametrizationEnergy>
struct ParametrizationNewtonProblem : public NewtonProblem {
    ParametrizationNewtonProblem(ParametrizationEnergy &energy)
        : m_energy(energy), m_hessianSparsity(energy.hessianSparsityPattern()) { }

    virtual void setVars(const Eigen::VectorXd &vars) override { m_energy.setVars(vars); }
    virtual const Eigen::VectorXd getVars() const override { return m_energy.getVars(); }
    virtual size_t numVars() const override { return m_energy.numVars(); }

    virtual Real energy() const override { return m_energy.energy(); }

    virtual Eigen::VectorXd gradient(bool /* freshIterate */ = false) const override {
        auto result = m_energy.gradient();
        return result;
    }

    virtual SuiteSparseMatrix hessianSparsityPattern() const override { /* m_hessianSparsity.fill(1.0); */ return m_hessianSparsity; }

protected:
    virtual void m_evalHessian(SuiteSparseMatrix &result, bool projectionMask) const override {
        result.setZero();
        m_energy.hessian(result, projectionMask);
    }
    virtual void m_evalMetric(SuiteSparseMatrix &result) const override {
        // TODO: mass matrix?
        result.setIdentity(true);
    }

    ParametrizationEnergy &m_energy;
    mutable SuiteSparseMatrix m_hessianSparsity;
};

template<class RParam>
ConvergenceReport regularized_parametrization_newton(RParam &rparam, const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts) {
    auto problem = std::make_unique<ParametrizationNewtonProblem<RParam>>(rparam);
    problem->addFixedVariables(fixedVars);
    NewtonOptimizer opt(std::move(problem));
    opt.options = opts;
    return opt.optimize();
}

////////////////////////////////////////////////////////////////////////////////
// Explicit instantiations
////////////////////////////////////////////////////////////////////////////////
template ConvergenceReport regularized_parametrization_newton<RegularizedParametrizer   >(RegularizedParametrizer    &rparam, const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts);
template ConvergenceReport regularized_parametrization_newton<RegularizedParametrizerSVD>(RegularizedParametrizerSVD &rparam, const std::vector<size_t> &fixedVars, const NewtonOptimizerOptions &opts);

}

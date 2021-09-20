#include <catch2/catch.hpp>
#include "../TensionFieldEnergy.hh"
#include <MeshFEM/EnergyDensities/NeoHookeanEnergy.hh>
#include <MeshFEM/EnergyDensities/TensionFieldTheory.hh>

using INeo_TFT = EnergyDensityFBasedFromCBased<RelaxedEnergyDensity<IncompressibleNeoHookeanEnergyCBased<double>>, 3>;
using M32d = Eigen::Matrix<double, 3, 2>;
using M2d  = Eigen::Matrix<double, 2, 2>;
using OTFE = EnergyDensityFBasedFromCBased<OptionalTensionFieldEnergy<double>, 3>;

template<class A, class B>
void requireApproxEqual(const A &a, const B &b) {
    for (int i = 0; i < a.rows(); i++) {
        for (int j = 0; j < a.cols(); j++) {
            REQUIRE(a(i, j) == Approx(b(i, j)));
        }
    }
}

template<class A, class B>
bool approxEqual(const A &a, const B &b, const double compare_eps = 2.5e-6) {
    return ((a - b).squaredNorm() < compare_eps * compare_eps * b.squaredNorm()) ||
           ((a - b).squaredNorm() < compare_eps);
}

void runComparison(INeo_TFT &psi_new, OTFE &psi_old) {
    for (size_t i = 0; i < 1000; ++i) {
        M32d F = M32d::Identity() + 1e-3 * M32d::Random();
        psi_new.setDeformationGradient(F);
        psi_old.setDeformationGradient(F);

        REQUIRE(psi_new.energy() == Approx(psi_old.energy()));
        REQUIRE(approxEqual(psi_new.denergy(), psi_old.denergy()));

        for (size_t j = 0; j < 100; ++j) {
            M32d dF = M32d::Random();
            // M2d dC = dF.transpose() * F + F.transpose() * dF;
            REQUIRE(approxEqual(psi_new.delta_denergy(dF), psi_old.delta_denergy(dF)));
        }
    }
}

TEST_CASE("tft_comparison", "[tft_comparison]") {
    INeo_TFT psi_new;
    OTFE     psi_old;
    psi_new.psi().stiffness = 1.0;
    psi_old.setStiffness(1.0);

    psi_new.setRelaxationEnabled(false);
    psi_old.useTensionField = false;
    runComparison(psi_new, psi_old);

    psi_new.setRelaxationEnabled(true);
    psi_old.useTensionField = true;
    runComparison(psi_new, psi_old);
}

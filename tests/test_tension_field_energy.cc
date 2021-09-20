#include <iostream>
#include "../TensionFieldEnergy.hh"

using OTFE = OptionalTensionFieldEnergy<double>;
using M2d  = OTFE::M2d;
using V2d  = OTFE::V2d;

void runTest(const M2d &C, double fd_eps) {
    OTFE tfe(C);

    auto energyAt = [&](const M2d &X) {
        return OTFE(X).energy();
    };

    auto dEnergyAt = [&](const M2d &X, const M2d &perturb) {
        return OTFE(X).denergy(perturb);
    };

    std::cout.precision(16);

    ////////////////////////////////////////////////////////////////////////////
    // First derivative tests
    ////////////////////////////////////////////////////////////////////////////
    M2d perturb(M2d::Random());
    perturb(0, 1) = perturb(1, 0);
    double fd_delta_energy = (energyAt(C + fd_eps * perturb) - energyAt(C - fd_eps * perturb)) / (2 * fd_eps);
    double an_delta_energy = tfe.denergy(perturb);

    std::cout.precision(16);
    std::cout << "fd_delta_energy:\t" << fd_delta_energy << std::endl;
    std::cout << "an_delta_energy:\t" << an_delta_energy << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    // Second derivative tests
    ////////////////////////////////////////////////////////////////////////////
    M2d fd_delta_grad_energy,
        an_delta_grad_energy;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            M2d ei_otimes_ej(M2d::Zero());
            if (i == j) ei_otimes_ej(i, j) = 1.0;
            else        ei_otimes_ej(i, j) = ei_otimes_ej(j, i) = 0.5;

            an_delta_grad_energy(i, j) = tfe.d2energy(ei_otimes_ej, perturb);
            fd_delta_grad_energy(i, j) = (dEnergyAt(C + fd_eps * perturb, ei_otimes_ej) - dEnergyAt(C - fd_eps * perturb, ei_otimes_ej)) / (2 * fd_eps);
        }
    }

    std::cout << std::endl;
    std::cout << "fd_delta_grad_energy" << std::endl << fd_delta_grad_energy << std::endl;
    std::cout << "an_delta_grad_energy" << std::endl << an_delta_grad_energy << std::endl;
}

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cout << "usage: ./test_tension_field_energy fd_eps" << std::endl;
        exit(-1);
    }
    const double fd_eps = std::stod(argv[1]);

    std::cout << "Full case" << std::endl;
    M2d C;
    C << 3.0, 0.0,
         0.0, 2.0;
    runTest(C, fd_eps);

    std::cout << std::endl;
    std::cout << "Relaxed case" << std::endl;
    C << 3.0, 0.0,
         0.0, 0.4;
    runTest(C, fd_eps);

    std::cout << std::endl;
    std::cout << "Perturbed relaxed case" << std::endl;
    C << 3.0, 0.0,
         0.0, 0.4;
    C += 0.1 * M2d::Random();
    C(1, 0) = C(0, 1);
    runTest(C, fd_eps);

    std::cout << std::endl;
    std::cout << "Test case from simulation" << std::endl;
    C << 1.00110208e+00, 5.49990066e-04,
         5.49990066e-04, 9.99143867e-01;
    runTest(C, fd_eps);

    return 0;
}

#include <iostream>
#include "../IncompressibleBalloonEnergy.hh"

using IBE = IncompressibleBalloonEnergy<double>;
using M2d  = IBE::M2d;
using V2d  = IBE::V2d;
using M32d = Eigen::Matrix<double, 3, 2>;

void runTest(const M2d &C, double fd_eps) {
    IBE balloon(C);

    auto energyAt = [&](const M2d &X) {
        return IBE(X).energy();
    };

    auto dEnergyAt = [&](const M2d &X, const M2d &perturb) {
        return IBE(X).denergy(perturb);
    };

    std::cout.precision(16);

    ////////////////////////////////////////////////////////////////////////////
    // First derivative tests
    ////////////////////////////////////////////////////////////////////////////
    M2d perturb(M2d::Random());
    perturb(0, 1) = perturb(1, 0);
    double fd_delta_energy = (energyAt(C + fd_eps * perturb) - energyAt(C - fd_eps * perturb)) / (2 * fd_eps);
    double an_delta_energy = balloon.denergy(perturb);

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

            an_delta_grad_energy(i, j) = balloon.d2energy(ei_otimes_ej, perturb);

            fd_delta_grad_energy(i, j) = (dEnergyAt(C + fd_eps * perturb, ei_otimes_ej) - dEnergyAt(C - fd_eps * perturb, ei_otimes_ej)) / (2 * fd_eps);
        }
    }

    std::cout << std::endl;
    std::cout << "fd_delta_grad_energy" << std::endl << fd_delta_grad_energy << std::endl;
    std::cout << "an_delta_grad_energy" << std::endl << an_delta_grad_energy << std::endl;
}

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cout << "usage: ./test_balloon_energy fd_eps" << std::endl;
        exit(-1);
    }
    const double fd_eps = std::stod(argv[1]);

    M2d C;
    C.setRandom();
    C(1, 0) = C(0, 1);

    std::cout << "Around random C:" << std::endl;
    runTest(C, fd_eps);
    std::cout << std::endl;

    std::cout << "Around identity:" << std::endl;
    C.setIdentity();
    runTest(C, fd_eps);

    M32d J;
    J << 1, 0,
         0, 1,
         0, 0;
    J.setRandom();

    M32d J_perturb_a, J_perturb_b;
    J_perturb_a.setRandom(); J_perturb_b.setRandom();

    auto evalAtJ = [&](const M32d &Jeval) {
        C = Jeval.transpose() * Jeval;
        return IBE(C).energy();
    };

    auto evalGradAtJ = [&](const M32d &Jeval, const M32d &perturb) {
        C = Jeval.transpose() * Jeval;
        M2d dC = 2 * Jeval.transpose() * perturb; // technically should be symmetrized, but denergy double contracts it with a symmetric tensor...
        return IBE(C).denergy(dC);
    };

    auto evalHessAtJ = [&](const M32d &Jeval, const M32d &perturb_a, const M32d &perturb_b) {
        C = Jeval.transpose() * Jeval;
        M2d dC_a = 2.0 * Jeval.transpose() * perturb_a, // pre-symmetrization...
            dC_b = 2.0 * Jeval.transpose() * perturb_b; 
        symmetrize(dC_a);
        symmetrize(dC_b);
        M2d d2C_ab = (perturb_b.transpose() * perturb_a + perturb_a.transpose() * perturb_b);
        IBE ibe(C);
        return ibe.d2energy(dC_a, dC_b) + ibe.denergy(d2C_ab);
    };

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << (evalAtJ(J + fd_eps * J_perturb_a) - evalAtJ(J - fd_eps * J_perturb_a)) / (2 * fd_eps) << std::endl;
    std::cout << evalGradAtJ(J, J_perturb_a) << std::endl;

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << (evalGradAtJ(J + fd_eps * J_perturb_b, J_perturb_a) - evalGradAtJ(J - fd_eps * J_perturb_b, J_perturb_a)) / (2 * fd_eps) << std::endl;
    std::cout << evalHessAtJ(J, J_perturb_a, J_perturb_b) << std::endl;

    return 0;
}

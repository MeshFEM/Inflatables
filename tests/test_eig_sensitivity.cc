#include <iostream>
#include "../EigSensitivity.hh"

using ES = EigSensitivity<double>;

using M2d = ES::M2d;
using V2d = ES::V2d;

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cout << "usage: ./test_eig_sensitivity fd_eps" << std::endl;
        exit(-1);
    }
    const double fd_eps = std::stod(argv[1]);

    M2d A;
    A.setRandom();
    A(1, 0) = A(0, 1);

    ES eigA(A);

    auto lambdaAt = [&](const M2d &X) {
        ES eigX(X);
        return eigX.Lambda();
    };

    auto dLambdaAt = [&](const M2d &X, const M2d &perturb) {
        ES eigX(X);
        return eigX.dLambda(perturb);
    };

    std::cout.precision(16);

    std::cout << "A: " << std::endl << A << std::endl << std::endl;;

    std::cout << "A eigs: " << eigA.Lambda().transpose() << std::endl << std::endl;;
    std::cout << "Q(A): " << std::endl << eigA.Q() << std::endl << std::endl;;

    ////////////////////////////////////////////////////////////////////////////
    // First derivative tests
    ////////////////////////////////////////////////////////////////////////////
    M2d perturb(M2d::Random());
    perturb(0, 1) = perturb(1, 0);
    V2d fd_delta_lambda = (lambdaAt(A + fd_eps * perturb) - lambdaAt(A - fd_eps * perturb)) / (2 * fd_eps);
    V2d an_delta_lambda = eigA.dLambda(perturb);

    std::cout.precision(16);
    std::cout << "fd_delta_lambda:\t" << fd_delta_lambda.transpose() << std::endl;
    std::cout << "an_delta_lambda:\t" << an_delta_lambda.transpose() << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    // Second derivative tests
    ////////////////////////////////////////////////////////////////////////////
    M2d fd_delta_grad_lambda_0, fd_delta_grad_lambda_1,
        an_delta_grad_lambda_0, an_delta_grad_lambda_1;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            M2d ei_otimes_ej(M2d::Zero());
            if (i == j) ei_otimes_ej(i, j) = 1.0;
            else        ei_otimes_ej(i, j) = ei_otimes_ej(j, i) = 0.5;

            V2d d2Lambda = eigA.d2Lambda(ei_otimes_ej, perturb);
            an_delta_grad_lambda_0(i, j) = d2Lambda[0];
            an_delta_grad_lambda_1(i, j) = d2Lambda[1];

            V2d fd_d2Lambda = (dLambdaAt(A + fd_eps * perturb, ei_otimes_ej) - dLambdaAt(A - fd_eps * perturb, ei_otimes_ej)) / (2 * fd_eps);
            fd_delta_grad_lambda_0(i, j) = fd_d2Lambda[0];
            fd_delta_grad_lambda_1(i, j) = fd_d2Lambda[1];
        }
    }

    std::cout << std::endl;
    std::cout << "fd_delta_grad_lambda_0" << std::endl << fd_delta_grad_lambda_0 << std::endl;
    std::cout << "an_delta_grad_lambda_0" << std::endl << an_delta_grad_lambda_0 << std::endl;

    std::cout << std::endl;
    std::cout << "fd_delta_grad_lambda_1" << std::endl << fd_delta_grad_lambda_1 << std::endl;
    std::cout << "an_delta_grad_lambda_1" << std::endl << an_delta_grad_lambda_1 << std::endl;

    return 0;
}

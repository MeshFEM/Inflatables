#include <catch2/catch.hpp>

#include <iostream>
#include "../SVDSensitivity.hh"

using M2d = Eigen::Matrix2d;
using V2d = Eigen::Vector2d;
constexpr double fd_eps = 1e-5;
constexpr double compare_eps = 1e-6;

template<class A, class B>
bool approxEqual(const A &a, const B &b) {
    return (a - b).squaredNorm() < compare_eps * compare_eps * b.squaredNorm();
}

void firstDerivativeTests() {
    M2d A(M2d::Random()), perturb(M2d::Random());
    SVDSensitivity svdA(A);
    auto sigmaAt = [&](const M2d &X) {
        SVDSensitivity svdX(X);
        return svdX.Sigma();
    };

    auto uAt = [&](size_t i, const M2d &X) -> V2d {
        SVDSensitivity svdX(X);
        return svdX.u(i);
    };

    auto vAt = [&](size_t i, const M2d &X) -> V2d {
        SVDSensitivity svdX(X);
        return svdX.v(i);
    };

    V2d fd_delta_sigma = (sigmaAt(A + fd_eps * perturb) - sigmaAt(A - fd_eps * perturb)) / (2 * fd_eps);
    V2d an_delta_sigma = svdA.dSigma(perturb);

    V2d fd_delta_u0 = (uAt(0, A + fd_eps * perturb) - uAt(0, A - fd_eps * perturb)) / (2 * fd_eps);
    V2d an_delta_u0 = svdA.du0(perturb);

    V2d fd_delta_u1 = (uAt(1, A + fd_eps * perturb) - uAt(1, A - fd_eps * perturb)) / (2 * fd_eps);
    V2d an_delta_u1 = svdA.du1(perturb);

    V2d fd_delta_v0 = (vAt(0, A + fd_eps * perturb) - vAt(0, A - fd_eps * perturb)) / (2 * fd_eps);
    V2d an_delta_v0 = svdA.dv0(perturb);

    V2d fd_delta_v1 = (vAt(1, A + fd_eps * perturb) - vAt(1, A - fd_eps * perturb)) / (2 * fd_eps);
    V2d an_delta_v1 = svdA.dv1(perturb);

    if (!approxEqual(fd_delta_u0, an_delta_u0)) {
        std::cout << perturb << std::endl << std::endl;
        std::cout << A << std::endl << std::endl;
        std::cout << uAt(0, A + fd_eps * perturb) << std::endl;
        std::cout << uAt(0, A - fd_eps * perturb) << std::endl;
    }

    std::cout.precision(16);
    REQUIRE(approxEqual(fd_delta_sigma, an_delta_sigma));
    REQUIRE(approxEqual(fd_delta_u0, an_delta_u0));
    REQUIRE(approxEqual(fd_delta_u1, an_delta_u1));
    REQUIRE(approxEqual(fd_delta_v0, an_delta_v0));
    REQUIRE(approxEqual(fd_delta_v1, an_delta_v1));
    REQUIRE(approxEqual(fd_delta_v1, an_delta_v1));
}

void secondDerivativeTests() {
    M2d A(M2d::Random()), perturb(M2d::Random());
    SVDSensitivity svdA(A);
    auto gradSigmaAt = [&](size_t i, const M2d &X) {
        SVDSensitivity svdX(X);
        return svdX.dsigma(i);
    };

    auto gradU0At = [&](size_t i, const M2d &X) {
        SVDSensitivity svdX(X);
        return svdX.du0(i);
    };

    auto gradV1At = [&](size_t i, const M2d &X) {
        SVDSensitivity svdX(X);
        return svdX.dv1(i);
    };

    M2d fd_delta_grad_sigma_0 = (gradSigmaAt(0, A + fd_eps * perturb) - gradSigmaAt(0, A - fd_eps * perturb)) / (2 * fd_eps);
    M2d fd_delta_grad_sigma_1 = (gradSigmaAt(1, A + fd_eps * perturb) - gradSigmaAt(1, A - fd_eps * perturb)) / (2 * fd_eps);

    M2d fd_delta_grad_u0_0     = (gradU0At   (0, A + fd_eps * perturb) - gradU0At   (0, A - fd_eps * perturb)) / (2 * fd_eps);
    M2d fd_delta_grad_u0_1     = (gradU0At   (1, A + fd_eps * perturb) - gradU0At   (1, A - fd_eps * perturb)) / (2 * fd_eps);

    M2d fd_delta_grad_v1_0     = (gradV1At   (0, A + fd_eps * perturb) - gradV1At   (0, A - fd_eps * perturb)) / (2 * fd_eps);
    M2d fd_delta_grad_v1_1     = (gradV1At   (1, A + fd_eps * perturb) - gradV1At   (1, A - fd_eps * perturb)) / (2 * fd_eps);

    M2d an_delta_grad_sigma_0, an_delta_grad_sigma_1;
    M2d an_delta_grad_u0_0,    an_delta_grad_u0_1;
    M2d an_delta_grad_v1_0,    an_delta_grad_v1_1;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            M2d ei_otimes_ej(M2d::Zero());
            ei_otimes_ej(i, j) = 1.0;
            auto d2Sigma = svdA.d2Sigma(ei_otimes_ej, perturb);
            an_delta_grad_sigma_0(i, j) = d2Sigma[0];
            an_delta_grad_sigma_1(i, j) = d2Sigma[1];

            auto d2u0 = svdA.d2u0(ei_otimes_ej, perturb);
            an_delta_grad_u0_0(i, j) = d2u0[0];
            an_delta_grad_u0_1(i, j) = d2u0[1];

            auto d2v1 = svdA.d2v1(ei_otimes_ej, perturb);
            an_delta_grad_v1_0(i, j) = d2v1[0];
            an_delta_grad_v1_1(i, j) = d2v1[1];
        }
    }

    // Note: d2sigma_0/(d_A d_B) and d2sigma_1/(d_A d_B) are closely related
    // (related by pi/2 rotation transformation)
    REQUIRE(approxEqual(fd_delta_grad_sigma_0, an_delta_grad_sigma_0));
    REQUIRE(approxEqual(fd_delta_grad_sigma_1, an_delta_grad_sigma_1));
    REQUIRE(approxEqual(fd_delta_grad_u0_0, an_delta_grad_u0_0));
    REQUIRE(approxEqual(fd_delta_grad_u0_1, an_delta_grad_u0_1));
    REQUIRE(approxEqual(fd_delta_grad_v1_0, an_delta_grad_v1_0));
    REQUIRE(approxEqual(fd_delta_grad_v1_1, an_delta_grad_v1_1));
}

TEST_CASE("svd_sensitivity", "[svd_sensitivity]") {
    for (size_t i = 0; i < 1000; ++i) {
        firstDerivativeTests();
        secondDerivativeTests();
    }
}

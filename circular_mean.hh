#ifndef CIRCULAR_MEAN_HH
#define CIRCULAR_MEAN_HH
#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>

// Measure the smallest signed arclength between two angles plotted on a unit circle.
template<typename R_>
R_ circularDistance(R_ a, R_ b) {
    R_ result = std::fmod(a - b, 2 * M_PI);
    if (result >  M_PI) result -= 2 * M_PI;
    if (result < -M_PI) result += 2 * M_PI;
    return result;
}

template<typename R_, typename AngleCollection>
R_ sumSquaredCircularDist(R_ x, const AngleCollection &angles) {
    R_ result = 0.0;
    for (R_ theta : angles) {
        R_ d = circularDistance(theta, x);
        result += d * d;
    }
    return result;
}

// Find the "circular mean" of a sequence of N angles "theta_i" by finding the angle "x"
// that minimizes the total squared circular distance to each angle:
//      min_{x, k_i} 1/2 (theta_i + 2 * pi * k_i - x)^2
// where k_i are integers chosen to ensure ensure the smaller of the two
// possible arcs is used to measure the angular distance (so dist is always in
// [-pi, pi]). Knowing the optimal k_i, we can easily solve for x:
//      N * x = (sum_i theta_i) + 2 * pi k
//  where integer "k" is the sum of the optimal k_i.
//  Since we really only care about the value of x mod 2 * pi, there are only N distinct
//  possible values for k ({0, ..., N - 1}). We simply check each value and pick
//  one that yields the minimal distance.
//  Note: the following implementation is O(N^2), but it is not too difficult to make
//  it O(N) by avoiding the full recalculation of the distance objective for each k.
template<typename AngleCollection>
auto circularMean(const AngleCollection &angles) -> std::remove_cv_t<std::remove_reference_t<decltype(angles[0])>> {
    using R_ = std::remove_cv_t<std::remove_reference_t<decltype(angles[0])>>;
    size_t N = angles.size();
    R_ meanAngle = std::accumulate(angles.begin(), angles.end(), 0.0) / N;

    R_ minDist = std::numeric_limits<R_>::max();
    R_ optimalAngle = 0;
    for (size_t n = 0; n < N; ++n) {
        R_ candidate = meanAngle + (2 * M_PI * n) / N;
        R_ candidateDist = sumSquaredCircularDist(candidate, angles);
        if (candidateDist < minDist) {
            minDist = candidateDist;
            optimalAngle = candidate;
        }

    }
    return std::fmod(optimalAngle, 2 * M_PI);
}

#endif /* end of include guard: CIRCULAR_MEAN_HH */

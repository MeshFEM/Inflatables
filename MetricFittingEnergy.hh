////////////////////////////////////////////////////////////////////////////////
// MetricFittingEnergy.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Sheet material energy that is a function of the **deformation gradient**.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  05/30/2019 17:15:18
////////////////////////////////////////////////////////////////////////////////
#ifndef METRICFITTINGENERGY_HH
#define METRICFITTINGENERGY_HH

#include <Eigen/Dense>
#include <MeshFEM/EnergyDensities/Tensor.hh>

struct MetricFittingEnergy {
    using M2d = Eigen::Matrix2d;

    M2d targetMetric;

    void setMatrix(Eigen::Ref<const M2d> C) {
        m_C = C;
    }

    double energy() const {
        return 0.5 * (m_C - targetMetric).squaredNorm();
    }

    M2d denergy() const { return m_C - targetMetric; }

    auto delta_denergy(Eigen::Ref<const M2d> dC) const { return dC; } // 4th order identity tensor

    M2d currMetric() const {
        return m_C;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    M2d m_C, m_diff;
};

#endif /* end of include guard: METRICFITTINGENERGY_HH */

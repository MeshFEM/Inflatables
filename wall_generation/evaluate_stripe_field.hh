////////////////////////////////////////////////////////////////////////////////
// evaluate_stripe_field.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Sampling of the stripe pattern on a subdivided mesh.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  04/27/2019 15:36:32
////////////////////////////////////////////////////////////////////////////////
#ifndef EVALUATE_STRIPE_FIELD_HH
#define EVALUATE_STRIPE_FIELD_HH

#include <MeshFEM/MeshIO.hh>
#include <Eigen/Dense>

void evaluate_stripe_field(const Eigen::MatrixX3d &vertices,
                           const Eigen::MatrixX3i &elements,
                           const std::vector<double> &stretchAngles,
                           const std::vector<double> &wallWidths,
                           const double frequency,
                           Eigen::MatrixX3d &outVerticesEigen,
                           Eigen::MatrixX3i &outTrianglesEigen,
                           std::vector<double> &stripeField,
                           const size_t nsubdiv = 3,
                           const bool glue = true);

#endif /* end of include guard: EVALUATE_STRIPE_FIELD_HH */

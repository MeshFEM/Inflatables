////////////////////////////////////////////////////////////////////////////////
// extract_contours.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Contour extraction routine to generate wall boundaries.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  04/28/2019 19:46:10
////////////////////////////////////////////////////////////////////////////////
#ifndef EXTRACT_CONTOURS_HH
#define EXTRACT_CONTOURS_HH

#include <MeshFEM/MeshIO.hh>
#include <vector>

void extract_contours(const std::vector<MeshIO::IOVertex > &vertices,
                      const std::vector<MeshIO::IOElement> &triangles,
                      const std::vector<double> &sdf,
                      const double targetEdgeSpacing,
                      const double minContourLen, // length below which contours are removed
                      Eigen::MatrixX3d                       &outVertices,
                      std::vector<std::pair<size_t, size_t>> &outEdges);

#endif /* end of include guard: EXTRACT_CONTOURS_HH */

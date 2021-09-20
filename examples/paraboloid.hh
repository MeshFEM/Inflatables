////////////////////////////////////////////////////////////////////////////////
// paraboloid.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Generate a triangle mesh of a paraboloid quadric surface with specified
//  principal curvatures at the origin.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  01/12/2018 18:05:49
////////////////////////////////////////////////////////////////////////////////
#ifndef PARABOLOID_HH
#define PARABOLOID_HH
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/filters/gen_grid.hh>
#include <MeshFEM/Triangulate.h>
#include <MeshFEM/filters/quad_tri_subdiv.hh>

#include <vector>

inline
void paraboloid(double triArea, double k1, double k2,
                std::vector<MeshIO::IOVertex> &vertices,
                std::vector<MeshIO::IOElement> &elements,
                bool delaunay = true)
{
    if (!delaunay) {
        std::vector<MeshIO::IOVertex > gridVertices;
        std::vector<MeshIO::IOElement> gridElements;
        size_t nx = 64, ny = 64;
        gen_grid({nx, ny}, gridVertices, gridElements);

        std::vector<size_t> quadIdx;
        quad_tri_subdiv(gridVertices, gridElements, vertices, elements, quadIdx);
        for (auto &v : vertices)  {
            v[0] = 2.0 * v[0] / nx - 1.0;
            v[1] = 2.0 * v[1] / ny - 1.0;
        }
    }
    else {
        std::vector<Point2D> pts = { {-1, -1}, {1, -1}, {1, 1}, {-1, 1} };
        std::vector<std::pair<size_t, size_t>> edges = { {0, 1}, {1, 2}, {2, 3}, {3, 0} };
        triangulatePSLG(pts, edges, std::vector<Point2D>(), vertices, elements, triArea, "Dq32");
    }

    // plot 1/2 (k1 x^2 + k2 y^2) over [-1, 1]^2
    for (auto &v : vertices)
        v[2] = 0.5 * (k1 * v[0] * v[0] + k2 * v[1] * v[1]);
}

#endif /* end of include guard: PARABOLOID_HH */

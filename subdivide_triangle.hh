#ifndef SUBDIVIDE_TRIANGLE_HH
#define SUBDIVIDE_TRIANGLE_HH

#include <map>
#include <tuple>
#include <algorithm>

using Pt = std::tuple<double, double, double>;
using PointGluingMap = std::map<Pt, size_t>;

// Example for nsubdiv == 2
// +                  +
// | .                | .
// |  .               +--.
// |   .              |\ |.
// |     .    ==>   ^ +--+--.
// |       .        | |\ |\ | .
// +--+--+--+      i| +--+--+--+
//                    --> j
// Grid index (i, j) ==> barycentric coordinates
//      ((nsubdiv + 1) - (i + j), j, i) / (nsubdiv + 1)
// "newPt"  is called for each new point insertion (called only once for edge/corner points);
//          It must do the actual point generation and return a unique index for the inserted point.
// "newTri" is called for each newly triangle; it must do the actual triangle generation.
template<typename Point3D_, typename NewPtCallback, typename NewTriCallback>
void subdivide_triangle(int nsubdiv, const Point3D_ &p0, const Point3D_ &p1, const Point3D_ &p2, PointGluingMap &indexForPoint,
                        const NewPtCallback &newPt, const NewTriCallback &newTri) {
    using Real_ = typename Point3D_::Scalar;
    auto gridVertex = [&](int i, int j) {
        // Note: when we evaluate the point with the Eigen template expression
        //      (((nsubdiv + 1) - i - j) * p0 + j * p1 + i * p2) / (nsubdiv + 1)
        // and compile with vectorization--at least on the Intel compiler--we
        // occasionally get different floating point values for the coordinates
        // on each halfedge of an edge. This is presumably due to the use of a
        // MAC instruction.
        // We guarantee consitent values by sorting the input points lexicographically
        // so that the multiply-accumulates always happen in the same order.
        std::array<int, 3> ilam{{(nsubdiv + 1) - i - j, j, i}};
        Point3D_ p;

        {
            std::array<size_t,           3> idx {{  0,   1,   2}};
            std::array<const Point3D_ *, 3> pts {{&p0, &p1, &p2}};
            std::sort(idx.begin(), idx.end(), [&](const size_t &i0, const size_t &i1) {
                        return std::lexicographical_compare(pts[i0]->data(), pts[i0]->data() + 3,
                                                            pts[i1]->data(), pts[i1]->data() + 3);
                    });
            p  = ilam[idx[0]] * (*pts[idx[0]]);
            p += ilam[idx[1]] * (*pts[idx[1]]);
            p += ilam[idx[2]] * (*pts[idx[2]]);
        }
        p /= (nsubdiv + 1);

        auto key = std::make_tuple(p[0], p[1], p[2]);

        auto it = indexForPoint.lower_bound(key);
        if ((it != indexForPoint.end()) && (it->first == key)) return it->second;

        Real_ lambda_1 = j  / (nsubdiv + 1.0),
              lambda_2 = i  / (nsubdiv + 1.0),
              lambda_0 = 1.0 - lambda_1 - lambda_2;
        size_t idx = newPt(p, lambda_0, lambda_1, lambda_2);
        indexForPoint.emplace_hint(it, key, idx);
        return idx;
    };

    // Triangulate the square with lower-lefthand corner (i, j)
    for (int i = 0; i <= nsubdiv; ++i) {         // Loop up to second-to-last vertex
        for (int j = 0; j <= nsubdiv - i; ++j) { // Loop up to second-to-last diagonal: i + j <= nsubdiv
            newTri(gridVertex(i    , j    ),
                   gridVertex(i    , j + 1),
                   gridVertex(i + 1, j    ));
            if ((i + 1) + (j + 1) > nsubdiv + 1) continue; // Upper subtriangle is out of bounds; discard
            newTri(gridVertex(i + 1, j    ),
                   gridVertex(i    , j + 1),
                   gridVertex(i + 1, j + 1));
        }
    }
}

#endif /* end of include guard: SUBDIVIDE_TRIANGLE_HH */

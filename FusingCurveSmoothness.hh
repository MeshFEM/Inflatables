////////////////////////////////////////////////////////////////////////////////
// FusingCurveSmoothness.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  A configurable smoothness energy for the fusing curve rest shapes.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  09/08/2020 21:50:48
////////////////////////////////////////////////////////////////////////////////
#ifndef FUSINGCURVESMOOTHNESS_HH
#define FUSINGCURVESMOOTHNESS_HH
#include <stdexcept>

#include "InflatableSheet.hh"
#include "Nondimensionalization.hh"

struct FusingCurveSmoothness {
    using Mesh = InflatableSheet::Mesh;
    using Real = InflatableSheet::Real;
    using MX2d = InflatableSheet::MX2d;
    using VXd  = InflatableSheet::VXd;
    using V3d  = InflatableSheet::V3d;
    using V2d  = InflatableSheet::V2d;
    using IdxPolyline = std::vector<size_t>;
    using IdxEdge     = std::array<size_t, 2>;
    using VtxNbhd     = std::array<size_t, 3>; // prev, curr, next

    FusingCurveSmoothness(const InflatableSheet &s) { std::tie(boundaryLoops, wallCurves) = s.getFusingPolylines(); }

    Real energy(const Mesh &currMesh, const Mesh &origMesh, const Nondimensionalization &n) const {
        Real result = 0.0;
        visitCurveEdges([&](const IdxEdge &e, Real weight) {
            std::array<V3d, 2> u;
            for (size_t i = 0; i < 2; ++i) u[i] = currMesh.node(e[i])->p - origMesh.node(e[i])->p;
            const Real l_ij = (origMesh.node(e[1])->p - origMesh.node(e[0])->p).norm();

            // Displacement field's 1D Dirichlet energy
            // (Note: this scales differently from the vertex-based terms)
            result += 0.5 * (weight * dirichletWeight * n.dirichletSmoothingScale() / l_ij) * (u[1] - u[0]).squaredNorm();
        });

        visitCurveInteriorVertices([&](const VtxNbhd &v, Real weight) {
            // The Laplacian, length scale Dirichlet, and curvature
            // regularization energies all have the same scaling behavior.
            weight *= n.smoothingScale();

            std::array<V2d, 3> u;
            std::array<V2d, 3> pcurr, porig;

            for (size_t i = 0; i < 3; ++i) {
                pcurr[i] = truncateFrom3D<V2d>(currMesh.node(v[i])->p);
                porig[i] = truncateFrom3D<V2d>(origMesh.node(v[i])->p);
                u[i] = pcurr[i] - porig[i];
            }

            // 0  a  1  b  2
            // o-----o-----o
            //    |_____|
            //       h
            const Real h_a     = (pcurr[1] - pcurr[0]).norm();
            const Real h_b     = (pcurr[2] - pcurr[1]).norm();
            const Real dbl_h   = h_a + h_b;
            const Real h_a_0   = (porig[1] - porig[0]).norm();
            const Real h_b_0   = (porig[2] - porig[1]).norm();
            const Real dbl_h_0 = h_a_0 + h_b_0;

            // Laplacian energy 0.5 * int (d^2u/ds^2)^2 ds
            result += (laplacianWeight * weight / dbl_h_0) * ((1 / h_a_0 + 1 / h_b_0) * u[1] - u[0] / h_a_0 - u[2] / h_b_0).squaredNorm();

            // Length scale Dirichlet energy (prefer uniform scaling of edge lengths)
            const Real scale_a = h_a / h_a_0;
            const Real scale_b = h_b / h_b_0;
            result += (lengthScaleSmoothingWeight * weight / dbl_h_0) * std::pow(scale_a - scale_b, 2);

            // Curvature energy:
            //      0.5 int max(|kappa| - |kappa_0| - eps, 0.0)^2 ds ~= 0.5 sum_i (theta / h - theta_0 / h_0 - eps)_^2 h_0
            //                                                        = 1 / 2 sum_i (h_0 / h theta - theta_0 - h_0 eps)_^2 / h_0
            // where eps is the curvatureSmoothingActivationThreshold, and
            // theta is the *unsigned* turning angle between the previous/next
            // edges (in [0, pi])
            const Real theta_0 = angle(porig[1] - porig[0],
                                       porig[2] - porig[1]);
            const Real theta   = angle(pcurr[1] - pcurr[0],
                                       pcurr[2] - pcurr[1]);
            result += (curvatureWeight * weight / dbl_h_0)
                      * std::pow(std::max(dbl_h_0 / dbl_h * theta - theta_0 - 0.5 * dbl_h_0 * curvatureSmoothingActivationThreshold, 0.0), 2.0);
        });
        return result;
    }

    InflatableSheet::MX2d gradient(const Mesh &currMesh, const Mesh &origMesh, const Nondimensionalization &n) const {
        InflatableSheet::MX2d g;
        g.setZero(currMesh.numVertices(), 2);
        accumulateGradient(g, currMesh, origMesh, n);
        return g;
    }

    void accumulateGradient(Eigen::Ref<MX2d> gradRestPositions, const Mesh &currMesh, const Mesh &origMesh, const Nondimensionalization &n) const {
        visitCurveEdges([&](IdxEdge e, Real weight) {
            std::array<V3d, 2> u;
            for (size_t i = 0; i < 2; ++i) u[i] = (currMesh.node(e[i])->p - origMesh.node(e[i])->p);
            const Real l_ij = (origMesh.node(e[1])->p - origMesh.node(e[0])->p).norm();

            // Displacement field's 1D Dirichlet energy
            V3d contrib = (weight * dirichletWeight * n.dirichletSmoothingScale() / l_ij) * (u[1] - u[0]);
            gradRestPositions.row(e[0]) -= truncateFrom3D<V2d>(contrib);
            gradRestPositions.row(e[1]) += truncateFrom3D<V2d>(contrib);
        });

        visitCurveInteriorVertices([&](const VtxNbhd &v, Real weight) {
            // The Laplacian, length scale Dirichlet, and curvature
            // regularization energies all have the same scaling behavior.
            weight *= n.smoothingScale();

            std::array<V2d, 3> u;
            std::array<V2d, 3> pcurr, porig;
            for (size_t i = 0; i < 3; ++i) {
                pcurr[i] = truncateFrom3D<V2d>(currMesh.node(v[i])->p);
                porig[i] = truncateFrom3D<V2d>(origMesh.node(v[i])->p);
                u[i] = pcurr[i] - porig[i];
            }

            // 0  a  1  b  2
            // o-----o-----o
            //    |_____|
            //       h
            V2d e_a = pcurr[1] - pcurr[0],
                e_b = pcurr[2] - pcurr[1];
            const Real h_a     = e_a.norm();
            const Real h_b     = e_b.norm();
            const Real dbl_h   = h_a + h_b;
            const Real h_a_0   = (porig[1] - porig[0]).norm();
            const Real h_b_0   = (porig[2] - porig[1]).norm();
            const Real dbl_h_0 = h_a_0 + h_b_0;

            // Laplacian energy
            V2d laplacianContrib = 2.0 * (laplacianWeight * weight / dbl_h_0) * ((1 / h_a_0 + 1 / h_b_0) * u[1] - u[0] / h_a_0 - u[2] / h_b_0);

            gradRestPositions.row(v[0]) -= (1 / h_a_0) * (laplacianContrib);
            gradRestPositions.row(v[2]) -= (1 / h_b_0) * (laplacianContrib);
            gradRestPositions.row(v[1]) += (1 / h_a_0 + 1 / h_b_0) * (laplacianContrib);

            const Real scale_a = h_a / h_a_0;
            const Real scale_b = h_b / h_b_0;

            // Length scale Dirichlet energy
            V2d d_de_a =  2.0 * (lengthScaleSmoothingWeight * weight / dbl_h_0) * (scale_a - scale_b) / (h_a * h_a_0) * e_a;
            V2d d_de_b = -2.0 * (lengthScaleSmoothingWeight * weight / dbl_h_0) * (scale_a - scale_b) / (h_b * h_b_0) * e_b;

            // Curvature energy
            const Real theta_0 = angle(porig[1] - porig[0],
                                       porig[2] - porig[1]);
            const Real theta_s = signedAngle(e_a, e_b);
            const Real theta   = std::abs(theta_s);

            const Real excess = 2.0 * (curvatureWeight * weight / dbl_h_0) * std::max(dbl_h_0 / dbl_h * theta - theta_0 - 0.5 * dbl_h_0 * curvatureSmoothingActivationThreshold, 0.0);
            if (excess > 0) {
                d_de_a += excess * (-dbl_h_0 * theta / (dbl_h * dbl_h * h_a) * e_a - std::copysign(dbl_h_0 / (dbl_h * h_a * h_a), theta_s) * V2d(-e_a[1], e_a[0]));
                d_de_b += excess * (-dbl_h_0 * theta / (dbl_h * dbl_h * h_b) * e_b + std::copysign(dbl_h_0 / (dbl_h * h_b * h_b), theta_s) * V2d(-e_b[1], e_b[0]));
            }

            gradRestPositions.row(v[0]) -= d_de_a;
            gradRestPositions.row(v[1]) += d_de_a - d_de_b;
            gradRestPositions.row(v[2]) += d_de_b;
        });
    }

    template<class Visitor>
    void visitCurveEdges(const Visitor &visit) const {
        // Boundary loop edges
        for (const IdxPolyline &l : boundaryLoops) {
            for (size_t i = 0; i < l.size() - 1; ++i)
                visit(IdxEdge{{l[i], l[i + 1]}}, boundaryWeight);
        }

        // Interior wall curves
        for (const IdxPolyline &l : wallCurves) {
            for (size_t i = 0; i < l.size() - 1; ++i)
                visit(IdxEdge{{l[i], l[i + 1]}}, interiorWeight);
        }
    }

    template<class Visitor>
    void visitCurveInteriorVertices(const Visitor &visit) const {
         // Boundary loop stencils
        for (const IdxPolyline &l : boundaryLoops) {
            for (size_t i = 1; i < l.size() - 1; ++i)
                visit(VtxNbhd{{l[i - 1], l[i], l[i + 1]}}, boundaryWeight);
            assert(l[0] == l[l.size() - 1]); // All boundary loops should be closed
            visit(VtxNbhd{{l[l.size() - 2], l[0], l[1]}}, boundaryWeight);
        }

        // Interior wall stencils
        for (const IdxPolyline &l : wallCurves) {
            for (size_t i = 1; i < l.size() - 1; ++i)
                visit(VtxNbhd{{l[i - 1], l[i], l[i + 1]}}, interiorWeight);
            // For closed polylines we also need to visit the stencil for the
            // start/endpoint.
            if (l[0] == l[l.size() - 1]) {
                visit(VtxNbhd{{l[l.size() - 2], l[0], l[1]}}, interiorWeight);
            }
        }
    }

    std::vector<IdxPolyline> boundaryLoops, wallCurves;
    Real dirichletWeight = 1.0, laplacianWeight = 0.0, curvatureWeight = 0.0, lengthScaleSmoothingWeight = 0.0;
    Real interiorWeight = 1.0, boundaryWeight = 1.0; // Global scale for the boundary loop and interior fuse curve contributions
    Real curvatureSmoothingActivationThreshold = 0.0;

    ////////////////////////////////////////////////////////////////////////////
    // Serialization + cloning support (for pickling)
    ////////////////////////////////////////////////////////////////////////////
    using State = std::tuple<std::vector<IdxPolyline>, std::vector<IdxPolyline>,
                             Real, Real, Real, Real,
                             Real, Real,
                             Real>;
    static State serialize(const FusingCurveSmoothness &fcs) {
        return std::make_tuple(fcs.boundaryLoops, fcs.wallCurves,
                               fcs.dirichletWeight, fcs.laplacianWeight, fcs.curvatureWeight, fcs.lengthScaleSmoothingWeight,
                               fcs.interiorWeight, fcs.boundaryWeight,
                               fcs.curvatureSmoothingActivationThreshold);
    }
    static std::shared_ptr<FusingCurveSmoothness> deserialize(const State &state) {
        auto fcs = std::shared_ptr<FusingCurveSmoothness>(new FusingCurveSmoothness()); // Need "new" since empty constructor is private...
        fcs->boundaryLoops                         = std::get<0>(state);
        fcs->wallCurves                            = std::get<1>(state);
        fcs->dirichletWeight                       = std::get<2>(state);
        fcs->laplacianWeight                       = std::get<3>(state);
        fcs->curvatureWeight                       = std::get<4>(state);
        fcs->lengthScaleSmoothingWeight            = std::get<5>(state);
        fcs->interiorWeight                        = std::get<6>(state);
        fcs->boundaryWeight                        = std::get<7>(state);
        fcs->curvatureSmoothingActivationThreshold = std::get<8>(state);
        return fcs;
    }
    std::shared_ptr<FusingCurveSmoothness> clone() const { return deserialize(serialize(*this)); }
private:
    // Empty constructor used by deserializer
    FusingCurveSmoothness() { }

};

#endif /* end of include guard: FUSINGCURVESMOOTHNESS_HH */

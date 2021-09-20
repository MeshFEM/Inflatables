////////////////////////////////////////////////////////////////////////////////
// Nondimensionalization.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
// Characteristic measurements needed to make the energy and optimization
// objectives invariant to spatial scaling transforms and to scaling
// of pressure and Young's modulus (equally).
// These measurements are meant to be taken on the *initial* design mesh (i.e.,
// they should not change during optimization).
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  12/13/2020 14:45:53
////////////////////////////////////////////////////////////////////////////////
#ifndef NONDIMENSIONALIZATION_HH
#define NONDIMENSIONALIZATION_HH

#include "InflatableSheet.hh"
#include <queue>
#include <algorithm>

struct Nondimensionalization {
    using Real = InflatableSheet::Real;

    // The characteristic length scale (initial sheet bounding box diagonal).
    Real length;

    // The initial sheet's full area and thickness
    Real sheetArea;
    Real sheetThickness;

    // The initial area of the fused regions.
    Real fusedArea;

    // Characteristic width of the initial fused regions (to make wall curve
    // regularization terms roughly invariant to stripe pattern frequency and
    // mesh scale).
    Real wallWidth;

    Real youngsModulus;

    // We apply a change of equlibrium variables in TargetAttractedInflation
    // to make gradients/Hessians scale invariant. We used scaled equilibrium
    // variables x_tilde so that `x = x_tilde * equilibriumVarScale()`.
    Real equilibriumVarScale() const {
        return length;
    }

    // We apply a change of design variables in ReducedSheetOptimizer
    // to make gradients scale invariant. We used scaled undeformed
    // wall vertex positions X_tilde so that `X = X_tilde * wallWidth()`.
    // This particular normalization makes it so a unit norm step moving
    // a single vertex should just barely collapse a wall
    // (so the unit inital step tried by Scipy's L-BFGS should be reasonable).
    Real restVarScale() const {
        return wallWidth;
    }

    // The elastic and pressure potential energy terms scale like volume
    // when the design is scaled; therefore we normalize it by the
    // flat sheet's volume. We additionally normalize by Young's modulus
    // to make the potential energy term unitless (and
    // the fitting energy weight independent of material stiffness).
    Real potentialEnergyScale() const {
        // Scale factor of "300" is to match the energy to the
        // pre-youngsModulus-normalized value when E = 300MPa.
        return 300 / (sheetArea * sheetThickness * youngsModulus);
    }

    // The fitting term is sum_i A_i ||x_i - P(x_i)||^2,
    // where A_i is the barycentric area for wall vertex X_i in the initial sheet,
    // and x_i is the deformed position.
    // We nondimensionalize this by dividing by the fused region area and the
    // squared wall width. The resulting nondimensionalized term is
    // approximately the mean-squared deviation relative to the (initial)
    // wall width.
    Real fittingEnergyScale() const {
        return 1.0 / (fusedArea * wallWidth * wallWidth);
    }

    // The collapse scales like the sheet area.
    Real collapseBarrierScale() const {
        return 1e6 / sheetArea;
        // return 1.0 / sheetArea;
    }

    // All smoothing regularization terms except the 1D u Dirichlet energy
    // scale like `1 / length`. To make them invariant to both uniform scaling
    // and changes to the stripe pattern frequency, we multiply by `wallWidth`.
    Real smoothingScale() const {
        return wallWidth;
    }

    // The 1D u Dirichlet energy scales like `length`. To make it invariant to
    // both uniform scaling and changes to the stripe pattern frequency, we
    // multiply by `wallWidth / length^2`. The division by `length^2` is
    // equivalent to computing the Dirichlet energy of `u / length`, the
    // displacements expressed in units of the bounding box diagonal.
    Real dirichletSmoothingScale() const {
        return wallWidth / (length * length);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Construction
    ////////////////////////////////////////////////////////////////////////////
    Nondimensionalization(const InflatableSheet &sheet) {
        const auto &m = sheet.mesh();

        length    = m.boundingBox().dimensions().norm();
        sheetArea = m.volume();
        sheetThickness = sheet.getThickness();
        youngsModulus  = sheet.getYoungModulus();

        // Compute total fused area as well as a characteristic wall width.
        // This wall width is the median of the wall polygon widths estimated
        // using the approach from: https://gis.stackexchange.com/a/20282. We
        // visit the connected components of the fused region, computing their
        // areas and perimeters.
        fusedArea = 0.0;
        std::vector<Real> estimatedWallWidths;
        std::vector<bool> visited(m.numElements());
        std::queue<size_t> bfsQueue;
        for (auto e : m.elements()) {
            if (!sheet.isWallTri(e.index()) || visited[e.index()]) continue;
            visited[e.index()] = true;
            bfsQueue.push(e.index());
            Real area = 0, perimeter = 0;
            while (!bfsQueue.empty()) {
                size_t u = bfsQueue.front();
                bfsQueue.pop();
                const auto &eu = m.element(u);
                area += eu->volume();
                for (auto he : eu.halfEdges()) {
                    size_t v = he.opposite().tri().index();
                    // Half-edges bordering this wall component contribute to the perimeter
                    if (he.isBoundary() || !sheet.isWallTri(v)) {
                        perimeter += (he.tip().node()->p - he.tail().node()->p).norm();
                        continue;
                    }
                    // Triangle v is fused (and thus part of this component)
                    if (visited[v]) continue;
                    bfsQueue.push(v);
                    visited[v] = true;
                }
            }
            fusedArea += area;

            // Estimate the wall width by solving the equations:
            //    P = 2L + 2w
            //    A = L w
            // for the polygon's "length" and "width". These are the large
            // and small roots, respectively, of the quadratic equation
            //    x^2 - (P / 2) + A
            estimatedWallWidths.push_back(perimeter / 4 - std::sqrt(perimeter * perimeter / 16 - area));
        }

        // Compute wallWidth as (approximate) median of estimated wall widths
        std::sort(estimatedWallWidths.begin(), estimatedWallWidths.end());
        wallWidth = estimatedWallWidths.at(estimatedWallWidths.size() / 2);
    }

    Nondimensionalization(Real l, Real A, Real h, Real f, Real w, Real E) : length(l), sheetArea(A), sheetThickness(h), fusedArea(f), wallWidth(w), youngsModulus(E) { }

    ////////////////////////////////////////////////////////////////////////////
    // Serialization
    ////////////////////////////////////////////////////////////////////////////
    using State = std::tuple<Real, Real, Real, Real, Real, Real>;
    static State serialize(const Nondimensionalization &n) { return std::make_tuple(n.length, n.sheetArea, n.sheetThickness, n.fusedArea, n.wallWidth, n.youngsModulus); }
    static Nondimensionalization deserialize(const State &state) { return Nondimensionalization(std::get<0>(state), std::get<1>(state), std::get<2>(state), std::get<3>(state), std::get<4>(state), std::get<5>(state)); }
};

#endif /* end of include guard: NONDIMENSIONALIZATION_HH */

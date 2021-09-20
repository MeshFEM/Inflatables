#include "evaluate_stripe_field.hh"
#include "../subdivide_triangle.hh"

// DDG includes
#include <Mesh.h>
#include <DenseMatrix.h>

// Evaluate a triangle's stripe scalar field at particular barycentric coordinates
double evalStripe(const std::array<DDG::Complex, 3> &g, double nu, double /* nv */, const Eigen::Vector3d &b /* barycentric coordinates */) {
    double theta = g[0].re * b[0] +
                   g[1].re * b[1] +
                   g[2].re * b[2];

    // compute lArg_n
    double lArg = 0.;
    if      (b[2] <= b[0] && b[2] <= b[1]) lArg = (M_PI / 3.0) * (1.0 + (b[1] - b[0]) / (1.0 - 3.0 * b[2]));
    else if (b[0] <= b[1] && b[0] <= b[2]) lArg = (M_PI / 3.0) * (3.0 + (b[2] - b[1]) / (1.0 - 3.0 * b[0]));
    else                                   lArg = (M_PI / 3.0) * (5.0 + (b[0] - b[2]) / (1.0 - 3.0 * b[1]));

    // adjust texture coordinates
    theta += lArg * nu;
    // return theta;
    // Signed distance field estimate
    //      equivalent to pi - acos(cos(theta))     (acos returns value in [0, pi]
    {
        theta = std::fmod(theta, 2 * M_PI);
        if (theta < 0) theta += 2 * M_PI;
        // theta now in [0, 2 pi].
        // Over [0, pi], distance should interpolate down from pi to 0,
        // then back up from 0 to pi over the range [pi, 2 * pi].
        if (theta < M_PI) return M_PI - theta;
        return theta - M_PI;
    }
    // return sin(theta);
    // return theta;

    const float f = 1.; // controls line frequency
    const float s = 30.; // controls line sharpness
    const float w = 0.6; // controls line width
    double u = 1./(1.+exp(s*(cos(f*( theta ))-w)));
    return u;
}

// Sample the stripe scalar field on a subdivided mesh
void evaluate_stripe_field(const Eigen::MatrixX3d &vertices,
                           const Eigen::MatrixX3i &elements,
                           const std::vector<double> &stretchAngles,
                           const std::vector<double> &wallWidths,
                           const double frequency,
                           Eigen::MatrixX3d &outVerticesEigen,
                           Eigen::MatrixX3i &outTrianglesEigen,
                           std::vector<double> &stripeField,
                           const size_t nsubdiv, const bool glue) {
    DDG::Mesh m;
    m.import(vertices, elements);
    std::vector<MeshIO::IOVertex > outVertices;
    std::vector<MeshIO::IOElement> outTriangles;

    if (stretchAngles.size() == 0)
        m.computeCurvatureAlignedSection();
    else {
        const size_t nv = m.vertices.size();
        if (stretchAngles.size() != nv) throw std::runtime_error("Field size mismatch");
        // Compute angle in the tangent plane between a vertex's reference halfedge and the input field vector.
        for (size_t i = 0; i < nv; ++i) {
            auto &vtx = m.vertices[i];
            const DDG::Vector dir(std::cos(stretchAngles[i]), std::sin(stretchAngles[i]), 0.0);
            auto n = vtx.normal();
            auto dir_tangent    = (dir - n * dot(n, dir)).unit();
            // std::cout << dir << "\t" << dir_tangent << "\t" << n << std::endl;
            auto refdir_tangent = (vtx.he->vector() - n * dot(n, vtx.he->vector())).unit();
            DDG::Complex f(dot(dir_tangent, refdir_tangent), dot(n, cross(refdir_tangent, dir_tangent)));
            vtx.directionField = f * f; // Convention is to work with twice the angle
        }
        m.lambda = frequency;
    }

    m.parameterize();

    PointGluingMap indexForPoint;

    stripeField.clear();

    auto newTri = [&](size_t i0, size_t i1, size_t i2) { outTriangles.emplace_back(i0, i1, i2); };

    // Replicate the interpolation performed by the stripe shader,
    // evaluating on a refined mesh...
    for (const auto &f : m.faces) {
        if (f.isBoundary()) continue;

         double k  = f.fieldIndex(2.0);
         // k = 0; // FOR DEBUGGING

         if (k == 0) {
             double nu = f.paramIndex[0];
             double nv = f.paramIndex[1];
             // nu = 0; // FOR DEBUGGING
             int i = 0;

             auto he = f.he;
             std::array<DDG::Complex,    3> g;
             std::array<Eigen::Vector3d, 3> p;
             std::array<double,          3> w;

             // // DEBUGGING...
             // auto hij = f.he;
             // auto hjk = hij->next;
             // auto hki = hjk->next;
             // int csCount = hij->edge->crossesSheets + hki->edge->crossesSheets + hjk->edge->crossesSheets;

             do {
                 g[i] = he->texcoord;
                 const auto &pos = he->vertex->position;
                 p[i] << pos[0], pos[1], pos[2];
                 w[i] = wallWidths.at(he->vertex->index);

                 // Debugging visualization outputs
                 // g[i] = he->vertex->parameterization.arg();
                 // g[i] = csCount;
                 // w[i] = 0.0;

                 i++;
                 he = he->next;
             } while (he != f.he);

             if (!glue) indexForPoint.clear();
             subdivide_triangle(nsubdiv, p[0], p[1], p[2], indexForPoint,
                                [&](const Eigen::Vector3d &p_sub, double b0, double b1, double b2) {
                                     outVertices.emplace_back(p_sub);
                                     stripeField.push_back(evalStripe(g, nu, nv, Eigen::Vector3d(b0, b1, b2))
                                                            - (b0 * w[0] + b1 * w[1] + b2 * w[2]) * 0.5); // linearly interpolated half-wall-width
                                     return outVertices.size() - 1;
                                }, newTri);
         }
         else // singular triangle
         {
             // Get the three half edges.
             auto hij = f.he;
             auto hjk = hij->next;
             auto hkl = hjk->next;

             // Get the three vertices.
             auto vi = hij->vertex;
             auto vj = hjk->vertex;
             auto vk = hkl->vertex;

             // Get the three parameter values---for clarity, let "l"
             // denote the other point in the same fiber as "i".  Purely
             // for clarity, we will explicitly define the value of psi
             // at l, which of course is always just the conjugate of the
             // value of psi at i.
             DDG::Complex psiI = vi->parameterization;
             DDG::Complex psiJ = vj->parameterization;
             DDG::Complex psiK = vk->parameterization;
             DDG::Complex psiL = psiI.bar();

             double cIJ = ( hij->edge->he != hij ? -1. : 1. );
             double cJK = ( hjk->edge->he != hjk ? -1. : 1. );
             double cKL = ( hkl->edge->he != hkl ? -1. : 1. );

             // Get the three omegas, which were used to define our energy.
             double omegaIJ = hij->omega();
             double omegaJK = hjk->omega();
             double omegaKL = hkl->omega();

             // Here's the trickiest part.  If the canonical orientation of
             // this last edge is from l to k (rather than from k to l)...
             omegaKL *= cKL;
             // SIMPLER // if( cKL == -1. )
             // SIMPLER // {
             // SIMPLER //    // ...then the value of omega needs to be negated, since the
             // SIMPLER //    // original value we computed represents transport away from
             // SIMPLER //    // vertex i rather than the corresponding vertex l
             // SIMPLER //    omegaKL = -omegaKL;
             // SIMPLER // }
             // Otherwise we're ok, because the original value was computed
             // starting at k, which is exactly where we want to start anyway.

             // Now we just get consecutive values along the curve from i to j to k to l.
             // (The following logic was already described in our routine for finding
             // zeros of the parameterization.)
             if( hij->crossesSheets() )
             {
                 psiJ = psiJ.bar();
                 omegaIJ =  cIJ * omegaIJ;
                 omegaJK = -cJK * omegaJK;
             }

             // Note that the flag hkl->crossesSheets() is the opposite of what we want here:
             // based on the way it was originally computed, it flags whether the vectors at
             // Xk and Xi have a negative dot product.  But here, we instead want to know if
             // the vectors at Xk and Xl have a negative dot product.  (And since Xi=-Xl, this
             // flag will be reversed.)
             if( !hkl->crossesSheets() )
             {
                 psiK = psiK.bar();
                 omegaKL = -cKL * omegaKL;
                 omegaJK =  cJK * omegaJK;
             }

             // From here, everthing gets computed as usual.
             DDG::Complex rij( cos(omegaIJ), sin(omegaIJ) );
             DDG::Complex rjk( cos(omegaJK), sin(omegaJK) );
             DDG::Complex rkl( cos(omegaKL), sin(omegaKL) );

             double sigmaIJ = omegaIJ - ((rij*psiI)/psiJ).arg();
             double sigmaJK = omegaJK - ((rjk*psiJ)/psiK).arg();
             double sigmaKL = omegaKL - ((rkl*psiK)/psiL).arg();
             //double xi = sigmaIJ + sigmaJK + sigmaKL;

             double betaI = psiI.arg();
             double betaJ = betaI + sigmaIJ;
             double betaK = betaJ + sigmaJK;
             double betaL = betaK + sigmaKL;
             double betaM = betaI + (betaL-betaI)/2.;

             std::array<Eigen::Vector3d, 3> p_ijk;
             p_ijk[0] << vi->position[0], vi->position[1], vi->position[2];
             p_ijk[1] << vj->position[0], vj->position[1], vj->position[2];
             p_ijk[2] << vk->position[0], vk->position[1], vk->position[2];
             Eigen::Vector3d pm = (p_ijk[0] + p_ijk[1] + p_ijk[2]) / 3.;

             std::array<double, 3> w_ijk = {{ wallWidths.at(vi->index), wallWidths.at(vj->index), wallWidths.at(vk->index) }};
             double wm = (w_ijk[0] + w_ijk[1] + w_ijk[2]) / 3.;

             const double nu = 0, nv = 0;

             std::array<DDG::Complex, 3> g;
             std::array<DDG::Complex, 4> beta_ijkl = {{ DDG::Complex(betaI), DDG::Complex(betaJ), DDG::Complex(betaK), DDG::Complex(betaL) }};

             for (int offset = 0; offset < 3; ++offset) {
                 int next = (offset + 1) % 3;
                 auto newPt = [&](const Eigen::Vector3d &p, double b0, double b1, double b2) {
                     outVertices.emplace_back(p);
                     stripeField.push_back(evalStripe(g, nu, nv, Eigen::Vector3d(b0, b1, b2))
                                           - (b0 * w_ijk[offset] + b1 * w_ijk[next] + b2 * wm) * 0.5); // linearly interpolated half-wall-width
                     return outVertices.size() - 1;
                 };

                 g[0] = beta_ijkl[offset];
                 g[1] = beta_ijkl[offset + 1];
                 g[2] = DDG::Complex(betaM);
                 if (!glue) indexForPoint.clear();
                 subdivide_triangle(nsubdiv, p_ijk[offset], p_ijk[next], pm, indexForPoint, newPt, newTri);
             }
         }
    }

    outVerticesEigen .resize(outVertices .size(), 3);
    outTrianglesEigen.resize(outTriangles.size(), 3);
    for (size_t i = 0; i < outVertices.size(); ++i)
        outVerticesEigen.row(i) = outVertices[i].point;
    for (size_t i = 0; i < outTriangles.size(); ++i)
        outTrianglesEigen.row(i) = Eigen::Vector3i(outTriangles[i][0], outTriangles[i][1], outTriangles[i][2]);
}

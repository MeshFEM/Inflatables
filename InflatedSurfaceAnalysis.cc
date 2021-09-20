#include "InflatedSurfaceAnalysis.hh"
#include <MeshFEM/Triangulate.h>
#include <MeshFEM/Utilities/MeshConversion.hh>

InflatedSurfaceAnalysis::InflatedSurfaceAnalysis(const InflatableSheet &sheet, const bool useWallTriCentroids) {
    // The analysis surface vertices consist of all wall triangle centroids and then
    // all vertices that are not part of a wall triangle.
    std::vector<V3d> referenceVertices, deformedVertices;

    const auto &imesh = sheet.mesh();
    if (useWallTriCentroids) {
        // Barycenter of each wall triangle
        for (auto tri : imesh.elements()) {
            if (sheet.isWallTri(tri.index())) {
                InflatableSheet::M3d triCornerPos;
                sheet.getDeformedTriCornerPositions(tri.index(), 0, triCornerPos);
                referenceVertices.emplace_back(((tri.node(0)->p + tri.node(1)->p + tri.node(2)->p) / 3.0).eval());
                deformedVertices .emplace_back(triCornerPos.rowwise().mean());
            }
        }
    }
    for (auto v : imesh.vertices()) {
        if (!sheet.isWallVtx(v.index())) continue;
        bool partOfWallTri = false;
        for (auto he : v.incidentHalfEdges()) {
            if (!he.tri()) continue;
            if (sheet.isWallTri(he.tri().index())) partOfWallTri = true;
        }
        if (useWallTriCentroids && partOfWallTri) continue;
        referenceVertices.emplace_back(v.node()->p);
        deformedVertices .emplace_back(sheet.getDeformedVtxPosition(v.index(), 0));
    }

    const size_t nv = referenceVertices.size();

    std::vector<MeshIO::IOVertex > verticesForTriangulation, triangulatedVertices;
    verticesForTriangulation.resize(nv);
    for (size_t i = 0; i < nv; ++i)
        verticesForTriangulation[i].point = referenceVertices[i].cast<double>();
    std::vector<MeshIO::IOElement> elements;
    triangulatePoints(verticesForTriangulation, triangulatedVertices, elements);
    std::cout << "Input vertices: " << referenceVertices.size() << std::endl;
    std::cout << "Triangulation vertices: " << nv << std::endl;
    if (nv != triangulatedVertices.size()) throw std::runtime_error("Triangle inserted new vertices...");

    // Clean up reference mesh (at the boundary, we can get 3 nearly colinear
    // vertices which leads to triangles with nearly zero area; remove these).
    {
        Mesh tmpMesh(elements, referenceVertices);
        // Delete extremely tiny triangles
        const double fullVol = tmpMesh.volume();
        size_t ei = 0;
        elements.erase(std::remove_if(elements.begin(), elements.end(), [&ei, fullVol, &tmpMesh](const MeshIO::IOElement &) {
                    return tmpMesh.element(ei++)->volume() / fullVol < 1e-15; }), elements.end());

        // Remove newly dangling referenceVertices/deformedVertices
        std::vector<bool> seen(referenceVertices.size(), false);
        for (const auto &e : elements) {
            for (size_t c = 0; c < e.size(); ++c)
                seen.at(e[c]) = true;
        }
        size_t curr = 0;
        std::vector<size_t> vertexRenumber(referenceVertices.size(), std::numeric_limits<size_t>::max());
        for (size_t i = 0; i < referenceVertices.size(); ++i) {
            if (seen[i]) {
                referenceVertices[curr] = referenceVertices[i];
                deformedVertices[curr] = deformedVertices[i];
                vertexRenumber[i] = curr++;
            }
        }
        for (auto &e : elements) {
            for (size_t c = 0; c < e.size(); ++c)
                e[c] = vertexRenumber.at(e[c]);
        }
        referenceVertices.resize(curr);
        deformedVertices.resize(curr);
    }

    m_analysisMesh = std::make_unique<Mesh>(elements, referenceVertices);
    inflatedPositions.resize(nv, 3);
    for (size_t i = 0; i < nv; ++i)
        inflatedPositions.row(i) = deformedVertices[i].transpose();
}

CurvatureInfo InflatedSurfaceAnalysis::curvature() const {
    auto F = getF(mesh());
    return CurvatureInfo(inflatedPositions.cast<double>(), F);
}

// Note: the mapping considered here is the inverse of the one used in the parametrization problem.
// So the deformed (contracted) direction is the singular vector corresponding
// to the *smaller* singular value instead of the larger.
// Also, the right "stretch" directions are the ones in 2D and the left
// "stretch" directions are in 3D.
InflatedSurfaceAnalysis::MetricInfo::MetricInfo(const Mesh &m, const MX3d &deformedPositions) {
    const size_t nt = m.numElements();
    left_stretch .resize(nt, 3);
    right_stretch.resize(nt, 3);
    sigma_1.resize(nt);
    sigma_2.resize(nt);

    for (auto tri : m.elements()) {
        const size_t ti = tri.index();
        const auto &gradLambda = tri->gradBarycentric();
        M3d F = deformedPositions.row(tri.node(0).index()).transpose() * gradLambda.col(0).transpose()
              + deformedPositions.row(tri.node(1).index()).transpose() * gradLambda.col(1).transpose()
              + deformedPositions.row(tri.node(2).index()).transpose() * gradLambda.col(2).transpose();

        Eigen::JacobiSVD<M3d> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);

        const auto &sigmas = svd.singularValues();
        sigma_1[ti] = sigmas[0];
        sigma_2[ti] = sigmas[1];
        if (std::abs(sigmas[2]) > 1e-10) throw std::runtime_error("Triangle Jacobian does not have rank 2");

        left_stretch .row(ti) = svd.matrixU().col(1).transpose();
        right_stretch.row(ti) = svd.matrixV().col(1).transpose();
    }
}

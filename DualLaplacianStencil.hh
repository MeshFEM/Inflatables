////////////////////////////////////////////////////////////////////////////////
// DualLaplacianStencil.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  A configurable stencil for Laplacian regularization terms for face-based
//  (piecewise constant) quantities. We support a plain 1D graph Laplacian
//  (with optional scaling by inverse dual edge lengths to ensure proper
//  scaling under refinement) and a Laplacian based on arbitrarily
//  triangulating the dual mesh and then using an intrinsic delaunay
//  triangulation-based Laplacian to cope with bad mesh quality.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  01/01/2020 16:26:45
////////////////////////////////////////////////////////////////////////////////
#ifndef DUALLAPLACIANSTENCIL_HH
#define DUALLAPLACIANSTENCIL_HH

#include "DualMesh.hh"
#include <Eigen/Sparse>
#include <MeshFEM/Utilities/MeshConversion.hh>

// We need to provide a wrapper for this function instead of calling it
// directly: libigl's `Triplet` will clash with ours if we include
// `intrinsic_delaunay_cotmatrix.h` in this header.
void igl_intrinsic_delaunay_cotmatrix(const Eigen::MatrixX3d &V,
                                      const Eigen::MatrixXi  &F,
                                      Eigen::SparseMatrix<Real> &L);

template<class Mesh>
struct DualLaplacianStencil {
    enum class Type { DualGraph, DualMeshIDT };

    Type type = Type::DualGraph;
    bool useUniformGraphWeights = true;

    DualLaplacianStencil(const Mesh &m) : m_mesh(m) {
        m_graphEdgeInvLen.resize(m.numHalfEdges());
        Real totalEdgeLen = 0;
        size_t num = 0;
        for (const auto &he : m.halfEdges()) {
            if (he.isBoundary()) continue;
            Real l = (m.elementBarycenter(he.           tri().index())
                    - m.elementBarycenter(he.opposite().tri().index())).norm();
            m_graphEdgeInvLen[he.index()] = 1.0 / l;
            totalEdgeLen += l;
            ++num;
        }
        m_averageEdgeLen = totalEdgeLen / num;

        std::vector<MeshIO::IOVertex > dualVertices;
        std::vector<MeshIO::IOElement> dualTris;
        std::vector<size_t> originatingPolygon;
        triangulatedBarycentricDual(m, dualVertices, dualTris, originatingPolygon);
        // LibIGL constructs the negative semi-definite Laplacian (which has
        // positive weights in the off-diagonals, assuming non-obtuse angles).
        igl_intrinsic_delaunay_cotmatrix(getV(dualVertices), getF(dualTris), m_dualIDTLaplacian);
    }

    // Call visitor(i, j, w_ij) for each stencil edge e_ij incident face i.
    template<class F>
    void visit(size_t i, F &&visitor) const {
        if (type == Type::DualGraph) {
            for (const auto &he : m_mesh.element(i).halfEdges()) {
                if (he.isBoundary()) continue;
                const auto &tri_j = he.opposite().tri();
                if (useUniformGraphWeights) visitor(i, tri_j.index(), 1.0);
                else                        visitor(i, tri_j.index(), m_averageEdgeLen * m_graphEdgeInvLen[he.index()]);
            }
        }
        else if (type == Type::DualMeshIDT) {
            // Loop over the ith column of the sparse Laplacian matrix...
            for (Eigen::SparseMatrix<Real>::InnerIterator it(m_dualIDTLaplacian, i); it; ++it) {
                size_t j = it.index(); // inner index (could also use it.row(), but that is storage-order-dependent)
                if (j == i) continue;  // skip diagonal
                visitor(i, j, it.value());
            }
        }
        else {
            assert(false);
        }
    }

    // Visit each edge e_ij with (i < j) in the graph
    template<class F>
    void visit_edges(F &&visitor) const {
        if (type == Type::DualGraph) {
            for (const auto &tri_i : m_mesh.elements()) {
                const size_t i = tri_i.index();
                for (const auto &he : tri_i.halfEdges()) {
                    if (he.isBoundary()) continue;
                    const auto &tri_j = he.opposite().tri();
                    const size_t j = tri_j.index();
                    if (i >= j) continue;
                    if (useUniformGraphWeights) visitor(i, j, 1.0);
                    else                        visitor(i, j, m_averageEdgeLen * m_graphEdgeInvLen[he.index()]);
                }
            }
        }
        else if (type == Type::DualMeshIDT) {
            for (const auto &tri_i : m_mesh.elements()) {
                const size_t i = tri_i.index();
                // Loop over the ith column of the sparse Laplacian matrix's upper triangle in order
                for (Eigen::SparseMatrix<Real>::InnerIterator it(m_dualIDTLaplacian, i); it; ++it) {
                    const size_t j = it.index(); // inner index (could also use it.row(), but that is storage-order-dependent)
                    if (j == i) break; // skip diagonal/lower triangle
                    visitor(i, j, it.value());
                }
            }
        }
        else {
            assert(false);
        }
    }

private:
    const Mesh &m_mesh;
    Eigen::SparseMatrix<Real> m_dualIDTLaplacian;
    Real m_averageEdgeLen;
    std::vector<Real> m_graphEdgeInvLen;
};

#endif /* end of include guard: DUALLAPLACIANSTENCIL_HH */

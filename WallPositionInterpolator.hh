////////////////////////////////////////////////////////////////////////////////
// WallPositionInterpolator.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Position the non-wall rest vertices of a sheet as smoothly as possible given
//  positions of the wall vertices. The wall vertices are referred to as
//  "boundary" in the following code since they are the boundary of the domain
//  throughout which we construct a harmonic interpolation.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  07/22/2019 16:55:15
////////////////////////////////////////////////////////////////////////////////
#ifndef WALLPOSITIONINTERPOLATOR_HH
#define WALLPOSITIONINTERPOLATOR_HH

#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/Laplacian.hh>
#include "InflatableSheet.hh"
#include <memory>

template<typename Real>
struct WallPositionInterpolator {
    using VXd  = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using MX2d = Eigen::Matrix<Real, Eigen::Dynamic, 2>;

    WallPositionInterpolator(const InflatableSheet &isheet) {
        const auto &m = isheet.mesh();
        const size_t nv = m.numVertices();
        std::vector<bool> isBoundary(nv);
        size_t nwv = 0;
        for (size_t i = 0; i < nv; ++i) {
            isBoundary[i] = isheet.isWallVtx(i);
            if (isBoundary[i]) ++nwv;
        }

        if (nwv != isheet.wallVertices().size()) {
            throw std::runtime_error("Inconsistent num wall vertices");
        }

        m_init(Laplacian::construct<1>(m), isBoundary, nwv);
    }

    size_t numVertices()     const { return m_initialRestLaplacian.m; }
    size_t numWallVertices() const { return m_numWallVertices; }

    // Get interpolated rest positions for all (top) sheet vertices given rest
    // positions for the wall vertices.
    void interpolate(Eigen::Ref<const MX2d> wallVtxPositions, Eigen::Ref<MX2d> result) const {
        if (size_t(wallVtxPositions.rows()) != numWallVertices()) throw std::runtime_error("Incorrect wallVtxPositions row count");
        if (size_t(result.rows()) != numVertices())               throw std::runtime_error("Incorrect result row count");

        for (size_t c = 0; c < 2; ++c) {
            VXd neg_v_b(VXd::Zero(numVertices()));
            injectBoundaryEntries(-wallVtxPositions.col(c), neg_v_b);

            auto neg_Lib_v_b = m_initialRestLaplacian.apply(neg_v_b.template cast<double>().eval());
            removeBoundaryEntriesInPlace(neg_Lib_v_b);

            auto v_i = m_LiiFactorizer->solve(neg_Lib_v_b);

            injectInteriorEntries(v_i.template cast<Real>(), result.col(c));
            injectBoundaryEntries(wallVtxPositions.col(c),   result.col(c));
        }
    }

    // Get the "boundary" differential form that, when dotted with a boundary
    // vertex perturbation, produces the same result as dotting vtxDiffForm with
    // the interpolation of the boundary perturbation.
    void adjoint(Eigen::Ref<const MX2d> vtxDiffForm, Eigen::Ref<MX2d> result) const {
        if (size_t(vtxDiffForm.rows()) != numVertices()) throw std::runtime_error("Incorrect vtxDiffForm row count");
        if (size_t(result.rows()) != numWallVertices())  throw std::runtime_error("Incorrect result row count");

        for (size_t c = 0; c < 2; ++c) {
            extractBoundaryEntries(vtxDiffForm.col(c), result.col(c));

            VXd dvi = vtxDiffForm.col(c);
            removeBoundaryEntriesInPlace(dvi);

            auto Lii_inv_dvi = m_LiiFactorizer->solve(dvi.template cast<double>().eval()).template cast<Real>().eval();
            VXd padded_Lii_inv_dvi(VXd::Zero(numVertices()));
            injectInteriorEntries(Lii_inv_dvi, padded_Lii_inv_dvi);
            VXd Lbi_Lii_inv_dvi(numWallVertices());
            extractBoundaryEntries(m_initialRestLaplacian.apply(padded_Lii_inv_dvi.template cast<double>().eval()).template cast<Real>().eval(),
                                   Lbi_Lii_inv_dvi);

            result.col(c) -= Lbi_Lii_inv_dvi;
        }
    }

    const SuiteSparseMatrix &initialRestLaplacian() const { return m_initialRestLaplacian; }
    const std::vector<bool> &isBoundary() const { return m_isBoundary; }

    ////////////////////////////////////////////////////////////////////////////
    // Serialization + cloning support (for pickling)
    ////////////////////////////////////////////////////////////////////////////
    using State = std::tuple<SuiteSparseMatrix, std::vector<bool>, size_t>;
    static State serialize(const WallPositionInterpolator &wpi) {
        return std::make_tuple(wpi.m_initialRestLaplacian, wpi.m_isBoundary, wpi.m_numWallVertices);
    }
    static std::shared_ptr<WallPositionInterpolator> deserialize(const State &state) {
        auto wpi = std::shared_ptr<WallPositionInterpolator>(new WallPositionInterpolator()); // std::make_shared can't call private constructor
        wpi->m_init(std::get<0>(state), std::get<1>(state), std::get<2>(state));
        return wpi;
    }
    std::shared_ptr<WallPositionInterpolator> clone() const { return deserialize(serialize(*this)); }

private:
    SuiteSparseMatrix m_initialRestLaplacian;
    std::unique_ptr<CholmodFactorizer> m_LiiFactorizer;
    std::vector<bool> m_isBoundary;
    size_t m_numWallVertices;

    // Empty constructor used for deserialization
    WallPositionInterpolator() { }

    // To be called by constructor/deserializer
    void m_init(const SuiteSparseMatrix &L, const std::vector<bool> &isBoundary, size_t nwv) {
        m_initialRestLaplacian = L;
        SuiteSparseMatrix Lii = L;
        m_isBoundary = isBoundary;
        m_numWallVertices = nwv;
        Lii.rowColRemoval([&](SuiteSparse_long i) { return m_isBoundary[i]; });
        m_LiiFactorizer = std::make_unique<CholmodFactorizer>(Lii);
    }

    template<class Real2>
    void removeBoundaryEntriesInPlace(Eigen::Matrix<Real2, Eigen::Dynamic, 1> &x) const {
        int back = 0;
        for (int i = 0; i < x.rows(); ++i)
            if (!m_isBoundary[i]) x[back++] = x[i];
        x.conservativeResize(back);
    }

    void injectInteriorEntries(Eigen::Ref<const VXd> interiorEntries, Eigen::Ref<VXd> result) const {
        if (size_t(result.rows()) != numVertices()) throw std::runtime_error("Incorrect size of result");
        int back = 0;
        for (int i = 0; i < result.rows(); ++i)
            if (!m_isBoundary[i]) result[i] = interiorEntries[back++];

        if (back != interiorEntries.rows()) throw std::runtime_error("Incorrect size of interiorEntries");
    }

    void injectBoundaryEntries(Eigen::Ref<const VXd> boundaryEntries, Eigen::Ref<VXd> result) const {
        if (size_t(result.rows()) != numVertices()) throw std::runtime_error("Incorrect size of result");
        int back = 0;
        for (int i = 0; i < result.rows(); ++i)
            if (m_isBoundary[i]) result[i] = boundaryEntries[back++];

        if (back != boundaryEntries.rows()) throw std::runtime_error("Incorrect size of boundaryEntries");
    }

    void extractBoundaryEntries(Eigen::Ref<const VXd> entries, Eigen::Ref<VXd> result) const {
        const size_t nv = numVertices();
        if (size_t(result.rows()) != numWallVertices()) throw std::runtime_error("Incorrect size of result");
        if (size_t(entries.rows()) != nv)               throw std::runtime_error("Incorrect size of result");
        int back = 0;
        for (size_t i = 0; i < nv; ++i)
            if (m_isBoundary[i]) result[back++] = entries[i];
    }
};

#endif /* end of include guard: WALLPOSITIONINTERPOLATOR_HH */

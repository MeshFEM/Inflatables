////////////////////////////////////////////////////////////////////////////////
// CompressionPenalty.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Penalize the amount of compression at equilibrium in (a subset of) the
//  sheet triangles. This is used as a proxy for stability: we assume that a
//  sheet with more regions in tension is more stable than one with wrinkled,
//  compressed regions.
//  It also serves as a regularization to address the fact that design
//  alterations expanding regions already under compression in the equilibrium
//  deformation are in the nullspace of the fitting energy. Therefore nothing
//  apart from the wall stretch barrier prevents the design mesh triangles of
//  compressed wall regions from growing arbitrarily.
//  We use an efficient, differentiable measure of compression the difference
//  between the elastic energy without and with the tension field theory
//  relaxation:
//      J(x, X) = E_full(x, X) - E_TFT(x, X)
//  By adding a slightly greater weight to `E_TFT` in this formula, we can
//  encourage a small amount of tension.
//
//  To prevent an extremely compressed triangle from dominating the energy,
//  we optionally apply a sigmoid or some other modulation function 'j' so that
//  the full energy is:
//      J(x, X) = int_Omega j(psi_full(grad_X x) - psi_tft(grad_X x)) dA(X)
//
//  By default, we apply the penalty to the wall triangles only since we expect
//  the concave side of tubes to be in compression even in good designs.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  12/26/2020 12:57:53
////////////////////////////////////////////////////////////////////////////////
#ifndef COMPRESSIONPENALTY_HH
#define COMPRESSIONPENALTY_HH

#include "InflatableSheet.hh"
#include <functional>
#include <MeshFEM/Utilities/Cloning.hh>

struct CPModulation {
    using Real = InflatableSheet::Real;
    virtual Real  j(Real x) const = 0;
    virtual Real dj(Real x) const = 0;
    virtual std::unique_ptr<CPModulation> clone() const = 0;
    virtual ~CPModulation() { }
};

struct CPModulationIdentity : public CloneableSubclass<CPModulation, CPModulationIdentity> {
    using Real = CPModulation::Real;
    virtual Real  j(Real x) const override { return   x; }
    virtual Real dj(Real x) const override { return 1.0; }
};

struct CPModulationTanh : public CloneableSubclass<CPModulation, CPModulationTanh> {
    using Real = CPModulation::Real;
    virtual Real  j(Real x) const override { return std::tanh(x); }
    virtual Real dj(Real x) const override { return std::pow(1.0 / std::cosh(x), 2); }
};

// j(x) = (x + eps)^(1/p) - eps^(1/p)
// where `eps` can be chosen to prevent the derivative from blowing up at x = 0.
// We recommend the value eps = p^(p / (1 - p)) since this ensures a derivative of 1 at x = 0;
// this is the value configured automatically by `set_p`
struct CPModulationPthRoot : public CloneableSubclass<CPModulation, CPModulationPthRoot> {
    using Real = CPModulation::Real;
    Real p = 8.0;
    Real eps = canonical_eps(8.0);

    static constexpr Real canonical_eps(Real p) noexcept { return p > 1 ? std::pow(p, p / (1 - p)) : 0.0; }
    // Choose an epsilon that gives a slope of 1 at x = 0.
    void set_p(Real pnew) { p = pnew; eps = canonical_eps(p); }
    virtual Real  j(Real x) const override { return std::pow(x + eps, 1.0 / p) - std::pow(eps, 1.0 / p); }
    virtual Real dj(Real x) const override { return x + eps > 0 ? ((1.0 / p) * std::pow(x + eps, 1.0 / p - 1.0)) : 0.0; }

    using State = std::tuple<Real, Real>;
    static State serialize(const CPModulationPthRoot &cpm) {
        return std::make_tuple(cpm.p, cpm.eps);
    }
    static std::shared_ptr<CPModulationPthRoot> deserialize(const State &state) {
        auto result = std::make_shared<CPModulationPthRoot>();
        result->p   = std::get<0>(state);
        result->eps = std::get<1>(state);
        return result;
    }
};

struct CPModulationCustom : public CloneableSubclass<CPModulation, CPModulationCustom> {
    using Real = CPModulation::Real;
    std::function<Real(Real)> j_func, dj_func;
    virtual Real  j(Real x) const override { return  j_func(x); }
    virtual Real dj(Real x) const override { return dj_func(x); }

    using State = std::tuple<std::function<Real(Real)>, std::function<Real(Real)>>;
    static State serialize(const CPModulationCustom &cpm) {
        return std::make_tuple(cpm.j_func, cpm.dj_func);
    }
    static std::shared_ptr<CPModulationCustom> deserialize(const State &state) {
        auto result = std::make_shared<CPModulationCustom>();
        result-> j_func = std::get<0>(state);
        result->dj_func = std::get<1>(state);
        return result;
    }
};

struct CompressionPenalty {
    using Mesh          = typename InflatableSheet::Mesh;
    using Real          = typename InflatableSheet::Real;
    using M2d           = typename InflatableSheet::M2d;
    using M3d           = typename InflatableSheet::M3d;
    using M32d          = typename InflatableSheet::M32d;
    using M23d          = typename InflatableSheet::M23d;
    using MX2d          = typename InflatableSheet::MX2d;
    using VXd           = typename InflatableSheet::VXd;
    using EnergyDensity = typename InflatableSheet::EnergyDensity;

    using AXb = Eigen::Array<bool, Eigen::Dynamic, 1>;

    CompressionPenalty(std::shared_ptr<const InflatableSheet> s)
        : m_sheet(s), includeSheetTri(AXb::Constant(s->numSheetTris(), false))
    {
        modulation = std::make_unique<CPModulationPthRoot>();
        // Apply the penalty to the wall triangles
        const size_t nt = mesh().numTris();
        for (size_t ti = 0; ti < nt; ++ti) {
            bool iwt = s->isWallTri(ti);
            includeSheetTri[s->sheetTriIdx(0, ti)] = iwt;
            includeSheetTri[s->sheetTriIdx(1, ti)] = iwt;
        }
    }

    const Mesh            &mesh()  const { return m_sheet->mesh(); }
    const InflatableSheet &sheet() const { return *m_sheet; }

    Real J() const {
        validateIncludeSheetTri();

        Real result = 0.0;

        // Compute the TFT and non-TFT energies of the included triangles
        const auto &s = sheet();
        const auto &ted = s.triEnergyDensities();
        for (const auto tri : mesh().elements()) {
            for (size_t sheetIdx = 0; sheetIdx < 2; ++sheetIdx) {
                size_t sheetTriIdx = s.sheetTriIdx(sheetIdx, tri.index());
                if (!includeSheetTri[sheetTriIdx]) continue;
                EnergyDensity psi = ted[sheetTriIdx];
                Real psi_full, psi_tft;

                psi.setRelaxationEnabled(false); psi_full = psi.energy();
                psi.setRelaxationEnabled(true ); psi_tft  = psi.energy();
                result += tri->volume() * modulation->j(psi_full - Etft_weight * psi_tft);
            }
        }

        return result;
    }

    VXd dJ_dx() const {
        const auto &s = sheet();
        const auto &BtGradLambdas = s.shapeFunctionGradients();
        const auto &ted = s.triEnergyDensities();

        auto accumulatePerTriContrib = [&](size_t tri_idx, VXd &out) {
            const auto &tri = mesh().tri(tri_idx);
            for (size_t sheetIdx = 0; sheetIdx < 2; ++sheetIdx) {
                const size_t sheetTriIdx = s.sheetTriIdx(sheetIdx, tri.index());
                if (!includeSheetTri[sheetTriIdx]) return;
                M3d contrib;
                {
                    EnergyDensity psi = ted[sheetTriIdx];
                    Real psi_full, psi_tft;
                    M32d dj_dF;
                    psi.setRelaxationEnabled(false); psi_full = psi.energy(); dj_dF  = psi.denergy();
                    psi.setRelaxationEnabled(true ); psi_tft  = psi.energy(); dj_dF -= psi.denergy() * Etft_weight;
                    dj_dF *= modulation->dj(psi_full - Etft_weight * psi_tft);
                    // dF/dx_{i,c} = d(JB)/x_{i,c} = (e_c otimes grad lambda_i) B = (e_c otimes (B^T grad lambda_i))
                    contrib = (tri->volume() * dj_dF) * BtGradLambdas[sheetTriIdx];
                }

                for (const auto v : tri.vertices())
                    out.segment<3>(s.varIdx(sheetIdx, v.index(), 0)) += contrib.col(v.localIndex());
            }
        };

        VXd result(VXd::Zero(s.numVars()));
        // assemble_parallel(accumulatePerTriContrib, result, mesh().numElements());
        // The serial version actually seems to be faster... (not enough work is done for each tri).
        const size_t ntri = mesh().numElements();
        for (size_t i = 0; i < ntri; ++i)
            accumulatePerTriContrib(i, result);

        return result;
    }

    MX2d dJ_dX() const {
        const auto &s = sheet();
        const auto &ted = s.triEnergyDensities();

        auto accumulatePerTriContrib = [&](size_t tri_idx, MX2d &out) {
            const auto &tri = mesh().tri(tri_idx);
            const auto &gradLambda  = tri->gradBarycentric();
            for (size_t sheetIdx = 0; sheetIdx < 2; ++sheetIdx) {
                const size_t sheetTriIdx = s.sheetTriIdx(sheetIdx, tri.index());
                if (!includeSheetTri[sheetTriIdx]) return;
                M23d contrib;
                {
                    EnergyDensity psi = ted[sheetTriIdx];
                    Real psi_full, psi_tft;
                    M32d dj_dF;
                    psi.setRelaxationEnabled(false); psi_full = psi.energy(); dj_dF  = psi.denergy();
                    psi.setRelaxationEnabled(true ); psi_tft  = psi.energy(); dj_dF -= psi.denergy() * Etft_weight;
                    dj_dF *= modulation->dj(psi_full - Etft_weight * psi_tft);

                    // d/dX_i,c j(psi(F)) * vol = j' dpsi_dF : (- (FB^T) (e_c otimes grad lambda_j) B) + j dvol/dX_i,c
                    //                          = -(B F^T (dj_dF)) : (e_c otimes B^T grad lambda_j) + j dvol/dX_i,c
                    //                          = -(B F^T (dj_dF)) grad lambda_j + j dvol/dX_i,c
                    const M3d &FBt = s.deformationGradient3D(sheetTriIdx);
                    const M23d &BtGradLambda = s.shapeFunctionGradients()[sheetTriIdx];
                    contrib = modulation->j(psi_full - Etft_weight * psi_tft) * gradLambda.template topRows<2>() // Dilation term
                                    - (FBt.transpose().template topRows<2>() * dj_dF) * BtGradLambda;           // Changing gradient operator
                    contrib *= tri->volume();
                    // std::cout << "F:" << std::endl << F << std::endl;
                    // std::cout << "dj_dF:" << std::endl << dj_dF << std::endl;
                    // std::cout << "BtGradLambdas:" << std::endl << BtGradLambdas[sheetTriIdx] << std::endl;
                }

                for (const auto v : tri.vertices())
                    out.row(v.index()) += contrib.col(v.localIndex()).transpose();
            }
        };

        MX2d result = MX2d::Zero(mesh().numVertices(), 2);
        const size_t ntri = mesh().numElements();
        for (size_t i = 0; i < ntri; ++i)
            accumulatePerTriContrib(i, result);

        return result;
    }

    void validateIncludeSheetTri() const {
        if (size_t(includeSheetTri.rows()) != 2 * mesh().numTris())
            throw std::runtime_error("invalid includeSheetTri size");
    }

    ////////////////////////////////////////////////////////////////////////////
    // Serialization support for pickling
    ////////////////////////////////////////////////////////////////////////////
    using State = std::tuple<std::shared_ptr<const InflatableSheet>, AXb, Real, std::shared_ptr<CPModulation>>;

    static State serialize(const CompressionPenalty &cp) {
        return std::make_tuple(cp.m_sheet, cp.includeSheetTri, cp.Etft_weight, cp.modulation->clone());
    }

    static std::shared_ptr<CompressionPenalty> deserialize(const State &state) {
        auto result = std::make_shared<CompressionPenalty>(std::get<0>(state));
        result->includeSheetTri = std::get<1>(state);
        result->Etft_weight     = std::get<2>(state);
        result->modulation      = std::get<3>(state);
        return result;
    }

private:
    ////////////////////////////////////////////////////////////////////////////
    // Private member variables
    ////////////////////////////////////////////////////////////////////////////
    std::shared_ptr<const InflatableSheet> m_sheet;
public:
    ////////////////////////////////////////////////////////////////////////////
    // Public member variables
    ////////////////////////////////////////////////////////////////////////////
    std::shared_ptr<CPModulation> modulation;
    AXb includeSheetTri;
    Real Etft_weight = 1.0;
};

#endif /* end of include guard: COMPRESSIONPENALTY_HH */

////////////////////////////////////////////////////////////////////////////////
// InflatedSurfaceAnalysis.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Analyze properties of the inflated structure by extracting an interpolating
//  surface.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  05/13/2019 11:49:15
////////////////////////////////////////////////////////////////////////////////
#ifndef INFLATEDSURFACEANALYSIS_HH
#define INFLATEDSURFACEANALYSIS_HH

#include "InflatableSheet.hh"
#include <memory>
#include <MeshFEM/MeshIO.hh>
#include "curvature.hh"

struct InflatedSurfaceAnalysis {
    using Real = InflatableSheet::Real;
    using V3d  = InflatableSheet::V3d;
    using M3d  = Eigen::Matrix<Real,              3, 3>;
    using MX3d = Eigen::Matrix<Real, Eigen::Dynamic, 3>;
    using VXd  = Eigen::VectorXd;
    using Mesh = InflatableSheet::Mesh;

    struct MetricInfo {
        MX3d  left_stretch;
        MX3d right_stretch;
        VXd sigma_1, sigma_2;

        MetricInfo(const Mesh &m, const MX3d &deformedPositions);
    };

    InflatedSurfaceAnalysis(const InflatableSheet &sheet, const bool useWallTriCentroids = true);

    const Mesh &mesh() const { return *m_analysisMesh; }
    Mesh inflatedSurface() const {
        Mesh imesh = mesh();
        imesh.setNodePositions(inflatedPositions);
        return imesh;
    }

    CurvatureInfo curvature() const;
    MetricInfo       metric() const { return MetricInfo(mesh(), inflatedPositions); }

    MX3d inflatedPositions;
private:
    std::unique_ptr<Mesh> m_analysisMesh;
};

#endif /* end of include guard: INFLATEDSURFACEANALYSIS_HH */

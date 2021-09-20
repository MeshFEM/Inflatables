#include "../parametrization_newton.hh"
#include <MeshFEM/MeshIO.hh>

#include "../parametrization.hh"

using Mesh = parametrization::Mesh;

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "usage: shear_arap in_mesh.obj" << std::endl;
        exit(-1);
    }

    std::string inPath = argv[1];

    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    MeshIO::load(inPath, vertices, elements);

    auto mesh = std::make_shared<Mesh>(elements, vertices);

    auto f = parametrization::lscm(*mesh);

    parametrization::LocalGlobalParametrizer param(mesh, f);
    param.setAlphaMin(1.0);
    param.setAlphaMax(M_PI / 2);

    std::cout.precision(19);

    // Local-global iterations
    const size_t nit = 1000;
    for (size_t it = 0; it < nit; ++it) {
        std::cout << "Local-Global energy:\t" << param.energy() << std::endl;
        param.runIteration();
    }
    std::cout << "Local-Global energy:\t" << param.energy() << std::endl;

    // Regularized parametrizer iterations
    parametrization::RegularizedParametrizer rparam(param);

    NewtonOptimizerOptions opts;
    opts.useIdentityMetric = true;

    const std::vector<size_t> fixedVars{rparam.uOffset(), rparam.vOffset(), rparam.phiOffset()};
    parametrization::regularized_parametrization_newton(rparam, fixedVars, opts);
    
    return 0;
}

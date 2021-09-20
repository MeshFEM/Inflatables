Inflatables
===========

<img src='http://julianpanetta.com/publication/inflatables/teaser.jpg' width='229px' height='220px' />

This is the codebase for the Siggraph paper
[Computational Inverse Design of Surface-based Inflatables](http://julianpanetta.com/publication/inflatables/).
The code is primarily written in C++, but is meant to be interacted with through the Python
bindings.

# Getting Started
##
The C++ code relies on `Boost` and `CHOLMOD/UMFPACK`, which must be installed
separately.

The code also relies on several dependencies that are included as submodules:
[MeshFEM](https://github.com/MeshFEM/MeshFEM),
[libigl](https://github.com/libigl/libigl),

Finally, it includes a version of Keenan Crane's [stripe patterns code](https://www.cs.cmu.edu/~kmcrane/Projects/StripePatterns/)
modified to generate fusing curve patterns and fixing a few issues with boundary handling.

### macOS
You can install all the mandatory dependencies on macOS with [MacPorts](https://www.macports.org).

```bash
# Build/version control tools, C++ code dependencies
sudo port install cmake boost suitesparse ninja
# Dependencies for jupyterlab/notebooks
sudo port install python39
sudo port install npm7
```

### Ubuntu 19.04
A few more packages need to be installed on a fresh Ubuntu 19.04 install:
```bash
# Build/version control tools
sudo apt install git cmake ninja-build
# Dependencies for C++ code
sudo apt install libboost-filesystem-dev libboost-system-dev libboost-program-options-dev libsuitesparse-dev
# Dependencies (pybind11, jupyterlab/notebooks)
sudo apt install python3-pip npm
# Ubuntu 19.04 packages an older version of npm that is incompatible with its nodejs version...
sudo npm install npm@latest -g
```

## Obtaining and Building

Clone this repository *recursively* so that its submodules are also downloaded:

```bash
git clone --recursive https://github.com/jpanetta/Inflatables
```

Build the C++ code and its Python bindings using `cmake` and your favorite
build system. For example, with [`ninja`](https://ninja-build.org):

```bash
cd Inflatables
mkdir build && cd build
cmake .. -GNinja
ninja
```

## Running the Jupyter Notebooks
The preferred way to interact with the rods code is in a Jupyter notebook,
using the Python bindings.
We recommend you install the Python dependencies and JupyterLab itself in a
virtual environment.

```bash
pip3 install wheel # Needed if installing in a virtual environment
pip3 install jupyterlab ipykernel=5.5.5 # Use a slightly older version of ipykernel to avoid cluttering notebook with stdout content.
# If necessary, follow the instructions in the warnings to add the Python user
# bin directory (containing the 'jupyter' binary) to your PATH...

git clone https://github.com/jpanetta/pythreejs
cd pythreejs
pip3 install -e .
cd js
jupyter labextension install .

pip3 install matplotlib scipy
pip3 install shapely # dependency of the fabrication file generation
```

Launch Jupyter lab from the root python directory:
```bash
cd python
jupyter lab
```

Now try opening and running an demo notebook, e.g.,
[`python/Demos/ConcentricCircles.ipynb`](https://github.com/jpanetta/Inflatables/blob/master/python/Demos/ConcentricCircles.ipynb).

For an example of the full inverse design pipeline from input surface to fabrication file output, please see
[`python/Demos/Lilium.ipynb`](https://github.com/jpanetta/Inflatables/blob/master/python/Demos/Lilium.ipynb).
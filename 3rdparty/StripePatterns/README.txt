Note: this directory contains a stripped-down version of Keenan's code from:
https://www.cs.cmu.edu/~kmcrane/Projects/StripePatterns/

StripePatterns - Keenan Crane (2015)
------------------------------------

This code implements the algorithm described in

   Knöppel, Crane, Pinkall, Schröder, "Stripe Patterns on Surfaces" (ACM Transactions on Graphics 2015)

It is not particularly well-written, well-documented, or well-optimized, but it does the right thing.
The basic function of this  program is to compute a pattern of evenly-spaced stripes aligned with a
given direction field.  In principal the direction field could be computed using any method, but for
convenience the code implements a couple techniques that make it easy to generate a nice direction
field on pretty much any surface.  In particular, it implements variants of the algorithms described in

   1. Knöppel, Crane, Pinkall, Schröder, "Globally Optimal Direction Fields" (ACM Transactions on Graphics 2013)
   2. Crane, Desbrun, and Schröder, "Trivial Connections on Discrete Surfaces" (SGP 2010)

The first algorithm is used to initialize the field with either (a) the smoothest possible line field on
the surface or (b) a field aligned with principal curvature directions.  The second method allows one to
manually change the placement of singularities in the initial field, if desired.  The stripe pattern is
updated interactively with changes in the field.


BUILDING
======================================================================================================

The code has two dependencies:

   1. the SuiteSparse library for numerical linear algebra (http://faculty.cse.tamu.edu/davis/suitesparse.html), and
   2. OpenGL/GLUT.

Both of these libraries are quite standard, but building and linking them can vary quite a bit on different platforms.  The current Makefile assumes that the platform is Mac OS X, and that SuiteSparse has been installed via Macports.  But modifying the Makefile for other platforms should not be difficult, and there are some (commented out) lines that may be helpful for Linux or Windows/Cygwin.  Alternatively, if you want to use your own linear algebra library (like Eigen), you can simply change the implementation of methods like solvePositiveDefinite() in DDG::SparseMatrix and DDG::DenseMatrix, which currently serve as wrappers around SuiteSparse.  OpenGL and GLUT should be available by default on many platforms.

Once the dependencies have been installed/built, the code can be built by simply typing

   ./make

in the root directory.  The result will be an executable called “stripe” (or possibly “stripe.exe” on Windows).


RUNNING
======================================================================================================

To run the code from the command line, type

   ./stripe input.obj

where “input.obj” is a path to a Wavefront OBJ file.  The mesh must be connected and manifold, possibly with boundary.  You will then be presented with a window displaying your mesh; hitting the spacebar will automatically generate a stripe pattern.  Further commands are listed below.




USER INTERFACE
======================================================================================================

SPACE - compute a stripe pattern aligned with the globally smoothest direction field
c     - compute a stripe pattern aligned with the minimum principal curvature direction
d     - toggle display of input direction field
s     - set draw mode to smooth shaded
f     - set draw mode to wireframe
w     - write mesh to the file "out.obj"; the stripe pattern is stored in the texture coordinates
1     - compute just a 1D stripe pattern
2     - compute two orthogonal coordinates to get a 2D parameterization (not visualized, but will be saved to disk)
e     - edit the input direction field
*     - toggle singularities
-     - decrease stripe frequency
+     - increase stripe frequency
(     - rotate field left
)     - rotate field right
TAB   - animate rotating stripe pattern
r     - reload the input mesh from disk
\     - write a screenshot to the frames/ directory
ESC   - exit

ALT-CLICK: in edit mode, alt-clicking on a pair of triangles will adjust the singularities; in particular:
   --clicking on a pair of nonsingular triangles will create an equal and opposite pair
   --clicking on an equal and opposite pair will remove both singularities
   --clicking on a singularity and a nonsingular triangle will move the singularity


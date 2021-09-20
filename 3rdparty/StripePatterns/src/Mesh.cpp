#include <map>
#include <fstream>
#include <array>
#include <queue>
#include <Mesh.h>
#include <MeshIO.h>
#include <Utility.h>

using namespace std;
using namespace DDGConstants;

namespace DDG
{
   Mesh :: Mesh( void )
   : fieldDegree( 2 ),  // line field, cross field, etc.---one can change this to compute n-direction fields for n other than 2, but currently parameterization only works for n=2
     lambda( 130. ), // initial global line frequency
     nCoordinateFunctions( 1 )
   {}
   
   Mesh :: Mesh( const Mesh& mesh )
   {
      *this = mesh;
   }
   
   class  HalfEdgeIterCompare { public: bool operator()( const  HalfEdgeIter& i, const  HalfEdgeIter& j ) const { return &*i < &*j; } };
   class HalfEdgeCIterCompare { public: bool operator()( const HalfEdgeCIter& i, const HalfEdgeCIter& j ) const { return &*i < &*j; } };
   class    VertexIterCompare { public: bool operator()( const    VertexIter& i, const    VertexIter& j ) const { return &*i < &*j; } };
   class   VertexCIterCompare { public: bool operator()( const   VertexCIter& i, const   VertexCIter& j ) const { return &*i < &*j; } };
   class      FaceIterCompare { public: bool operator()( const      FaceIter& i, const      FaceIter& j ) const { return &*i < &*j; } };
   class     FaceCIterCompare { public: bool operator()( const     FaceCIter& i, const     FaceCIter& j ) const { return &*i < &*j; } };
   class      EdgeIterCompare { public: bool operator()( const      EdgeIter& i, const      EdgeIter& j ) const { return &*i < &*j; } };
   class     EdgeCIterCompare { public: bool operator()( const     EdgeCIter& i, const     EdgeCIter& j ) const { return &*i < &*j; } };
   
   const Mesh& Mesh :: operator=( const Mesh& mesh )
   {
      map< HalfEdgeCIter, HalfEdgeIter, HalfEdgeCIterCompare > halfedgeOldToNew;
      map<   VertexCIter,   VertexIter,   VertexCIterCompare >   vertexOldToNew;
      map<     EdgeCIter,     EdgeIter,     EdgeCIterCompare >     edgeOldToNew;
      map<     FaceCIter,     FaceIter,     FaceCIterCompare >     faceOldToNew;
   
      // copy geometry from the original mesh and create a
      // map from pointers in the original mesh to
      // those in the new mesh
      halfedges.clear(); for( HalfEdgeCIter he = mesh.halfedges.begin(); he != mesh.halfedges.end(); he++ ) halfedgeOldToNew[ he ] = halfedges.insert( halfedges.end(), *he );
       vertices.clear(); for(   VertexCIter  v =  mesh.vertices.begin();  v !=  mesh.vertices.end();  v++ )   vertexOldToNew[ v  ] =  vertices.insert(  vertices.end(), *v  );
          edges.clear(); for(     EdgeCIter  e =     mesh.edges.begin();  e !=     mesh.edges.end();  e++ )     edgeOldToNew[ e  ] =     edges.insert(     edges.end(), *e  );
          faces.clear(); for(     FaceCIter  f =     mesh.faces.begin();  f !=     mesh.faces.end();  f++ )     faceOldToNew[ f  ] =     faces.insert(     faces.end(), *f  );
   
      // "search and replace" old pointers with new ones
      for( HalfEdgeIter he = halfedges.begin(); he != halfedges.end(); he++ )
      {
         he->next   = halfedgeOldToNew[ he->next   ];
         he->flip   = halfedgeOldToNew[ he->flip   ];
         he->vertex =   vertexOldToNew[ he->vertex ];
         he->edge   =     edgeOldToNew[ he->edge   ];
         he->face   =     faceOldToNew[ he->face   ];
      }
   
      for( VertexIter v = vertices.begin(); v != vertices.end(); v++ ) v->he = halfedgeOldToNew[ v->he ];
      for(   EdgeIter e =    edges.begin(); e !=    edges.end(); e++ ) e->he = halfedgeOldToNew[ e->he ];
      for(   FaceIter f =    faces.begin(); f !=    faces.end(); f++ ) f->he = halfedgeOldToNew[ f->he ];
   
      return *this;
   }

   int Mesh::read( const string& filename )
   {
      inputFilename = filename;
      ifstream in( filename.c_str() );

      if( !in.is_open() )
      {
         cerr << "Error reading from mesh file " << filename << endl;
         return 1;
      }

      int rval;
      if( !( rval = MeshIO::read( in, *this )))
      {
         indexElements();
         initializeSmoothStructure();
         buildMassMatrix();
      }
      return rval;
   }

   int Mesh::write( const string& filename )
   // reads a mesh from a Wavefront OBJ file; return value is nonzero
   // only if there was an error
   {
      ofstream out( filename.c_str() );

      // if we computed two orthogonal coordinate functions
      // (instead of just the 1D parameterization used to
      // draw stripes), identify edges in parameter space to
      // get a coherent (i.e., continuous almost everywhere)
      // parameterization
      if( nComputedCoordinateFunctions == 2 )
      {
         glueParameterization();
      }

      if( !out.is_open() )
      {
         cerr << "Error writing to mesh file " << filename << endl;
         return 1;
      }

      MeshIO::write( out, *this );

      return 0;
   }

   void Mesh :: indexElements( void )
   {
      int nV = 0;
      for( VertexIter v = vertices.begin(); v != vertices.end(); v++ )
      {
         v->index = nV;
         nV++;
      }

      int nE = 0;
      for( EdgeIter e = edges.begin(); e != edges.end(); e++ )
      {
         e->index = nE;
         nE++;
      }

      int nF = 0;
      for( FaceIter f = faces.begin(); f != faces.end(); f++ )
      {
         f->index = nF;
         nF++;
      }
   }

   bool Mesh::reload( void )
   {
      return read( inputFilename );
   }

   void Mesh::normalize( void )
   {
      // compute center of mass
      Vector c( 0., 0., 0. );
      for( VertexCIter v = vertices.begin(); v != vertices.end(); v++ )
      {
         c += v->position;
      }
      c /= (double) vertices.size();

      // translate to origin
      for( VertexIter v = vertices.begin(); v != vertices.end(); v++ )
      {
         v->position -= c;
      }

      // rescale such that the mesh sits inside the unit ball
      double rMax = 0.;
      for( VertexCIter v = vertices.begin(); v != vertices.end(); v++ )
      {
         rMax = max( rMax, v->position.norm() );
      }
      for( VertexIter v = vertices.begin(); v != vertices.end(); v++ )
      {
         v->position /= rMax;
      }
   }

   int Mesh :: eulerCharacteristic( void ) const
   {
      int nV = vertices.size();
      int nE = edges.size();
      int nF = faces.size();

      return nV - nE + nF;
   }

   void Mesh :: initializeSmoothStructure( void )
   {
      // compute angular coordinates of each outgoing halfedge
      for( VertexIter v  = vertices.begin();
                      v != vertices.end();
                      v ++ )
      {
         // compute the cumulative angle at each outgoing
         // halfedge, relative to the initial halfedge
         double cumulativeAngle = 0.;
         HalfEdgeIter he = v->he;

         // JP: for vertices on the boundary, v->he is always the
         // outward-pointing boundary halfedge (i.e., actually outside the mesh),
         // so the first step of the ccw circulation used in the non-boundary
         // case will accumulate an undefined angle. In fact, the first step
         // doesn't even circulate around v since he->next for a
         // boundary half-edge is actually the next boundary half-edge in the
         // chain (not the next half-edge in the ghost "boundary triangle"
         // he->face)....
         // We correct this by circulating clockwise and accumulating a
         // negative angle.
         if (he->onBoundary) {
            do {
                he->angularCoordinate = cumulativeAngle;
                he = he->flip->next;
                cumulativeAngle -= he->next->angle();
                if (cumulativeAngle < 0) cumulativeAngle += 2 * M_PI;
            }
            while( he != v->he );
         }
         else {
             do {
                he->angularCoordinate = cumulativeAngle;
                cumulativeAngle += he->next->angle();
                he = he->next->next->flip;
             }
             while( he != v->he );

             // normalize angular coordinates so that they sum to two pi
             do
             {
                he->angularCoordinate *= 2.*M_PI/cumulativeAngle;
                he = he->flip->next;
             }
             while( he != v->he );
         }
      }
   }

   void Mesh :: buildMassMatrix( void )
   {
      int nV = vertices.size();
      massMatrix.resize( nV, nV );
      realMassMatrix.resize( 2*nV, 2*nV );

      for( VertexCIter v = vertices.begin(); v != vertices.end(); v++ )
      {
         int i = v->index;
         double A = v->dualArea();
         massMatrix( i, i ) = A;
         realMassMatrix( i*2+0, i*2+0 ) = A;
         realMassMatrix( i*2+1, i*2+1 ) = A;
      }
   }

   void Mesh :: buildFieldEnergy( void )
   // build per-face
   {
      double k = fieldDegree;

      SparseMatrix<Complex>& A( energyMatrix );
      int nV = vertices.size();
      A.resize( nV, nV );
      for( FaceCIter f = faces.begin(); f != faces.end(); f++ )
      {
         if( f->isBoundary() ) continue;

         HalfEdgeCIter he = f->he;
         do
         {
            int i = he->vertex->index;
            int j = he->flip->vertex->index;
            double w = he->cotan() / 2.;
            double thetaI = he->angularCoordinate;
            double thetaJ = he->flip->angularCoordinate;
            double phi = k*( thetaI - thetaJ + M_PI );
            Complex r( cos(phi), sin(phi) );

            A( i, i ) += w;
            A( i, j ) -= w*r;

            A( j, j ) += w;
            A( j, i ) -= w*r.inv();

            he = he->next;
         }
         while( he != f->he );
      }
      
      // Some domains will admit a trivial section (e.g., a flat disk),
      // hence we shift to make the matrix strictly positive-definite for
      // the Cholesky solver.  Note, however, that a constant shift will
      // not change the eigenvectors of the matrix, hence we get exactly
      // the same solution.  I.e., it is only the eigenvectors (and not the
      // eigenvalues) that are used in the end.
      A.shift( 1e-4 );
   }

   void Mesh :: buildDualLaplacian( void )
   {
      int nF = faces.size();
      SparseMatrix<Real>& L( dualLaplacian );
      L = SparseMatrix<Real>( nF, nF );

      for( FaceCIter f = faces.begin(); f != faces.end(); f++ )
      {
         int i = f->index;

         HalfEdgeCIter h = f->he;
         do
         {
            int j = h->flip->face->index;
            double wij = 1.; // 2. / ( cotAlpha + cotBeta );

            L( i, i ) += wij;
            L( i, j ) -= wij;

            h = h->next;
         }
         while( h != f->he );
      }

      L.shift( 1e-10 );
   }

   void Mesh :: computeTrivialSection( void )
   {
      // store previous section
      for( VertexIter v = vertices.begin(); v != vertices.end(); v++ )
      {
         v->oldDirectionField = v->directionField;
      }

      // solve for scalar potential
      buildDualLaplacian();
      int nV = vertices.size();
      int nE = edges.size();
      int nF = faces.size();
      int chi = nV - nE + nF;
      DenseMatrix<Real> Omega( nF );
      double indexSum = 0.;
      for( FaceCIter f = faces.begin(); f != faces.end(); f++ )
      {
         Omega( f->index ) = -f->curvature() + 2.*M_PI*f->singularIndex;
         indexSum += f->singularIndex;
      }
      if( indexSum != chi )
      {
         // Fact of life: you can't comb the hair on a billiard ball...
         cerr << "WARNING: singularity indices do not satisfy Poincaré-Hopf!" << endl;
      }

      // extract connection 1-form
      DenseMatrix<Real> u( nF );
      solvePositiveDefinite( dualLaplacian, u, Omega );

      for( EdgeIter e = edges.begin(); e != edges.end(); e++ )
      {
         int i = e->he->face->index;
         int j = e->he->flip->face->index;
         e->omega = u(j) - u(i);
      }
      
      // construct parallel section
      for( VertexIter v = vertices.begin(); v != vertices.end(); v++ )
      {
         v->visited = false;
      }
      vertices.begin()->directionField = Complex( cos(fieldOffsetAngle), sin(fieldOffsetAngle) );
      vertices.begin()->visited = true;
      queue<VertexIter> Q;
      Q.push( vertices.begin() );
      while( !Q.empty() )
      {
         VertexIter vi = Q.front(); Q.pop();
         HalfEdgeIter h = vi->he;
         do
         {
            VertexIter vj = h->flip->vertex;
            if( !vj->visited )
            {

               double thetaI = h->angularCoordinate;
               double thetaJ = h->flip->angularCoordinate + M_PI;
               double dTheta = thetaJ - thetaI;
               Complex rij( cos(dTheta), sin(dTheta) );

               double omegaIJ = -h->omega();
               Complex sij( cos(omegaIJ), sin(omegaIJ) );

               Complex Xi = vi->directionField;
               Complex Xj = rij*sij*Xi;

               vj->directionField = Xj;
               vj->visited = true;
               Q.push( vj );
            }

            h = h->flip->next;
         }
         while( h != vi->he );
      }
      for( VertexIter v = vertices.begin(); v != vertices.end(); v++ )
      {
         v->directionField *= v->directionField;
      }
   }

   void Mesh :: alignTrivialSection( void )
   // Suppose we take the singularities of the globally smoothest section and construct the
   // corresponding trivial connection.  A section parallel with respect to this connection
   // will resemble the smoothest section only up to a constant rotation in each tangent
   // space.  This method looks for the "best fit" rotation.
   {
      // compute the mean angle difference between the old and new section
      Complex mean( 0., 0. );
      for( VertexIter v = vertices.begin(); v != vertices.end(); v++ )
      {
         Complex z0 = v->oldDirectionField;
         Complex z1 = v->directionField;
         Complex w = z1.inv()*z0;
         mean += w;
      }
      mean.normalize();

      // align 
      for( VertexIter v = vertices.begin(); v != vertices.end(); v++ )
      {
         v->directionField *= mean;
      }

      fieldOffsetAngle = vertices.begin()->directionField.arg()/2.;
   }

   void Mesh :: computeSmoothestSection( void )
   {
      cout << "Computing globally smoothest direction field..." << endl;
      srandom( 1234325 );

      buildFieldEnergy();

      int nV = vertices.size();
      DenseMatrix<Complex> groundState( nV );
      smallestEigPositiveDefinite( energyMatrix, massMatrix, groundState );
      
      for( VertexIter v = vertices.begin(); v != vertices.end(); v++ )
      {
         v->directionField = groundState( v->index );
      }
   }

   void Mesh :: computeCurvatureAlignedSection( void )
   {
      cerr << "Computing curvature-aligned direction field..." << endl;

      buildFieldEnergy();

      int nV = vertices.size();
      DenseMatrix<Complex> principalField( nV );
      DenseMatrix<Complex> smoothedField( nV );

      for( VertexCIter v = vertices.begin(); v != vertices.end(); v++ )
      {
         principalField( v->index ) = v->principalDirection().unit();
      }

      SparseMatrix<Complex> A;
      const double t = 1.;
      A = energyMatrix + Complex(t)*massMatrix;

      solvePositiveDefinite( A, smoothedField, principalField );
      
      for( VertexIter v = vertices.begin(); v != vertices.end(); v++ )
      {
         v->directionField = smoothedField( v->index ).unit();
      }
   }

   void Mesh :: parameterize( void )
   {
      // compute the first coordinate
      computeParameterization( 0 );

      // at the user's request, we can also compute a second coordinate
      // aligned with the orthogonal direction field---these two coordinates
      // together describe a 2D parameterization rather than a 1D parameterization
      if( nCoordinateFunctions == 2 )
      {
         // rotate 1-vector field by 90 degrees, which in our complex representation
         // is equivalent to rotating a 2-vector field by 180 degrees
         for( VertexIter v = vertices.begin(); v != vertices.end(); v++ )
         {
            v->directionField = -v->directionField;
         }

         // compute the second coordinate
         computeParameterization( 1 );

         // rotate back
         for( VertexIter v = vertices.begin(); v != vertices.end(); v++ )
         {
            v->directionField = -v->directionField;
         }
      }

      // keep track of how many coordinates are actually valid
      nComputedCoordinateFunctions = nCoordinateFunctions;
   }

   void Mesh :: buildEnergy( SparseMatrix<Real>& A, int /* coordinate */ )
   {
      int nV = vertices.size();
      A.resize( 2*nV, 2*nV );

      for( EdgeIter e = edges.begin(); e != edges.end(); e++ )
      {
         // get the endpoints
         VertexCIter vi = e->he->vertex;
         VertexCIter vj = e->he->flip->vertex;

         // get the angle of the edge w.r.t. the endpoints' bases
         double thetaI = e->he->angularCoordinate;
         double thetaJ = e->he->flip->angularCoordinate + M_PI;

         // compute the parallel transport coefficient from i to j
         double dTheta = thetaJ - thetaI;
         Complex rij( cos(dTheta), sin(dTheta) );

         // compute the cotan weight
         double cotAlpha = e->he->cotan();
         double cotBeta  = e->he->flip->cotan();
         if(       e->he->face->fieldIndex(2.) != 0 ) cotAlpha = 0.;
         if( e->he->flip->face->fieldIndex(2.) != 0 ) cotBeta  = 0.;
         double w = (cotAlpha+cotBeta)/2.;

         // pick an arbitrary root at each endpoint
         Complex Xi = vi->canonicalVector();
         Complex Xj = vj->canonicalVector();

         // check if the roots point the same direction
         double s = dot( rij*Xi, Xj ) > 0. ? 1. : -1.;
         if( fieldDegree == 1 ) s = 1.;
         e->crossesSheets = ( s < 0. );

         // compute the 1-form value along edge ij
         double lij = e->length();
         double phiI = (Xi).arg();
         double phiJ = (s*Xj).arg();
         double omegaIJ = lambda * (lij/2.) * ( cos(phiI-thetaI) + cos(phiJ-thetaJ) );

         e->omega = omegaIJ;

         // compute the components of the new transport coefficient
         double a = w * cos(omegaIJ);
         double b = w * sin(omegaIJ);

         int i = 2 * vi->index;
         int j = 2 * vj->index;

         // add the diagonal terms
         A(i+0,i+0) += w;
         A(i+1,i+1) += w;

         A(j+0,j+0) += w;
         A(j+1,j+1) += w;

         // if both vectors pointed the same direction, use a
         // 2x2 block that represents complex multiplication
         if( s > 0. )
         {
            A(i+0,j+0) = -a; A(i+0,j+1) = -b;
            A(i+1,j+0) =  b; A(i+1,j+1) = -a;

            A(j+0,i+0) = -a; A(j+0,i+1) =  b;
            A(j+1,i+0) = -b; A(j+1,i+1) = -a;
         }
         // otherwise, use a block that represents both
         // complex conjugation and multiplication
         else
         {
            A(i+0,j+0) = -a; A(i+0,j+1) =  b;
            A(i+1,j+0) =  b; A(i+1,j+1) =  a;

            A(j+0,i+0) = -a; A(j+0,i+1) =  b;
            A(j+1,i+0) =  b; A(j+1,i+1) =  a;
         }
      }

#if 0 // Debugging
      {
          std::ofstream outFile("angularCoordinates.txt");
          for (const auto &he : halfedges) {
              outFile << he.angularCoordinate << "\t" << (he.onBoundary || he.flip->onBoundary) << std::endl;
          }
      }

      {
          size_t fidx = 76020; // inspect a bad face.
          auto f = faces[fidx];

          std::array<HalfEdgeIter, 3> HEs{{ f.he, f.he->next, f.he->next->next }};
          std::array<VertexIter,   3> Vs{{ HEs[0]->vertex,  HEs[1]->vertex,  HEs[2]->vertex }};

          std::cout << "face " << fidx << " half edges on boundary: ";
          for (auto he : HEs) std::cout << he->flip->onBoundary;
          std::cout << std::endl;

          std::cout << "face " << fidx << " vertices on boundary: ";
          for (auto v : Vs) std::cout << v->onBoundary();
          std::cout << std::endl;

          std::cout << "Crosses sheet: ";
          for (auto he : HEs) std::cout << he->edge->crossesSheets;
          std::cout << std::endl;
          std::cout << "Canonical vectors: " << std::endl;
          for (auto v : Vs) {
              std::cout << v->canonicalVector() << std::endl;
          }
          std::cout << "global canonical vectors: " << std::endl;
          for (auto v : Vs) {
              auto n = v->normal();
              auto ref_v = v->he->vector();
              ref_v = (ref_v - n * dot(n, ref_v)).unit();
              std::cout << (v->canonicalVector() * DDG::Complex(ref_v[0], ref_v[1])) << std::endl;
          }

          std::cout << "Edge quantities: " << std::endl;
          for (auto he : HEs) {
             // get the angle of the edge w.r.t. the endpoints' bases
             double thetaI = he->angularCoordinate;
             double thetaJ = he->flip->angularCoordinate + M_PI;
             std::cout << thetaI << "\t" << thetaJ << "\t";

             VertexCIter vi = he->vertex;
             VertexCIter vj = he->flip->vertex;

             Complex Xi = vi->canonicalVector();
             Complex Xj = vj->canonicalVector();

             // compute the parallel transport coefficient from i to j
             double dTheta = thetaJ - thetaI;
             Complex rij( cos(dTheta), sin(dTheta) );
             std::cout << dot(rij*Xi, Xj);
             std::cout << std::endl;
          }
          std::cout << std::endl;
      }
#endif

      A.shift( 1e-4 );
   }

   void Mesh :: computeParameterization( int coordinate )
   {
      cerr << "Computing stripe pattern..." << endl;

      //srandom( time( NULL ) );
      srandom( 1234567 );

      SparseMatrix<Real> A;
      buildEnergy( A, coordinate );

      int nV = vertices.size();
      DenseMatrix<Real> groundState( 2.*nV );
      smallestEigPositiveDefinite( A, realMassMatrix, groundState );

      for( VertexIter v = vertices.begin(); v != vertices.end(); v++ )
      {
         int i = v->index;
         v->parameterization = Complex( groundState( i*2+0 ),
                                        groundState( i*2+1 ) ).unit();
      }

      assignTextureCoordinates( coordinate );
   }

   double Mesh :: energy( const SparseMatrix<Real>& A, const DenseMatrix<Real>& x, double eps )
   {
      // evaluate quadratic part of energy
      double mu = inner( A*x, x );

      // evaluate quartic part of energy
      int nV = vertices.size();
      for( VertexCIter v = vertices.begin(); v != vertices.end(); v++ )
      {
         if( v->index != nV-1 )
         {
            Complex psi = v->parameterization;
            mu += eps * sqr( psi.norm2() - 1. ) / 4.;
         }
      }

      return mu;
   }
   
   void Mesh :: assignTextureCoordinates( int p )
   // This method computes the final texture coordinates that are actually
   // used by OpenGL to draw the stripes, starting with the solution to
   // the eigenvalue problem.
   {
      for( FaceIter f = faces.begin(); f != faces.end(); f++ )
      {
         if( f->isBoundary() ) continue;

         // grab the halfedges
         HalfEdgeIter hij = f->he;
         HalfEdgeIter hjk = hij->next;
         HalfEdgeIter hki = hjk->next;

         // grab the parameter values at vertices
         Complex psiI = hij->vertex->parameterization;
         Complex psiJ = hjk->vertex->parameterization;
         Complex psiK = hki->vertex->parameterization;
         
         double cIJ = ( hij->edge->he != hij ? -1. : 1. );
         double cJK = ( hjk->edge->he != hjk ? -1. : 1. );
         double cKI = ( hki->edge->he != hki ? -1. : 1. );

         // grab the connection coeffients
         double omegaIJ = cIJ * hij->edge->omega;
         double omegaJK = cJK * hjk->edge->omega;
         double omegaKI = cKI * hki->edge->omega;

         if( hij->edge->crossesSheets )
         {
            psiJ = psiJ.bar();
            omegaIJ =  cIJ * omegaIJ;
            omegaJK = -cJK * omegaJK;
         }

         if( hki->edge->crossesSheets )
         {
            psiK = psiK.bar();
            omegaKI = -cKI * omegaKI;
            omegaJK =  cJK * omegaJK;
         }

         // construct complex transport coefficients
         Complex rij( cos(omegaIJ), sin(omegaIJ) );
         Complex rjk( cos(omegaJK), sin(omegaJK) );
         Complex rki( cos(omegaKI), sin(omegaKI) );

         // compute the angles at the triangle corners closest to the target omegas
         double alphaI = psiI.arg();
         double alphaJ = alphaI + omegaIJ - (rij*psiI/psiJ).arg(); //fmodPI((varphiI + omegaIJ) - varphiJ); // could do this in terms of angles instead of complex numbers...
         double alphaK = alphaJ + omegaJK - (rjk*psiJ/psiK).arg(); //fmodPI((varphiJ + omegaJK) - varphiK); // mostly a matter of taste---possibly a matter of performance?
         double alphaL = alphaK + omegaKI - (rki*psiK/psiI).arg(); //fmodPI((varphiK + omegaKI) - varphiI);

         // // JP debugging: assign angles to minimize variation regardless of omega
         // double alphaI = psiI.arg();
         // double alphaJ = alphaI + fmodPI(psiJ.arg() - alphaI);
         // double alphaK = alphaJ + fmodPI(psiK.arg() - alphaJ);
         // double alphaL = alphaI;

         // adjust triangles containing zeros
         double n = lround((alphaL-alphaI)/(2.*M_PI));
         alphaJ -= 2.*M_PI*n/3.;
         alphaK -= 4.*M_PI*n/3.;

         // store the coordinates
         hij->texcoord[p] = alphaI;
         hjk->texcoord[p] = alphaJ;
         hki->texcoord[p] = alphaK;
         f->paramIndex[p] = n;
      }
   }

   void Mesh :: glueParameterization( void )
   // Running this method is completely optional - if you compute two orthogonal
   // stripe coordinates, it simply glues together the triangles in the parameter
   // domain so that they describe a continuous parameterization (otherwise, each
   // triangle can be different from its neighbor by appropriate reflections/
   // rotations/translations).  It is not usually needed to visualize the stripe
   // patterns.  But in some rendering packages, for certain effects like
   // displacement mapping, it can reduce rendering artifacts to have a continuous
   // parameterization, rather than one where each triangle is its own little island.
   {
      queue<FaceIter> Q;

      // start at an arbitrary nonsingular face
      FaceIter f0 = faces.begin();
      while( f0->paramIndex[0] != 0. ||
             f0->paramIndex[1] != 0. ||
             f0->fieldIndex(2.) != 0. )
      {
         f0++;
      }
      f0->visited = true;
      Q.push( f0 );

      while( !Q.empty() )
      {
         FaceIter f = Q.front(); Q.pop();

         HalfEdgeIter he = f->he;
         do
         {
            FaceIter fj = he->flip->face;
            // traverse neighbors only if they're nonsingular
            if( !f->isBoundary() &&
                !fj->visited &&
                fj->paramIndex[0]  == 0. &&
                fj->paramIndex[1]  == 0. &&
                fj->fieldIndex(2.) == 0. )
            {
               // grab handles to the three neighboring vertex coordinates
               Complex& bi = he->flip->next->texcoord;
               Complex& bj = he->flip->texcoord;
               Complex& bk = he->flip->next->next->texcoord;

               // compute the two parameter-space vectors along the shared edge
               Complex ai = he->texcoord;
               Complex aj = he->next->texcoord;
               Complex u = aj-ai;
               Complex v = bj-bi;

               // compute the rotation between these two edges
               double theta = (u*v.inv()).arg();
               Complex z( cos(theta), sin(theta) );

               // if the angle is anything other than 0 or 180...
               if( fabs(z.im) > 1e-9 )
               {
                  // ...apply a reflection to the neighboring triangle
                  bi=bi.bar();
                  bj=bj.bar();
                  bk=bk.bar();
               }

               // now recompute the rotation
               v = bj-bi;
               theta = (u*v.inv()).arg();
               z = Complex( cos(theta), sin(theta) );

               // as long as we now have a valid rotation...
               if( fabs(z.im) < 1e-9 )
               {
                  // ...rotate and translate the neighbor so that
                  // the shared edge matches up with the parent
                  Complex b0 = bi;
                  bi = z*(bi-b0) + ai;
                  bj = z*(bj-b0) + ai;
                  bk = z*(bk-b0) + ai;
               }

               // enqueue the neighbor
               fj->visited = true;
               Q.push( fj );
            }

            he = he->next;
         }
         while( he != f->he );
      }
   }

   void Mesh :: extractSingularities( void )
   {
      for( FaceIter f = faces.begin(); f != faces.end(); f++ )
      {
         f->singularIndex = f->fieldIndex( fieldDegree ) / (double) fieldDegree;
      }
   }
}


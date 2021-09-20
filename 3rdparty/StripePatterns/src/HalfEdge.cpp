#include <HalfEdge.h>
#include <Mesh.h>
#include <limits>

namespace DDG
{
   HalfEdgeIter HalfEdge :: self( void )
   {
      return flip->flip;
   }

   HalfEdgeCIter HalfEdge :: self( void ) const
   {
      return flip->flip;
   }

   bool HalfEdge :: isCanonical( void ) const
   {
      return edge->he == self();
   }

   bool HalfEdge :: crossesSheets( void ) const
   {
      return edge->crossesSheets;
   }

   Vector HalfEdge :: vector( void ) const
   {
      return flip->vertex->position - vertex->position;
   }

   double HalfEdge :: angle( void ) const
   {
      Vector a = next->next->vertex->position;
      Vector b = vertex->position;
      Vector c = next->vertex->position;

      Vector u = (b-a).unit();
      Vector v = (c-a).unit();

      return acos( max( -1., min( 1., dot( u, v ))));
   }

   double HalfEdge :: cotan( void ) const
   {
      if( onBoundary ) return 0.;

      Vector a = next->next->vertex->position;
      Vector b = vertex->position;
      Vector c = next->vertex->position;

      Vector u = b-a;
      Vector v = c-a;

      const double epsilon = 1e-7;
      double cotTheta = dot( u, v ) / cross( u, v ).norm();
      if( cotTheta < epsilon )
      {
         cotTheta = epsilon;
      }
      return cotTheta;
   }

   void HalfEdge :: updateTexCoord( int p )
   {
      const double infty = numeric_limits<double>::max();
      HalfEdgeCIter hij;

      if( next->texcoord[p] != infty )
      {
         hij = flip;
         updateTexCoordFromIToJ( p, hij );
      }
      else if( next->next->texcoord[p] != infty )
      {
         hij = next->next;
         updateTexCoordFromIToJ( p, hij );
      }
      else
      {
         texcoord[p] = vertex->parameterization.arg();
      }
   }

   void HalfEdge :: updateTexCoordFromIToJ( int p, HalfEdgeCIter hij )
   {
      Complex phiI = hij->vertex->parameterization;
      Complex phiJ = vertex->parameterization;
      double omegaIJ = hij->edge->omega;

      if( hij->edge->crossesSheets )
      {
         phiJ = phiJ.bar();
      }
      else if( hij->edge->he == hij )
      {
         omegaIJ = -omegaIJ;
      }

      Complex rij( cos(omegaIJ), sin(omegaIJ) );
      texcoord[p] = phiI.arg() + ( (rij*phiI).inv()*(phiJ) ).arg() + omegaIJ;
   }

   double HalfEdge :: omega( void ) const
   {
      if( isCanonical() )
      {
         return edge->omega;
      }
      return -edge->omega;
   }
}


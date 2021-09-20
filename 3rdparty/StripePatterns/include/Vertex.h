// -----------------------------------------------------------------------------
// libDDG -- Vertex.h
// -----------------------------------------------------------------------------
//
// Vertex stores attributes associated with a mesh edge.  The iterator he
// points to its "outgoing" halfedge.  (See the documentation for a more
// in-depth discussion of the halfedge data structure.)
// 

#ifndef DDG_VERTEX_H
#define DDG_VERTEX_H

#include "Complex.h"
#include "Vector.h"
#include "Types.h"

namespace DDG
{
   class Vertex
   {
      public:
         HalfEdgeIter he;
         // points to the "outgoing" halfedge

         Vector position;
         // location of vertex in Euclidean 3-space

         Vector normal( void ) const;
         // returns the vertex normal

         double dualArea( void ) const;
         // returns the barycentric dual area, equal to one-third
         // the sum of the areas of all incident triangles

         bool onBoundary( void ) const;
         // returns true if the vertex is contained in the domain boundary

         bool isIsolated( void ) const;
         // returns true if the vertex is not contained in any face or edge; false otherwise

         int valence( void ) const;
         // returns the number of incident faces / edges

         double angleSum( void ) const;
         // returns sum of interior angles incident on this vertex

         int index;
         // unique integer ID in the range [0,nVertices-1]

         Vector fieldVector( double fieldDegree, double whichVector ) const;
         // vector representing direction field in world coordinates

         Complex canonicalVector( void ) const;
         // canonical choice of one of the 2-vectors

         Complex principalDirection( void ) const;
         // principal curvature direction, expressed as an intrinsic tangent vector relative to the outgoing halfedge

         Complex directionField, oldDirectionField;
         Complex parameterization;
         Complex embedding;

         double constantValue;
         bool visited;
         // for traversal

         Vector tangent;
   };
}

#endif


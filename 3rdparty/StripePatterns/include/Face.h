// -----------------------------------------------------------------------------
// libDDG -- Face.h
// -----------------------------------------------------------------------------
//
// Face stores attributes associated with a mesh edge.  The iterator he points
// to one of its associated halfedges.  (See the documentation for a more
// in-depth discussion of the halfedge data structure.)
// 

#ifndef DDG_FACE_H
#define DDG_FACE_H

#include "Types.h"
#include <vector>

namespace DDG
{
   class Face
   {
      public:
         HalfEdgeIter he;
         // points to one of the halfedges associated with this face

         bool visited;
         // flag for traversal

         bool isBoundary( void ) const;
         // returns true if this face corresponds to a
         // boundary loop; false otherwise

         Vector normal( void ) const;
         // returns the unit normal associated with this face; normal
         // orientation is determined by the circulation order of halfedges

         double curvature( void ) const;
         // returns the intrinsic curvature, integrated over this triangle

         double fieldIndex( double fieldDegree ) const;
         // returns the degree of the direction field around this triangle

         int index;
         // unique ID in range [0,nF-1]

         double singularIndex;
         // index for trivial connection

         double paramIndex[2];
         // degree of the parameterization around this triangle, for each
         // coorinate function (computed by Mesh::assignTextureCoordinates)

         double area( void ) const;
         // returns the triangle area

         Vector barycenter( void ) const;
         // returns arithmetic mean of vertex coordinates

         void orientTexCoords( void );
         // flip texture coordinates if they are negatively oriented

         void getLocalSheet( std::vector<Complex>& psi, std::vector<double>& omega );
         // returns three parameter values psi and transport coefficients
         // consistent with the canonical sheet at f->he->vertex
   };
}

#endif

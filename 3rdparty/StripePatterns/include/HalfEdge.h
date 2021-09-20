// -----------------------------------------------------------------------------
// libDDG -- HalfEdge.h
// -----------------------------------------------------------------------------
//
// HalfEdge is used to define mesh connectivity.  (See the documentation for a
// more in-depth discussion of the halfedge data structure.)
// 

#ifndef DDG_HALFEDGE_H
#define DDG_HALFEDGE_H

#include "Vector.h"
#include "Types.h"
#include "Complex.h"
#include "Utility.h"

namespace DDG
{
   class HalfEdge
   {
      public:
         HalfEdgeIter next;
         // points to the next halfedge around the current face

         HalfEdgeIter flip;
         // points to the other halfedge associated with this edge

         VertexIter vertex;
         // points to the vertex at the "tail" of this halfedge

         EdgeIter edge;
         // points to the edge associated with this halfedge

         FaceIter face;
         // points to the face containing this halfedge

         HalfEdgeIter  self( void );
         HalfEdgeCIter self( void ) const;
         // returns an iterator to this halfedge

         bool isCanonical( void ) const;
         // returns true iff this half edge is the canonical halfedge for its associated edge

         bool crossesSheets( void ) const;
         // returns true iff the canonical vectors at the two endpoints have opposite sign

         bool onBoundary;
         // true if this halfedge is contained in a boundary
         // loop; false otherwise

         double height;
         // used to define embedding

         Complex texcoord;
#ifdef SP_FLAT_TORUS
         Complex origTexCoord;
#endif
         // texture coordinates associated with the triangle corner at the
         // "tail" of this halfedge

         double angularCoordinate;
         // angle of this half edge relative to this->vertex->he,
         // normalized by 2π times the angle sum around this->vertex

         Complex connectionCoefficient;
         // apples *half* the rotation from this->vertex to this->flip->vertex

         Vector vector( void ) const;
         // returns the vector along this halfedge

         double angle( void ) const;
         // returns the interior angle of the opposite corner

         double cotan( void ) const;
         // returns the cotangent of the opposite angle

         double omega( void ) const;
         // returns the (properly-oriented) value of the 1-form omega

         void updateTexCoord( int coordinate );
         void updateTexCoordFromIToJ( int coordinate, HalfEdgeCIter hij );
         // methods for computing angular texture coordinates from complex texture coordinates
   };
}

#endif


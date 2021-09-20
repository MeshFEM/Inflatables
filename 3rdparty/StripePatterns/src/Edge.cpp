#include <Edge.h>
#include <Mesh.h>
#include <Utility.h>

namespace DDG
{
   double Edge :: length( void ) const
   {
#ifdef SP_FLAT_TORUS
      return 1.;
#endif
      return ( he->flip->vertex->position - he->vertex->position ).norm();
   }

   double Edge :: dihedralAngle( void ) const
   {
#ifdef SP_FLAT_TORUS
      return 0.;
#endif
      Vector N1 = he->face->normal();
      Vector N2 = he->flip->face->normal();
      Vector e = ( he->flip->vertex->position - he->vertex->position ).unit();

      return atan2( dot(e,cross(N1,N2)), dot(N1,N2) );
   }
}


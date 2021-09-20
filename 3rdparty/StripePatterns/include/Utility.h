#ifndef DDG_UTILITY_H
#define DDG_UTILITY_H

#include <cmath>
#include <cstdlib>
#include "Utility.h"
#include "Complex.h"

#undef SP_FLAT_TORUS
#undef SP_DEBUG

namespace DDG
{
   inline double sqr( double x )
   {
      return x*x;
   }

   inline double sgn( double x )
   {
      if( x > 0. ) return  1.;
      if( x < 0. ) return -1.;
      return 0.;
   }

   inline double unitRand( void )
   {
      const double rRandMax = 1. / (double) RAND_MAX;

      return rRandMax * (double) random();
   }

   inline double randomReal( double minVal, double maxVal )
   {
      return unitRand()*(maxVal-minVal) + minVal;
   }

   inline double seconds( int t0, int t1 )
   {
      return (double)(t1-t0) / (double) CLOCKS_PER_SEC;
   }

   inline double angle( const Vector& u, const Vector& v )
   {
      return acos( std::max( -1., std::min( 1., dot( u.unit(), v.unit() ))));
   }

   inline double fmodPI( const double theta )
   {
      return theta - (2.*M_PI) * floor( (theta+M_PI) / (2.*M_PI) );
   }
}

namespace DDGConstants
{
   static DDG::Complex ii( 0., 1. );
}

#endif

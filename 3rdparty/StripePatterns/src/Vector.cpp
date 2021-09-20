#include <math.h>

#include <Vector.h>
#include <Utility.h>

namespace DDG
{
   Vector :: Vector( void )
   : x( 0. ),
     y( 0. ),
     z( 0. )
   {}
   
   Vector :: Vector( double x0,
                     double y0,
                     double z0 )
   : x( x0 ),
     y( y0 ),
     z( z0 )
   {}
   
   Vector :: Vector( const Vector& v )
   : x( v.x ),
     y( v.y ),
     z( v.z )
   {}

   double& Vector :: operator[]( const int& index )
   {
      return ( &x )[ index ];
   }
   
   const double& Vector :: operator[]( const int& index ) const
   {
      return ( &x )[ index ];
   }
   
   Vector Vector :: operator+( const Vector& v ) const
   {
      return Vector( x + v.x,
                     y + v.y,
                     z + v.z );
   }
   
   Vector Vector :: operator-( const Vector& v ) const
   {
      return Vector( x - v.x,
                     y - v.y,
                     z - v.z );
   }
   
   Vector Vector :: operator-( void ) const
   {
      return Vector( -x,
                     -y,
                     -z );
   }
   
   Vector Vector :: operator*( const double& c ) const
   {
      return Vector( x*c,
                     y*c,
                     z*c );
   }
   
   Vector operator*( const double& c, const Vector& v )
   {
      return v*c;
   }
   
   Vector Vector :: operator/( const double& c ) const
   {
      return (*this) * ( 1./c );
   }
   
   void Vector :: operator+=( const Vector& v )
   {
      x += v.x;
      y += v.y;
      z += v.z;
   }
   
   void Vector :: operator-=( const Vector& v )
   {
      x -= v.x;
      y -= v.y;
      z -= v.z;
   }
   
   void Vector :: operator*=( const double& c )
   {
      x *= c;
      y *= c;
      z *= c;
   }
   
   void Vector :: operator/=( const double& c )
   {
      (*this) *= ( 1./c );
   }
   
   double Vector :: norm( void ) const
   {
      return sqrt( norm2());
   }
   
   double Vector :: norm2( void ) const
   {
      return dot( *this, *this );
   }

   double Vector :: normInf( void ) const
   {
      return std::max( std::max( fabs(x), fabs(y) ), fabs(z) );
   }
   
   void Vector :: normalize( void )
   {
      (*this) /= norm();
   }
   
   Vector Vector :: unit( void ) const
   {
      return (*this) / norm();
   }
   
   Vector Vector :: abs( void ) const
   {
      return Vector( fabs( x ),
                     fabs( y ),
                     fabs( z ) );
   }

   Vector Vector :: randSphere( void )
   {
      double z1 = unitRand()*2.;
      double z2 = unitRand();

      return Vector(
            sqrt( z1*(2.-z1)) * cos(2.*M_PI*z2),
            sqrt( z1*(2.-z1)) * sin(2.*M_PI*z2),
            1.-z1
            );
   }

   double dot( const Vector& u, const Vector& v )
   {
      return u.x*v.x +
             u.y*v.y +
             u.z*v.z ;
   }
   
   Vector cross( const Vector& u, const Vector& v )
   {
      return Vector( u.y*v.z - u.z*v.y,
                     u.z*v.x - u.x*v.z,
                     u.x*v.y - u.y*v.x );
   }
   
   std::ostream& operator << (std::ostream& os, const Vector& o)
   {
      os << "[ "
         << o.x << " "
         << o.y << " "
         << o.z
         << " ]";
   
      return os;
   }
}

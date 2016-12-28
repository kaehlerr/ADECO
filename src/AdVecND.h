/*
 
 Ralf Kaehler
 
 14 December 2016
 
 
	Copyright (c) 2016, The Board of Trustees of the Leland Stanford Junior University,
	through SLAC National Accelerator Laboratory (subject to receipt of any required approvals
	from the U.S. Dept. of Energy). All rights reserved. Redistribution and use in source and
	binary forms, with or without modification, are permitted provided that the following
	conditions are met:
	(1) Redistributions of source code must retain the above copyright notice,
	this list of conditions and the following disclaimer.
	(2) Redistributions in binary form must reproduce the above copyright notice,
	this list of conditions and the following disclaimer in the documentation and/or other
	materials provided with the distribution.
	(3) Neither the name of the Leland Stanford Junior University, SLAC National Accelerator
	Laboratory, U.S. Dept. of Energy nor the names of its contributors may be used to endorse
	or promote products derived from this software without specific prior written permission.
 
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
	OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
	MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
	COPYRIGHT OWNER, THE UNITED STATES GOVERNMENT, OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
	INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
	LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
	BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
	STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
	USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
	You are under no obligation whatsoever to provide any bug fixes, patches, or upgrades to
	the features, functionality or performance of the source code ("Enhancements") to anyone;
	however, if you choose to make your Enhancements available either publicly, or directly to
	SLAC National Accelerator Laboratory, without imposing a separate written license agreement
	for such Enhancements, then you hereby grant the following license: a non-exclusive,
	royalty-free perpetual license to install, use, modify, prepare derivative works,
	incorporate into other computer software, distribute, and sublicense such Enhancements or
	derivative works thereof, in binary and source code form.
 
 */



#ifndef _AD_VEC_ND_
#define _AD_VEC_ND_

#include <math.h>
#include <vector>
#include <iostream>

#include "AdAssert.h"
#include "AdException.h"



namespace AdaptiveMassDeposit
{

 template <typename T> class AdVec3D
 {
     
 public:

     inline AdVec3D( )
     {
         data_[0] = data_[1] = data_[2] = T(0.0);
     }

     inline explicit AdVec3D( const T a, const T b, const T c)
     {
         data_[0] = a;
         data_[1] = b;
         data_[2] = c;
     }
     
     inline explicit AdVec3D( const T a )
     {
         data_[0] = data_[1] = data_[2] = a;
     }
     
     
     inline AdVec3D( const AdVec3D& other )
     {
         data_[0] = other.data_[0];
         data_[1] = other.data_[1];
         data_[2] = other.data_[2];
     }
     
     
     template <typename U>
     inline explicit AdVec3D<T>( const AdVec3D<U>& source)
     {
         data_[0] = T(source[0]);
         data_[1] = T(source[1]);
         data_[2] = T(source[2]);
     }
     
         
     inline void print() const
     {
         std::cout << "INFO: AdVec3D::print(): vec = [" << data_[0]<<","<<data_[1]<<","<<data_[2]<<"]\n"; 
     }
     
     inline AdVec3D& operator=( const AdVec3D& vec )
     {
         data_[0] = vec.data_[0];
         data_[1] = vec.data_[1];
         data_[2] = vec.data_[2];
         return *this;
         
     }
     
     
     inline AdVec3D operator+( const AdVec3D& a ) const 
     {
         
         return AdVec3D(a.data_[0]+data_[0],
                        a.data_[1]+data_[1],
                        a.data_[2]+data_[2] );
         
     }
     
     inline AdVec3D operator-( const AdVec3D& a )  const
     {
         
         return AdVec3D(data_[0]-a.data_[0],
                        data_[1]-a.data_[1],
                        data_[2]-a.data_[2] );
         
     }
     
     
     inline bool operator==(const AdVec3D& b) const
     {
         
         return (data_[0]==b.data_[0] && 
                 data_[1]==b.data_[1] &&
                 data_[2]==b.data_[2] );
         
     }
     
     inline bool operator!=(const AdVec3D& b) const
     {
         
         return (data_[0]!=b.data_[0] || 
                 data_[1]!=b.data_[1] ||
                 data_[2]!=b.data_[2] );
         
     }

     inline AdVec3D operator*( const T val ) const 
     {
        return AdVec3D( val*data_[0], val*data_[1], val*data_[2] );         
     }
     
     inline friend AdVec3D operator*(const T val, const AdVec3D& a)
     {
        return AdVec3D(val*a.data_[0], val*a.data_[1], val*a.data_[2]);
     }
     
     inline AdVec3D  operator/( const T val ) const 
     {
         if ( val==T(0.0) ) 
         {
             printf( "WARNING: AdVec3D: division by zero.\n");
             //AV_DEBUG( AV_ERROR, { throw AvDivisionByZeroException("ERROR: AdVec3D::operator/ by zero.\n",false); } );
         }
         return AdVec3D( data_[0]/val, data_[1]/val, data_[2]/val );
    }
     
     inline void normalize()
     {
         const long double len = length();
         if ( len<=0.0 )
         {
             printf( "WARNING: AdVec3D::normalize(): division by zero.\n");
             //AV_DEBUG( AV_ERROR, {throw AvDivisionByZeroException("ERROR: AdVec3D::normalize(): Division by zero.\n",false); });
         }
         
        data_[0] /= len;
        data_[1] /= len;
        data_[2] /= len;  
        
     }
     
     inline AdVec3D normalized( ) const 
     {
         const long double len = sqrtl( data_[0]*data_[0] + data_[1]*data_[1] + data_[2]*data_[2] );
         if ( len<=0.0 )
         {
             printf( "WARNING: AdVec3D::normalized(): by zero.\n");
             //AV_DEBUG( AV_ERROR, {throw AvDivisionByZeroException("ERROR: AdVec3D::normalized(): Division by zero.\n",false);} );
         }
         return AdVec3D( data_[0]/len, data_[1]/len, data_[2]/len );
     }
     
     
     inline long double length2( ) const 
     {
         return ( data_[0]*data_[0] + data_[1]*data_[1] + data_[2]*data_[2] );
     }
     
     inline long double length( ) const 
     {
         return sqrtl( length2() );
     }
     
     
     inline T operator[]( int i ) const 
     {
        return data_[i];
     }
     
     
     inline T& operator[]( int i ) 
     {
         return data_[i];
     }
     
     inline T* get_ptr()
     {
         return data_;
     }
     
     inline const T* get_ptr() const
     {
         return data_;
     }
     
     
 private:
     T data_[3];
     
     
 };



} // end namespace 






#endif

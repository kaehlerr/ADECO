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


#ifndef AMD_TET_INTERSECTION
#define AMD_TET_INTERSECTION

#include <stdio.h>
#include <math.h>

#ifdef __CUDACC__

#define assert(arg) ( (void) 0 )

#else

#include <assert.h>

#define __constant__
#define __shared__
#define __device__
#define __global__

#endif


__constant__ const unsigned char tet_connectivity[16][3][4] =
{
    // {255,255,255,255} is used to fill up empty spots for cases with less than 3 tets
    
    { {255,255,255,255},    {255,255,255,255},      {255,255,255,255}   },      // lookup index:  0 ( vertex flags = 0, 0, 0, 0)
    { {0,5,6,4},            {255,255,255,255},      {255,255,255,255}   },      // lookup index:  1 ( vertex flags = 1, 0, 0, 0}
    { {7,1,8,4},            {255,255,255,255},      {255,255,255,255}   },      // lookup index:  2 ( vertex flags = 0, 1, 0, 0}
    { {0,6,8,7},            {6,0,5,7},              {1,0,8,7},          },      // lookup index:  3 ( vertex flags = 1, 1, 0, 0}
    { {2,7,9,5},            {255,255,255,255},      {255,255,255,255}   },      // lookup index:  4 ( vertex flags = 0, 0, 1, 0}
    { {6,0,9,7},            {0,6,4,7},              {0,2,9,7},          },      // lookup index:  5 ( vertex flags = 1, 0, 1, 0}
    { {5,1,9,8},            {9,1,5,2},              {1,5,4,8},          },      // lookup index:  6 ( vertex flags = 0, 1, 1, 0}
    { {6,0,9,8},            {0,2,9,8},              {0,1,2,8},          },      // lookup index:  7 ( vertex flags = 1, 1, 1, 0}
    { {8,3,9,6},            {255,255,255,255},      {255,255,255,255}   },      // lookup index:  8 ( vertex flags = 0, 0, 0, 1}
    { {0,5,9,8},            {5,0,4,8},              {3,0,9,8},          },      // lookup index:  9 ( vertex flags = 1, 0, 0, 1}
    { {1,6,9,7},            {1,9,6,3},              {6,1,4,7},          },      // lookup index: 10 ( vertex flags = 0, 1, 0, 1}
    { {0,5,9,7},            {3,0,9,7},              {1,0,3,7},          },      // lookup index: 11 ( vertex flags = 1, 1, 0, 1}
    { {6,2,8,7},            {6,8,2,3},              {2,6,5,7},          },      // lookup index: 12 ( vertex flags = 0, 0, 1, 1}
    { {4,0,8,7},            {0,3,8,7},              {0,2,3,7},          },      // lookup index: 13 ( vertex flags = 1, 0, 1, 1}
    { {1,4,6,5},            {3,1,6,5},              {2,1,3,5},          },      // lookup index: 14 ( vertex flags = 0, 1, 1, 1}
    { {0,2,3,1},            {255,255,255,255},      {255,255,255,255}   }       // lookup index: 15 ( vertex flags = 1, 1, 1, 1)
    
};


__constant__ const int tet_num[16] =
{
    0, 1, 1, 3, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 1
};


__constant__ const int edges[6][2] = { {0,1},  {0,2}, {0,3}, {1,2}, {1,3}, {2,3} };


template <typename T> __device__  inline T sign( const T& a)
{
    
    return (T(a > 0.f) - T(a < 0.f));

}



template <typename T> class dvec3
{
public:
    
    __device__  inline dvec3()
    {
        data_[0] = 0.f;
        data_[1] = 0.f;
        data_[2] = 0.f;
    }
    
    __device__  inline dvec3( const T a, const T b, const T c )
    {
        data_[0] = a;
        data_[1] = b;
        data_[2] = c;
    }
    
    __device__  inline T operator[]( int i ) const
    {
        assert( i>=0 && i<3 );
        return data_[i];
    }
    
    __device__  inline T& operator[]( int i )
    {
        return data_[i];
    }
    
    __device__  inline dvec3& operator=( const dvec3& vec )
    {
        data_[0] = vec.data_[0];
        data_[1] = vec.data_[1];
        data_[2] = vec.data_[2];
        return *this;
        
    }
    
    __device__  inline dvec3 operator+( const dvec3& a ) const
    {
        return dvec3( a.data_[0]+data_[0], a.data_[1]+data_[1], a.data_[2]+data_[2] );
    }
    
    __device__  inline dvec3 operator-( const dvec3& a )  const
    {
        return dvec3( data_[0]-a.data_[0], data_[1]-a.data_[1], data_[2]-a.data_[2] );
    }
    
    __device__  inline dvec3 operator*( const T val ) const
    {
        return dvec3( val*data_[0], val*data_[1], val*data_[2] );
    }
    
    
    __device__  inline T* getPtr()
    {
        return data_;
    }
    
    __device__ inline const T* getPtr() const
    {
        return data_;
    }
    
    
private:
    T data_[3];
    
};





// this version omits the factor 1/6 since
template <typename T>
__device__ inline T get_tet_volume_times6( const T* a, const T* b, const T* c, const T* d )
{
    
#if 0
    return dot( (a-b), cross( (b-d),(c-d) ));
#else
    
    const T ady = (a[1] - d[1]);
    const T bdy = (b[1] - d[1]);
    const T cdy = (c[1] - d[1]);
    const T adz = (a[2] - d[2]);
    const T bdz = (b[2] - d[2]);
    const T cdz = (c[2] - d[2]);
    
    return ((a[0] - d[0]) * (bdy * cdz - bdz * cdy) +
            (b[0] - d[0]) * (cdy * adz - cdz * ady) +
            (c[0] - d[0]) * (ady * bdz - adz * bdy) );
#endif
    
}

template <typename T>
__device__ inline void get_intersection(const T* line_a,       const T* line_b,
                                        const T* plane_offset, const T* plane_normal,
                                        T* intersection )
{
    
    const T nom_a =
        (plane_offset[0]-line_a[0])*plane_normal[0] +
        (plane_offset[1]-line_a[1])*plane_normal[1] +
        (plane_offset[2]-line_a[2])*plane_normal[2];
    
    const T l[3] = { line_b[0]-line_a[0], line_b[1]-line_a[1], line_b[2]-line_a[2] };
    const T den = l[0]*plane_normal[0] + l[1]*plane_normal[1] + l[2]*plane_normal[2];
    
    const T tmp = den==0. ? 0. : nom_a/den;
    
    const T d = (tmp<=0.) ? 0. : ( (tmp<=1.) ? tmp : 1.);
    
    intersection[0] = line_a[0] + d*l[0];
    intersection[1] = line_a[1] + d*l[1];
    intersection[2] = line_a[2] + d*l[2];
    
}


template < typename T>
__device__ inline void get_intersection_opt(const int face, const int o, const int dir, const T* line_a, const T* line_b, T* intersection )
{
    T nom_a;
    
    if ( face<4 )
        nom_a = dir==0 ?  line_a[o] : 1.-line_a[o]; //(plane_offset[o]-line_a[o])*plane_normal[o];
    else
        nom_a = dir==0 ?  -line_a[o] : line_a[o]-1.;
    
    const T l[3] = { line_b[0]-line_a[0], line_b[1]-line_a[1], line_b[2]-line_a[2] };
    
    T den;
    
    if ( face<4 )
        den = dir==0 ?  -l[o] : l[o];
    else
        den = dir==0 ?  l[o] : -l[o];

    const T tmp = den==0. ? 0. : nom_a/den;
    
    const T d = (tmp<=0.) ? 0. : ( (tmp<=1.) ? tmp : 1.);
    
    intersection[0] = line_a[0] + d*l[0];
    intersection[1] = line_a[1] + d*l[1];
    intersection[2] = line_a[2] + d*l[2];
    
}


//.. this whole loop as templated function over f ?
template <typename T>
__device__ void get_vertex_orientation(const int face, const int sweep_line, const int dir, const T points[10][3], int vo[4]  )
{
    
    // loop over the 4 vertices of the next tet (first 4 entries in 'points')
    for ( int p=0; p<4; ++p )
    {
        /*
         flag to indicate on which side of the plane this tet vertex is located
         
         in order to be consistent with the lookup table, (0,0,0,0) indicates all vertices
         are outside the tet, so no intersection and no sub-tets (see table above)
         
         normals are facing outward -> so vo[i]==0 means point can't be inside tet
         */
        // make sure to make consistent choices, if point almost in plane ...
        // hack/to-do: better use relative EPS here ...
        
        T orientation;
        
        if ( face<4 )
            orientation = dir==0 ?  points[p][sweep_line] : 1.-points[p][sweep_line];
        else
            orientation = dir==0 ?  -points[p][sweep_line] : -1.+points[p][sweep_line];
        
        // optimize this !!!
        const int s = sign(orientation);//(orientation>0.) ? 1 : ( orientation<0. ? -1 : 0 );
        
        vo[p] = (1 + s) >> 1;
        
    } // end loop over p
    
}


__device__ int get_lookup_index( const int vo[4] )
{
    return ( vo[0] ) | ( vo[1]<<1 ) |  ( vo[2]<<2 ) |  ( vo[3]<<3 ) ;
}

//.. this whole loop as templated function over f ?
template <typename T>
__device__ unsigned char process_face(const int f, const int sweep_line, const int dir,  unsigned char& lookup_idx, T points[10][3], T& dvol  )
{
    
    // this should always be the case by lookup table construction
    assert( get_tet_volume_times6( points[0],points[1],points[2],points[3] )>-1.E-3 );
    
    
    // array to store vertex orientations relative to the face's plane
    // hack/to-do: use char instead of int
    int vo[4];// = { 0,0,0,0 };
    
    get_vertex_orientation( f, sweep_line, dir, points, vo );
    
    // compute lookup index
    lookup_idx = get_lookup_index(vo);
    
    // and now use index to lookup table to generate set of tets for the next face
    const unsigned char num_new_tets = tet_num[lookup_idx];
    
    if ( num_new_tets==0 )
    {
        return 0;
    }
        
        
    // compute lookup index and intersection points on each axis
#pragma unroll
    for ( int e=0; e<6; ++e )
    {
        // hack/to-do: unroll this and get rid of edges lookup table !!!
        if ( vo[edges[e][0]]!=vo[edges[e][1]] )
        {
            get_intersection_opt(f,sweep_line, dir, points[edges[e][0]], points[edges[e][1]], points[e+4] );
        }
    }
    
   
    if ( f>3 )
    {
        // and add resulting tets to tet_buffer
        for ( int i=0; i<num_new_tets; ++i )
        {
            // add at position num_tets
            const unsigned char* con_char = tet_connectivity[lookup_idx][i];
            const int con[4] = { int(con_char[0]),int(con_char[1]),int(con_char[2]),int(con_char[3]) };  assert( con[0]>=0 && con[1]>=0 &&con[2]>=0 &&con[3]>=0 );
            //const T vol = get_tet_volume_times6( points[con[0]],points[con[1]],points[con[2]],points[con[3]] );
            assert ( get_tet_volume_times6( points[con[0]],points[con[1]],points[con[2]],points[con[3]] ) > -1.E-1 );
            dvol += get_tet_volume_times6( points[con[0]],points[con[1]],points[con[2]],points[con[3]]);
        }
    
    }
    
    return num_new_tets;
    
    
}


template <typename T>
__device__ void get_tet_from_lookup_index( const unsigned char lookup_idx, const unsigned char tet_num, const T points_f1[10][3],  T points_f2[10][3] )
{
  
    assert( tet_num<=3 );
    assert( lookup_idx<16 );
    
    const unsigned char* con = tet_connectivity[lookup_idx][tet_num];
  
    assert ( get_tet_volume_times6( points_f1[con[0]],points_f1[con[1]],points_f1[con[2]],points_f1[con[3]] ) > -1.E-3 );
    
    points_f2[0][0] = points_f1[con[0]][0]; points_f2[0][1] = points_f1[con[0]][1]; points_f2[0][2] = points_f1[con[0]][2];
    points_f2[1][0] = points_f1[con[1]][0]; points_f2[1][1] = points_f1[con[1]][1]; points_f2[1][2] = points_f1[con[1]][2];
    points_f2[2][0] = points_f1[con[2]][0]; points_f2[2][1] = points_f1[con[2]][1]; points_f2[2][2] = points_f1[con[2]][2];
    points_f2[3][0] = points_f1[con[3]][0]; points_f2[3][1] = points_f1[con[3]][1]; points_f2[3][2] = points_f1[con[3]][2];
    
}



// normals need to be defined such that they point away from the tet's center of mass
// dst_tets mus be ordered such that tet volume is positive
template <typename T>
__device__  inline T compute_unit_cube_tet_intersection_volume( const dvec3<T>* dst_tet )
{
    
    
#if 1
    
    T volume = 0.;
    T dummy_vol;
    
    
    // face 1
    //process_face<0,0,0,T>( num_processed_tets, num_tets, tet_buffer );
    T points_f1[10][3] = {
        {dst_tet[0][0],dst_tet[0][1],dst_tet[0][2]},
        {dst_tet[1][0],dst_tet[1][1],dst_tet[1][2]},
        {dst_tet[2][0],dst_tet[2][1],dst_tet[2][2]},
        {dst_tet[3][0],dst_tet[3][1],dst_tet[3][2]}};
    
    // store face idx, sweeplines, orientation and number of tets for each face
    unsigned char faces[6][4] =
    {
        {0,0,0,0},
        {1,0,1,0},
        {2,1,0,0},
        {3,1,1,0},
        {4,2,0,0},
        {5,2,1,0}
        
    };
    
    const int o[6] = {0,1,2,3,4,5};
    
    unsigned char lookup_idx_f1;
    const unsigned char num_new_tets_f1 = process_face<T>(faces[o[0]][0],faces[o[0]][1],faces[o[0]][2], lookup_idx_f1, points_f1, dummy_vol  );
    
    //if ( num_new_tets_f1==0 ) return 0.;
    
    for ( unsigned char next_tet_f1=0; next_tet_f1<3 ; ++next_tet_f1 )
    {
        if ( next_tet_f1==num_new_tets_f1 ) break;
        T points_f2[10][3]; get_tet_from_lookup_index( lookup_idx_f1, next_tet_f1, points_f1, points_f2 );
        unsigned char lookup_idx_f2; const unsigned char num_new_tets_f2 = process_face<T>(faces[o[1]][0],faces[o[1]][1],faces[o[1]][2],lookup_idx_f2,  points_f2, dummy_vol  );
        
        
        for ( unsigned char next_tet_f2=0; next_tet_f2<3 ; ++next_tet_f2 )
        {
            if ( next_tet_f2==num_new_tets_f2 ) break;
            T points_f3[10][3]; get_tet_from_lookup_index( lookup_idx_f2, next_tet_f2, points_f2, points_f3 );
            unsigned char lookup_idx_f3; const unsigned char num_new_tets_f3 = process_face<T>(faces[o[2]][0],faces[o[2]][1],faces[o[2]][2],lookup_idx_f3,  points_f3, dummy_vol  );
            
            for ( unsigned char next_tet_f3=0; next_tet_f3<3 ; ++next_tet_f3 )
            {
                if ( next_tet_f3==num_new_tets_f3 ) break;
                T points_f4[10][3]; get_tet_from_lookup_index( lookup_idx_f3, next_tet_f3, points_f3, points_f4 );
                unsigned char lookup_idx_f4; const unsigned char num_new_tets_f4 = process_face<T>(faces[o[3]][0],faces[o[3]][1],faces[o[3]][2],lookup_idx_f4,  points_f4, dummy_vol  );
                
                
                
                // process face 4 and compute inverse volume
                for ( unsigned char next_tet_f4=0; next_tet_f4<3 ; ++next_tet_f4 )
                {
                    
                    if ( next_tet_f4==num_new_tets_f4 ) break;
                    
                    T points_f56[10][3]; get_tet_from_lookup_index( lookup_idx_f4, next_tet_f4, points_f4, points_f56 );
                    unsigned char lookup_idx_f56;
                    
                    // check against left edge
                    T dvol1=0.;
                    process_face<T>( faces[o[4]][0],faces[o[4]][1],faces[o[4]][2],lookup_idx_f56,  points_f56, dvol1  ); assert( dvol1>=-1.E-1 );
                    
                    // check against right edge
                    T dvol2=0.;
                    process_face<T>( faces[o[5]][0],faces[o[5]][1],faces[o[5]][2], lookup_idx_f56,  points_f56, dvol2  ); assert( dvol2>=-1.E-1 );
                    
                    
                    // add volume of tet itself
                    volume += get_tet_volume_times6( points_f56[0], points_f56[1],points_f56[2],points_f56[3]) - dvol1 - dvol2;
                    
                    
                } // end loop next_tet_f4
                
                
            } // end loop next_tet_f3
            
            
        } // end loop next_tet_f2
        
        
    } // end loop next_tet_f1
    
    
    return 0.16666666666666666*volume;
    
    
    
#else
    
    
   T tet_buffer[121][4][3];
    
    tet_buffer[0][0][0] = dst_test[0][0]; tet_buffer[0][0][1] = dst_test[0][1]; tet_buffer[0][0][2] = dst_test[0][2];
    tet_buffer[0][1][0] = dst_test[1][0]; tet_buffer[0][1][1] = dst_test[1][1]; tet_buffer[0][1][2] = dst_test[1][2];
    tet_buffer[0][2][0] = dst_test[2][0]; tet_buffer[0][2][1] = dst_test[2][1]; tet_buffer[0][2][2] = dst_test[2][2];
    tet_buffer[0][3][0] = dst_test[3][0]; tet_buffer[0][3][1] = dst_test[3][1]; tet_buffer[0][3][2] = dst_test[3][2];
    
    int num_processed_tets = 0;
    int num_tets = 1;
    
    process_face<0,0,0,T>( num_processed_tets, num_tets, tet_buffer );
    process_face<1,0,1,T>( num_processed_tets, num_tets, tet_buffer );
    process_face<2,1,0,T>( num_processed_tets, num_tets, tet_buffer );
    process_face<3,1,1,T>( num_processed_tets, num_tets, tet_buffer );
    
    T delta_vol = 0;
    delta_vol += process_face<4,2,0,T>( num_processed_tets, num_tets, tet_buffer );
    delta_vol += process_face<5,2,1,T>( num_processed_tets, num_tets, tet_buffer );
    
 
    // the remaining tets between processed_tets and num_tets are the ones that tessellate the intersection volume
    T intersection_volume = 0.f;
    for ( int t=num_processed_tets; t<num_tets; ++t )
    {
        intersection_volume += fabs(get_tet_volume_times6( tet_buffer[t][0],tet_buffer[t][1],tet_buffer[t][2],tet_buffer[t][3] ));
    }
  
    return (0.16666666666666666*(intersection_volume-delta_vol));

#endif
    
}



#endif


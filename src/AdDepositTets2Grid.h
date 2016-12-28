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

#ifndef AD_TET_TET_INTERSECTION
#define AD_TET_TET_INTERSECTION

#include "stddef.h"
#include "math.h"

#include "AdTetIntersection.h"



#ifndef __CUDACC__

#include <assert.h>
#define __constant__
#define __shared__
#define __device__
#define __global__

#endif



template <typename T> class dbox
{
public:
    
    __device__  dbox() {};
    __device__  dbox( const dvec3<T>& mi, const dvec3<T>& ma ) : min(mi), max(ma) {}
   
    __device__  inline T getVolume() const
    {
        return ( (max[0]-min[0])*(max[1]-min[1])*(max[2]-min[2]) );
    }
    
    __device__  inline bool inside( const dvec3<T>& pos) const
    {
        return ((pos[0]>=min[0]&&pos[0]<=max[0]) &&
                (pos[1]>=min[1]&&pos[1]<=max[1]) &&
                (pos[2]>=min[2]&&pos[2]<=max[2]) );
    }
    
    dvec3<T> min;
    dvec3<T> max;
    
};


template <typename T>
__device__  inline bool intersection(const dbox<T>& b1, const dbox<T>& b2)
{
    return (b1.min[0] < b2.max[0] && b2.min[0] < b1.max[0] &&
            b1.min[1] < b2.max[1] && b2.min[1] < b1.max[1] &&
            b1.min[2] < b2.max[2] && b2.min[2] < b1.max[2]);
}


template <typename T>
__device__  inline bool get_intersection( const dbox<T>& region1, const dbox<T>& region2, dbox<T>& result )
{
    
    if ( intersection(region1, region2)==false )
    {
        return false;
    }
    else
    {
        dvec3<T>& ll = result.min;
        dvec3<T>& ur = result.max;
        for ( int i=0; i<3; i++  )
        {
            ll[i] = ( region1.min[i] < region2.min[i] )  ? region2.min[i] : region1.min[i];
            ur[i] = ( region1.max[i] < region2.max[i] )  ? region1.max[i] : region2.max[i];
        }
    }
    
    return true;
    
}


template <typename T>
__device__  inline dvec3<T> cross( const dvec3<T>& a, const dvec3<T>& b )
{
    
    return dvec3<T>(a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]);
}


template <typename T> __device__ inline T clamp( const T val, const T min, const T max )
{
    if ( val<=min ) return min;
    else if ( val>=max ) return max;
    else return val;
}


template <typename T>
__device__  inline void dvec_array_2_c_array( const dvec3<T> src_tet[4], T dst_ptr[4][3] )
{
    
    
    for ( int j=0; j<4; ++j )
        for ( int i=0; i<3; ++i )
            dst_ptr[j][i] = src_tet[j][i];
    
}


template <typename T>
__device__  inline void cross( const T a[3], const T b[3], T res[3] )
{
    res[0] = a[1]*b[2]-a[2]*b[1];
    res[1] = a[2]*b[0]-a[0]*b[2];
    res[2] = a[0]*b[1]-a[1]*b[0];
}



template <typename T> __device__  inline dbox<T> get_bbox( const dvec3<T> points[4] )
{
    
    dbox<T> res;
    
    res.min = dvec3<T>(points[0]);
    res.max = res.min;
    
    for ( int i=1; i<4; ++i )
    {
        
        const dvec3<T>& p = points[i];
        
        if      ( p[0]<res.min[0] ) res.min[0] = p[0];
        else if ( p[0]>res.max[0] ) res.max[0] = p[0];
        
        if      ( p[1]<res.min[1] ) res.min[1] = p[1];
        else if ( p[1]>res.max[1] ) res.max[1] = p[1];
        
        if      ( p[2]<res.min[2] ) res.min[2] = p[2];
        else if ( p[2]>res.max[2] ) res.max[2] = p[2];
    }
    
    return res;
}


__constant__  const int vert[8][3] =
{
    {0,0,0}, {0,0,1}, {0,1,0}, {0,1,1},
    {1,0,0}, {1,0,1}, {1,1,0}, {1,1,1}
};


// to-do: optimize this !!!
template <typename T>
__device__  inline void get_cell( const dvec3<T>& cell_min, const dvec3<T>& delta,  dvec3<T> result[8] )
{
    for( int i=0; i<8; ++i )
    {
        result[i] = cell_min + dvec3<T>(delta[0]*vert[i][0], delta[1]*vert[i][1],delta[2]*vert[i][2]);
    }
    
}


__constant__ const int tet_conn[6][4] =
{
    
#if 1
    
    // this 6 tet table
    {1,0,2,4},
    {4,2,1,3},
    {3,5,1,4},
    {4,5,6,3},
    {3,2,6,4},
    {3,7,5,6}
    
#else
    
    // 5 tet table based
    {0,4,1,2},
    {4,2,6,7},
    {1,7,4,5},
    {1,7,3,2},
    {4,1,2,7}
#endif
    
    
};


template <typename T>
__device__  inline void get_cell_tet( const unsigned int id, const dvec3<T> cell[8], dvec3<T> grid_tet[4] )
{
    
    assert( id<5 );
    
    grid_tet[0] = cell[tet_conn[id][0]];
    grid_tet[1] = cell[tet_conn[id][1]];
    grid_tet[2] = cell[tet_conn[id][2]];
    grid_tet[3] = cell[tet_conn[id][3]];
    
    // are the tets oriented correctly ?
    assert( get_tet_volume_times6( grid_tet[0].getPtr(), grid_tet[1].getPtr(), grid_tet[2].getPtr(), grid_tet[3].getPtr() ) > 0. );
    
}


//.. this whole loop as template function for f ?
template <typename T>
__device__ unsigned char get_num_new_tets(const int f, const int sweep_line, const int dir,  const dvec3<T>* points )
{
    
    int vo[4] = {0,0,0,0};
    // important: we can not use the function: get_vertex_orientation( f,sweep_line,dir, points, vo ) because it handles f==4&5 differently
    {
        // loop over the 4 vertices of the next tet (first 4 entries in 'points')
        for ( int p=0; p<4; ++p )
        {
            T orientation = dir==0 ?  points[p][sweep_line] : 1.-points[p][sweep_line];
            const int s = (orientation>0.) ? 1 : ( orientation<0. ? -1 : 0 );
            vo[p] = (1 + s) >> 1;
            
        } // end loop over p
    }
    
    const unsigned char lookup_idx = get_lookup_index( vo );
    
    return tet_num[lookup_idx];
    
}




enum INTERSECTION_TEST_RESULT { NO_INTERSECTION=0, TET_INSIDE_CELL, CELL_INSIDE_TET, POTENTIAL_PARTIAL_INTERSECTION, INVALID_INTERSECTION_RESULT };

/*
   convervative check for intersection between cell and tet
    ( == there might still be no intersection if this method return true,
        but if it return false, there definitively is not intersection (up to round-off errors) )
*/

template <typename T>
__device__  inline INTERSECTION_TEST_RESULT intersection_test( const dbox<T>& cell_box, const dvec3<T> cell_vertices[8], const dvec3<T> src_tet[4] )
{
  
    assert( get_tet_volume_times6( src_tet[0].getPtr(), src_tet[1].getPtr(), src_tet[2].getPtr(), src_tet[3].getPtr() )>-1.E-01 );
    
    const dvec3<T> normals[4] = {
        cross( src_tet[1]-src_tet[0], src_tet[2]-src_tet[0] ),
        cross( src_tet[2]-src_tet[0], src_tet[3]-src_tet[0] ),
        cross( src_tet[3]-src_tet[0], src_tet[1]-src_tet[0] ),
        cross( src_tet[3]-src_tet[1], src_tet[2]-src_tet[1] )
    };
    // one vertex for each face
    const dvec3<T>* offsets[4] = { &src_tet[0], &src_tet[0], &src_tet[0], &src_tet[1] };

    
    
    int all_faces_combined = 0;
    // now loop over all 4 faces
    for ( int f=0; f<4; ++f )
    {
        int lookup_idx = 0;
        for ( int p=0; p<8; ++p )
        {
            
            /*
             flag to indicate on which side of the plane this tet vertex is located
             
             in order to be consistent with the lookup table, (0,0,0,0,...) indicates all vertices
             are outside the tet, so no intersection and no sub-tets (see table above)
             
             normals are facing outward -> so vo[i]==0 means point can't be inside tet
            */
            // make sure to make consistent choices, even if point almost in plane ...
            // hack/to-do: better use relative EPS here ...
            const T orientation =
                ( (*(offsets[f]))[0]-cell_vertices[p][0])*normals[f][0] +
                ( (*(offsets[f]))[1]-cell_vertices[p][1])*normals[f][1] +
                ( (*(offsets[f]))[2]-cell_vertices[p][2])*normals[f][2];
            const int v = ( (1 + int ( sign( orientation ) )) >> 1);
            // compute lookup index
            lookup_idx |= (v<<p);
            
        }
    
        assert( lookup_idx>=0 && lookup_idx<256);
        
        if (  lookup_idx==0 )
            return NO_INTERSECTION;
        
        all_faces_combined += lookup_idx;
        
        
    }
    
    /*
        next check if cell is inside tet:
        combined lookup results of 1020==4*255 indicate that all cell vertices 
        are located on 'inside'-half space for each tet face -> cell inside tet
     */
    if ( all_faces_combined==1020 )
    {
        return CELL_INSIDE_TET;
    }
    
    // now check if tet is completety inside cell
    if (cell_box.inside(src_tet[0]) &&
        cell_box.inside(src_tet[1]) &&
        cell_box.inside(src_tet[2]) &&
        cell_box.inside(src_tet[3]) )
    {
        return TET_INSIDE_CELL;
   }
    
    // this is a candidate for the partial intersection - need to inspect this case using the most expensive way ;-(
    return POTENTIAL_PARTIAL_INTERSECTION;
    
}



__constant__ float unit_cell_vertices[8][3] =
{
    { 0.f, 0.f, 0.f },
    { 0.f, 0.f, 1.f },
    { 0.f, 1.f, 0.f },
    { 0.f, 1.f, 1.f },
    { 1.f, 0.f, 0.f },
    { 1.f, 0.f, 1.f },
    { 1.f, 1.f, 0.f },
    { 1.f, 1.f, 1.f }
};


template <typename T>
__device__ void scale_tet(const dvec3<T>* src_tet, const dbox<T>& grid_bbox, const dvec3<T>& delta,
                          const T scale,
                          const int i, const int j, const int k,
                          dvec3<T>* scaled_tet )
{
    
    const dvec3<T> offset(grid_bbox.min[0] + i*delta[0],
                          grid_bbox.min[1] + j*delta[1],
                          grid_bbox.min[2] + k*delta[2]);
    scaled_tet[0] = (src_tet[0]-offset)*scale;
    scaled_tet[1] = (src_tet[1]-offset)*scale;
    scaled_tet[2] = (src_tet[2]-offset)*scale;
    scaled_tet[3] = (src_tet[3]-offset)*scale;
}


template <typename T>
__device__ INTERSECTION_TEST_RESULT classify_cell(const dbox<T>& tet_box,      const dvec3<T>* src_tet, const T scale,
                                                  const dbox<T>& grid_bbox,    const dvec3<T>& delta,
                                                  const int i,                 const int j,                const int k )
{
    
    
    // now loop over each of the covered cells
    dbox<T> cell_box;
    
    cell_box.min[0] = grid_bbox.min[0] + i*delta[0];
    cell_box.max[0] = cell_box.min[0] + delta[0];
    cell_box.min[1] = grid_bbox.min[1] + j*delta[1];
    cell_box.max[1] = cell_box.min[1] + delta[1];
    cell_box.min[2] = grid_bbox.min[2] + k*delta[2];
    cell_box.max[2] = cell_box.min[2] + delta[2];
    
    // perform the cheapest intersection tests first
    if ( !intersection(tet_box, cell_box) )
    {
        return NO_INTERSECTION;
    }
    else if (cell_box.inside(src_tet[0]) &&
             cell_box.inside(src_tet[1]) &&
             cell_box.inside(src_tet[2]) &&
             cell_box.inside(src_tet[3]))
    {
        return TET_INSIDE_CELL;
    }
    
    const dvec3<T> scaled_src_tet[4] =
    {
        (src_tet[0]-cell_box.min)*scale,
        (src_tet[1]-cell_box.min)*scale,
        (src_tet[2]-cell_box.min)*scale,
        (src_tet[3]-cell_box.min)*scale
    };
    
    {
        // it seems to be faster to recompute normal each time - might reduce register spilling ...
        const dvec3<T> normals[4] = {
            cross( scaled_src_tet[1]-scaled_src_tet[0], scaled_src_tet[2]-scaled_src_tet[0] ),
            cross( scaled_src_tet[2]-scaled_src_tet[0], scaled_src_tet[3]-scaled_src_tet[0] ),
            cross( scaled_src_tet[3]-scaled_src_tet[0], scaled_src_tet[1]-scaled_src_tet[0] ),
            cross( scaled_src_tet[3]-scaled_src_tet[1], scaled_src_tet[2]-scaled_src_tet[1] )
        };
        // one vertex for each face
        const dvec3<T>* offsets[4] = { &scaled_src_tet[0], &scaled_src_tet[0], &scaled_src_tet[0], &scaled_src_tet[1] };
        
        
        
        int all_faces_combined = 0;
        // now loop over all 4 faces
        for ( int f=0; f<4; ++f )
        {
            int lookup_idx = 0;
            for ( int p=0; p<8; ++p )
            {
                
                /*
                 flag to indicate on which side of the plane this tet vertex is located
                 
                 in order to be consistent with the lookup table, (0,0,0,0,...) indicates all vertices
                 are outside the tet, so no intersection and no sub-tets (see table above)
                 
                 normals are facing outward -> so vo[i]==0 means point can't be inside tet
                 */
                // make sure to make consistent choices, even if point almost in plane ...
                // hack/to-do: better use relative EPS here ...
                const T orientation =
                    ( (*(offsets[f]))[0]-unit_cell_vertices[p][0])*normals[f][0] +
                    ( (*(offsets[f]))[1]-unit_cell_vertices[p][1])*normals[f][1] +
                    ( (*(offsets[f]))[2]-unit_cell_vertices[p][2])*normals[f][2];
              
                const int v = ( (1 + int ( sign( orientation ) )) >> 1);
                
                // compute lookup index
                lookup_idx |= (v<<p);
                
            }
            
            assert( lookup_idx>=0 && lookup_idx<256);
            
            if (  lookup_idx==0 )
                return NO_INTERSECTION;
            
            all_faces_combined += lookup_idx;
            
            
        }
        
        /*
         next check if cell is inside tet:
         combined lookup results of 1020==4*255 indicate that all cell vertices
         are located on 'inside'-half space for each tet face -> cell inside tet
         */
        if ( all_faces_combined==1020 )
        {
            return CELL_INSIDE_TET;
        }
        
        
    }
    
    
    // this is a candidate for the partial intersection - need to inspect this case using the most expensive way ;-(
    return POTENTIAL_PARTIAL_INTERSECTION;
    
}



template <typename G, typename T>
__device__  inline void deposit_tet_2_grid( const dvec3<T> tet_vertices[4], const G tet_mass, const int grid_dims[3],
                                            const dbox<T>& grid_bbox, G* grid_mass )
{
    
    const dbox<T> tet_box = get_bbox<T>( tet_vertices);
    
    dbox<T> overlap;
    if ( !get_intersection<T>( tet_box, grid_bbox, overlap ) )
    {
        // nothing to do - there is no intersection between tet and grid
        return;
    }
    
    const dvec3<T> delta((grid_bbox.max[0]-grid_bbox.min[0])/grid_dims[0],
                         (grid_bbox.max[1]-grid_bbox.min[1])/grid_dims[1],
                         (grid_bbox.max[2]-grid_bbox.min[2])/grid_dims[2] );
    
    const T cell_vol = (delta[0]*delta[1]*delta[2]);
    
    
    int covered_cells[3][2];
    
    for ( int i=0; i<3; ++i )
    {
        covered_cells[i][0] = clamp( int(floor((overlap.min[i]-grid_bbox.min[i])/delta[i])), 0, grid_dims[i]-1);
        covered_cells[i][1] = clamp( int(ceil( (overlap.max[i]-grid_bbox.min[i])/delta[i])), 0, grid_dims[i]-1);
        
        assert( covered_cells[i][0]<=covered_cells[i][1] && covered_cells[i][0]>=0 && covered_cells[i][1]<grid_dims[i] );
        
    }
    
    
    const T tet_vol = get_tet_volume_times6( tet_vertices[0].getPtr(), tet_vertices[1].getPtr(), tet_vertices[2].getPtr(), tet_vertices[3].getPtr() )/6.;
  
    dvec3<T> src_tet[4];
    
    
    if ( tet_vol==0.f )
    {
        //std::cout << "WARNING: deposit_tet_2_grid(): detected tet with volume 0" << std::endl;
        //assert(tet_vol!=0.);
        return;
    }
    
    if ( tet_vol<0.f )
    {
        src_tet[0] = dvec3<T>(tet_vertices[3]);
        src_tet[1] = dvec3<T>(tet_vertices[1]);
        src_tet[2] = dvec3<T>(tet_vertices[2]);
        src_tet[3] = dvec3<T>(tet_vertices[0]);
    }
    else
    {
        src_tet[0] = dvec3<T>(tet_vertices[0]);
        src_tet[1] = dvec3<T>(tet_vertices[1]);
        src_tet[2] = dvec3<T>(tet_vertices[2]);
        src_tet[3] = dvec3<T>(tet_vertices[3]);
    }
    
    const T scale = 1.f/delta[0];
    
    // hack/to-do: precompute this ?
    dvec3<T> unit_cell[8];
    get_cell( dvec3<T>(0.f,0.f,0.f), dvec3<T>(1.f,1.f,1.f), unit_cell );
    
    dbox<T> unit_cell_box;
    unit_cell_box.min = dvec3<T>(0.f,0.f,0.f);
    unit_cell_box.max = dvec3<T>(1.f,1.f,1.f);
    
    // now loop over each of the covered cells
    dbox<T> cell_box;
    for ( int k=covered_cells[2][0]; k<=covered_cells[2][1]; ++k )
    {
        cell_box.min[2] = grid_bbox.min[2] + k*delta[2];
        cell_box.max[2] = cell_box.min[2] + delta[2];
        
        for ( int j=covered_cells[1][0]; j<=covered_cells[1][1]; ++j )
        {
            
            cell_box.min[1] = grid_bbox.min[1] + j*delta[1];
            cell_box.max[1] = cell_box.min[1] + delta[1];
            
            for ( int i=covered_cells[0][0]; i<=covered_cells[0][1]; ++i )
            {
                cell_box.min[0] = grid_bbox.min[0] + i*delta[0];
                cell_box.max[0] = cell_box.min[0] + delta[0];
                
                // perform the cheapest intersection test first
                if ( !intersection(tet_box, cell_box) )
                    continue;
                
                // shift cell-center to origin to increase resolution
                const dvec3<T> shift(cell_box.min);
                
                // scale tet such that each grid cell is of [0,1] size
                const dvec3<T> scaled_src_tet[4] =
                    { (src_tet[0]-shift)*scale, (src_tet[1]-shift)*scale, (src_tet[2]-shift)*scale, (src_tet[3]-shift)*scale } ;
                
                
                // optimize this intersection test !!!
                const INTERSECTION_TEST_RESULT intersection_type = intersection_test( unit_cell_box, unit_cell, scaled_src_tet);
                
                if ( intersection_type==NO_INTERSECTION )
                {
                    // nothing do do for this cell
                    continue;
                }
                
                
                G dmass = 0.f;
                
                if ( intersection_type==TET_INSIDE_CELL)
                {
                    assert( cell_vol>=fabs(tet_vol) );
                    // and deposit the whole tet mass into this cell
                    dmass = tet_mass;
                }
                else if ( intersection_type==CELL_INSIDE_TET )
                {
                    dmass = tet_mass*( cell_vol/fabs(tet_vol) );
                }
                else if ( intersection_type==POTENTIAL_PARTIAL_INTERSECTION)
                {
                    
                    T dvol = compute_unit_cube_tet_intersection_volume( scaled_src_tet );
      
                    // convert vol into rho and store in data array
                    if ( dvol>0.f )
                    {
                        // and rescale the volume ...
                        dvol *= cell_vol;
                        dmass = tet_mass*(dvol/fabs(tet_vol));
                    }
                
                }

                assert ( intersection_type<=INVALID_INTERSECTION_RESULT );
                
                if ( dmass>0.f )
                {
                    // compute index of cell
                    const size_t idx = i + grid_dims[0]*( j + grid_dims[1]*k ); assert( idx<(size_t(grid_dims[0])*size_t(grid_dims[1])*size_t(grid_dims[2])) );
#ifdef __CUDACC__
                    // in cuda case use atomic add for now
                    atomicAdd(grid_mass+idx, dmass );
#else
                    grid_mass[idx] += dmass;
#endif
                }
                
            }
        }
    }
    
    
}

__device__ void store_mass( const float dmass,  const int* grid_dims, const int i, const int j, const int k, float* grid_mass )
{
    
    if ( dmass>0. )
    {
        // compute index of cell
        const size_t idx = i + grid_dims[0]*( j + grid_dims[1]*k ); assert( idx<(size_t(grid_dims[0])*size_t(grid_dims[1])*size_t(grid_dims[2])) );
#ifdef __CUDACC__
        // in cuda case use atomic add for now
        atomicAdd(grid_mass+idx, dmass );
#else
        grid_mass[idx] += dmass;
#endif
    }

}

__device__ void decode_index( const int key, int& i, int& j, int& k )
{
    i = (key & 0xFF);
    j = (key & 0xFF00) >> 8;
    k = (key & 0xFF0000) >> 16;
    
}

__device__ int encode_index( const int i, const int j, const int k )
{
    return (k << 16) + (j << 8) + i;
}



__device__ int get_num_buffered( const int tail, const int head )
{
    return (head-tail);
}


__device__ void get_3D_cell_index( const int covered_cells[3][2], const int idx, int& i, int& j, int& k )
{
    
    // hack/to-do: precompute this
    const int nb  = covered_cells[2][1]-covered_cells[2][0]+1;
    const int nb2 = nb*(covered_cells[1][1]-covered_cells[1][0]+1);
    
    int pid = idx;
    i = ( pid/nb2 );
    pid -= i*nb2;
    j = int(pid/nb);
    k = pid-j*nb;
    
    // and do not forget to add the offset
    i += covered_cells[0][0];
    j += covered_cells[1][0];
    k += covered_cells[2][0];
    
    
}

#ifdef __CUDACC__

template <typename G, typename T>
__device__  inline void deposit_tet_2_grid_refactored(const dvec3<T> src_tet[4],
                                                      const T tet_vol,
                                                      const G tet_mass,
                                                      const int grid_dims[3],
                                                      const dbox<T>& grid_bbox, G* grid_mass )
{
    
    // get id of this thread withing its warp
    const int lane_id = threadIdx.x % 32;
    
    const dbox<T> tet_box = get_bbox<T>( src_tet );
    
    // hack/to-do: compute this outside the this routine and hand it over as an arg
    const dvec3<T> delta((grid_bbox.max[0]-grid_bbox.min[0])/grid_dims[0],
                         (grid_bbox.max[1]-grid_bbox.min[1])/grid_dims[1],
                         (grid_bbox.max[2]-grid_bbox.min[2])/grid_dims[2] );
    
    // compute minimal region of covered cells
    int covered_cells[3][2];
    {
        dbox<T> overlap;
        
        // use intersection result from one lane to make sure we have identical results for whole warp ...
        if ( __shfl( get_intersection<T>( tet_box, grid_bbox, overlap ), 0 )==0 )
        {
            // nothing to do - there is no intersection between tet and grid
            return;
        }
        
        for ( int i=0; i<3; ++i )
        {
            covered_cells[i][0] = clamp( int(floor((overlap.min[i]-grid_bbox.min[i])/delta[i])), 0, grid_dims[i]-1);
            covered_cells[i][1] = clamp( int(ceil( (overlap.max[i]-grid_bbox.min[i])/delta[i])), 0, grid_dims[i]-1);
            
            assert( covered_cells[i][0]<=covered_cells[i][1] && covered_cells[i][0]>=0 && covered_cells[i][1]<grid_dims[i] );
        }
    }
    
    const T cell_vol = (delta[0]*delta[1]*delta[2]);
    
    
    const T scale = 1.f/delta[0];
    
    
    // first deal with trivial cases
    {
        
        const int warp_id = threadIdx.x/32;
        const int num_warps_per_block = AMD_CUDA_BLOCK_DIM/32;//blockDim.x/32;
        
        int tail = 0;
        __shared__ int head[num_warps_per_block];
        head[warp_id] = 0;
        
        const int buffer_size= 64;
        __shared__ int buffer[num_warps_per_block][buffer_size];
        
        __syncthreads();;
        
        const int num_cells = (covered_cells[0][1]-covered_cells[0][0]+1)*(covered_cells[1][1]-covered_cells[1][0]+1)*(covered_cells[2][1]-covered_cells[2][0]+1);
        
        for ( int c=lane_id; c<num_cells; c+=32 )
        {
            int i,j,k;
            get_3D_cell_index( covered_cells, c, i, j, k );
            
            const INTERSECTION_TEST_RESULT intersection_type = classify_cell( tet_box, src_tet, scale, grid_bbox, delta, i, j, k );
            
            if ( intersection_type==TET_INSIDE_CELL || intersection_type==CELL_INSIDE_TET )
            {
                G dmass = 0.f;
                
                if ( intersection_type==TET_INSIDE_CELL)
                {
                    dmass = tet_mass;
                }
                else if ( intersection_type==CELL_INSIDE_TET )
                {
                    dmass = tet_mass*( cell_vol/fabs(tet_vol) );
                }
                store_mass( dmass, grid_dims, i,j,k, grid_mass );
            }
            else if ( intersection_type==POTENTIAL_PARTIAL_INTERSECTION )
            {
                const int pos = (atomicAdd(&head[warp_id],1)%buffer_size);
                buffer[warp_id][pos  ] = encode_index( i,j,k );
            }
            
            
            // is at least half of the buffer filled ?
            if  ( get_num_buffered(tail,head[warp_id])>31 )
            {
                __syncthreads();
                //__threadfence_block();
                
                // need to process the expensive cells in parallel
                {
                    const int offest = (lane_id+tail%buffer_size);
                    decode_index( buffer[warp_id][offest], i, j, k );;
                    
                    dvec3<T> scaled_src_tet[4];
                    scale_tet( src_tet, grid_bbox, delta, scale, i,j,k,scaled_src_tet );
                    
                    // compute intersection volume
                    const T dvol = compute_unit_cube_tet_intersection_volume( scaled_src_tet );
                    // convert vol into rho and store in data array
                    store_mass( tet_mass*(cell_vol*dvol/fabs(tet_vol)), grid_dims, i,j,k, grid_mass );
                    
                }
                tail += 32;
                
            }
            
            
        } // end loop over cells 'c'
        
        __syncthreads();
        
        // not take care of remaining buffered cells
        for ( int n=0; n<buffer_size/32; ++n )
        {
            const int num_current_items = get_num_buffered(tail,head[warp_id]);
            
            if ( num_current_items>0 && lane_id<num_current_items )
            {
                
                const int offest = (lane_id+tail%buffer_size);
                int ii,jj,kk;
                decode_index( buffer[warp_id][offest], ii, jj, kk );;
                
                dvec3<T> scaled_src_tet[4];
                scale_tet( src_tet, grid_bbox, delta, scale, ii,jj,kk,scaled_src_tet );
                // compute intersection volume
                const T dvol = compute_unit_cube_tet_intersection_volume( scaled_src_tet );
                // convert vol into rho and store in data array
                store_mass( tet_mass*(cell_vol*dvol/fabs(tet_vol)), grid_dims, ii,jj,kk, grid_mass );
            }
            if ( num_current_items>32 )
                tail += 32;
            else
                break;
        }
        
        
        
    }
    
    
}

#endif


template <typename T>
__device__ dvec3<T> map_linear_to_3D_idx( const T idx, const dvec3<T>& global_dims )
{
    assert( global_dims[0]>=0 && global_dims[1]>=0 && global_dims[2]>=0 );
    const T nb  = global_dims[0];
    const T nb2 = global_dims[0]*global_dims[1];
    
    T i,j,k;
    
    k = idx/(nb2);
    const T rest1 = idx%(nb2);
    j = rest1/nb;
    i = rest1%nb;
    
    assert( int(i)<global_dims[0] && int(j)<global_dims[1] && int(k)<global_dims[2] );
    
    return dvec3<T>(i,j,k);
}


template <typename T>
__device__ T map_3D_to_linear_idx( const dvec3<T>& idx_3d, const dvec3<T>& dims )
{
    return idx_3d[0] + dims[0] * ( idx_3d[1] + idx_3d[2]*dims[1] ) ;
}


__device__ void cpy_3ptr( const float* src, float* dst )
{
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
}


template <typename T>
__device__ void sample_cell_vertices(const int cell_num,
                                     const T* positions,
                                     const int meta_cells_per_patch,
                                     T* cell_pos)
{
    assert( cell_num < (meta_cells_per_patch*meta_cells_per_patch*meta_cells_per_patch) );
    
    // compute offset into positions array ( lower left corner of our cell )
    const dvec3<int> cell_id_3D = map_linear_to_3D_idx( cell_num, dvec3<int>(meta_cells_per_patch,meta_cells_per_patch,meta_cells_per_patch) );
    
    assert( cell_id_3D[0]>=0 && cell_id_3D[0]<meta_cells_per_patch );
    assert( cell_id_3D[1]>=0 && cell_id_3D[1]<meta_cells_per_patch );
    assert( cell_id_3D[2]>=0 && cell_id_3D[2]<meta_cells_per_patch );
    
    const int offset = 3*map_3D_to_linear_idx( cell_id_3D, dvec3<int>(meta_cells_per_patch+1,meta_cells_per_patch+1,meta_cells_per_patch+1) );
    
    // extract the 8 cell vertices
    //float cell_pos[8*3];
    
    const int addX = 3;
    const int addY = 3*(meta_cells_per_patch+1);
    const int addZ = 3*(meta_cells_per_patch+1)*(meta_cells_per_patch+1);
    
    
    //        d000 = log10((double) dp[0]);
    const T* pos_ptr = positions+offset;
    
    
    
#if 1 // important: this order corresponds to the order of the 6 tet table 'tet_conn' !!!
    cpy_3ptr( pos_ptr, cell_pos );
    
    //        d100 = log10((double) dp[addX]); dp += addY;
    cpy_3ptr( pos_ptr+addX, cell_pos+3 );
    pos_ptr += addY;
    
    //        d010 = log10((double) dp[0]);
    cpy_3ptr( pos_ptr, cell_pos+6 );
    
    //        d110 = log10((double) dp[addX]); dp += addZ;
    cpy_3ptr( pos_ptr+addX, cell_pos+9 );
    pos_ptr += addZ;
    
    //        d011 = log10((double) dp[0]);
    cpy_3ptr( pos_ptr, cell_pos+12 );
    
    //        d111 = log10((double) dp[addX]); dp -= addY;
    cpy_3ptr( pos_ptr+addX, cell_pos+15 );
    pos_ptr -= addY;
    
    //        d001 = log10((double) dp[0]);
    cpy_3ptr( pos_ptr, cell_pos+18 );
    
    //        d101 = log10((double) dp[addX]);
    cpy_3ptr( pos_ptr+addX, cell_pos+21 );
    
#else // important: this order corresponds to the order of the 5 tet table 'tet_conn' !!!
    cpy_3ptr( pos_ptr, cell_pos );
    
    //        d100 = log10((double) dp[addX]); dp += addY;
    cpy_3ptr( pos_ptr+addX, cell_pos+12 );
    pos_ptr += addY;
    
    //        d010 = log10((double) dp[0]);
    cpy_3ptr( pos_ptr, cell_pos+6 );
    
    //        d110 = log10((double) dp[addX]); dp += addZ;
    cpy_3ptr( pos_ptr+addX, cell_pos+18 );
    pos_ptr += addZ;
    
    //        d011 = log10((double) dp[0]);
    cpy_3ptr( pos_ptr, cell_pos+9 );
    
    //        d111 = log10((double) dp[addX]); dp -= addY;
    cpy_3ptr( pos_ptr+addX, cell_pos+21 );
    pos_ptr -= addY;
    
    //        d001 = log10((double) dp[0]);
    cpy_3ptr( pos_ptr, cell_pos+3 );
    
    //        d101 = log10((double) dp[addX]);
    cpy_3ptr( pos_ptr+addX, cell_pos+15 );
#endif
    
}




template <typename M, typename T>
__device__ void resample_meta_patch(const int cell_num,
                                    const T* positions,
                                    const int meta_cells_per_patch,
                                    const T* bbox,
                                    const int dims,
                                    M* h_grid_mass)
{
    
    
    assert( cell_num < (meta_cells_per_patch*meta_cells_per_patch*meta_cells_per_patch) );
    
    const int grid_dims[3] = { dims, dims, dims };
    const dbox<T> grid_bbox( dvec3<T>(bbox[0],bbox[2], bbox[4]), dvec3<T>(bbox[1],bbox[3], bbox[5]) );
 
    
    // extract the 8 cell vertices
    float cell_pos[24];
    
    sample_cell_vertices(cell_num, positions, meta_cells_per_patch, cell_pos);
    
    // check if any of the vertices == 666.f - if yes-> return
    for ( int i=0; i<24; i+=3 )
    {
        // hack/to-do: magic number that indicates that this is a cell outside the Lagrangian domain ... bad style ...
        if ( fabsf(cell_pos[i]-666.f) + fabsf(cell_pos[i+1]-666.f) + fabsf(cell_pos[i+2]-666.f) < 1.E-03 )
        {
            // nothing to do for this cell, so lets return
            return;
        }
        
    }
        
    // construct each of the 5 tets
    for ( int t=0; t<6; ++t )
    {
        dvec3<T> tet_vertices[4];
        for ( int v=0; v<4; ++v )
        {
            const size_t idx =  3*(tet_conn[t][v]);
            tet_vertices [v] = dvec3<T>( cell_pos[idx],cell_pos[idx+1],cell_pos[idx+2] );
        }
        
        deposit_tet_2_grid<M,T>( tet_vertices, 1.f, grid_dims, grid_bbox,  h_grid_mass );
    }
    
}



#endif

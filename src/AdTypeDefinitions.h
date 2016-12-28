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


#ifndef _AD_TYPE_DEFINITIONS_
#define _AD_TYPE_DEFINITIONS_

#if __cplusplus <= 199711L
#include <tr1/memory>
#define shared_ptr tr1::shared_ptr
#else
#include <memory>
#endif


#include "AdException.h"
#include "AdVecND.h"
#include "AdUtils.h"
#include "AdOctree.h"

namespace AdaptiveMassDeposit
{
    
    // reserve largest uint65_t to mask invalid tet ids
    const uint64_t INVALID_TET_ID = ~(uint64_t(0));
    
    typedef float           pos_t;
    typedef float           vel_t;
    typedef float           mass_t;
    
    typedef AdVec3D<int> AvVec3i;
    
    typedef AdVec3D<pos_t> PosVec;
    
    class AABBox
    {
    public:
        AABBox() {};
        AABBox( const PosVec& mi, const PosVec& ma ) : min(mi), max(ma) {}
        inline pos_t get_diag2() const { return (max-min).length2(); }
        inline pos_t get_extension( const int i ) const { return max[i]-min[i]; }
        inline pos_t getVolume() const { return ( (max[0]-min[0])*(max[1]-min[1])*(max[2]-min[2]) ); }
        inline bool inside( const PosVec& pos) const
        {
            return ( (pos[0]>=min[0]&&pos[0]<=max[0]) && (pos[1]>=min[1]&&pos[1]<=max[1]) && (pos[2]>=min[2]&&pos[2]<=max[2]) );
        }
        inline void print() const { min.print(); max.print(); }
        PosVec min;
        PosVec max;
    };
    
    typedef size_t          ids_t;
    
    
    
    class RowMajorOrder
    {
        
    public:
        RowMajorOrder()
        {
            assert( unit_test_()==true );
        }
        
        static inline AvVec3i map_linear_to_3D_idx( const uint64_t idx, const AvVec3i& global_dims )
        {
            
            assert( idx < uint64_t(global_dims[0])*uint64_t(global_dims[1])*uint64_t(global_dims[2]) );
            
            const uint64_t nb  = global_dims[2];
            const uint64_t nb2 = global_dims[2]*global_dims[1];
            
            uint64_t pid = idx;
            const uint64_t ix = uint64_t( pid/nb2 );
            pid -= ix*nb2;
            const uint64_t iy = uint64_t(pid/nb);
            const uint64_t iz = pid-iy*nb;
   
            assert( ix<global_dims[0] );
            assert( iy<global_dims[1] );
            assert( iz<global_dims[2] );
            
            
            return AvVec3i(ix,iy,iz);
        }
        
        static inline uint64_t map_3D_to_linear_idx( const AvVec3i& idx_3d, const AvVec3i& dims )
        {
            assert( idx_3d[0]>=0 && idx_3d[0] < dims[0] );
            assert( idx_3d[1]>=0 && idx_3d[1] < dims[1] );
            assert( idx_3d[2]>=0 && idx_3d[2] < dims[2] );
            
            return uint64_t(idx_3d[2]) + uint64_t(dims[2]) * ( uint64_t(idx_3d[1]) + uint64_t(idx_3d[0])*uint64_t(dims[1]) ) ;
            
        }
        
        static inline uint64_t map_3D_to_linear_idx( const int* idx_3d, const int* dims )
        {
            assert( idx_3d[0]>=0 && idx_3d[0] < dims[0] );
            assert( idx_3d[1]>=0 && idx_3d[1] < dims[1] );
            assert( idx_3d[2]>=0 && idx_3d[2] < dims[2] );
            
            return uint64_t(idx_3d[2]) + uint64_t(dims[2]) * ( uint64_t(idx_3d[1]) + uint64_t(idx_3d[0])*uint64_t(dims[1]) ) ;
        }
        
    private:
        bool unit_test_()
        {
            
            const AvVec3i dims(3, 5, 2 );
            
            size_t linear_id = 0;
       
            for ( int i=0; i<dims[0]; ++i )
                for ( int j=0; j<dims[1]; ++j )
                    for ( int k=0; k<dims[2]; ++k )
                    {
                        const AvVec3i index_3d = map_linear_to_3D_idx( linear_id, dims );
                        
                        if ( index_3d[0]!=i || index_3d[1]!=j || index_3d[2]!=k )
                        {
                            assert(0);
                            return false;
                        }
                        
                        if ( map_3D_to_linear_idx( AvVec3i(i,j,k), dims) != linear_id )
                        {
                            assert(0);
                            return false;
                        }
                        
                        ++linear_id;
                    }
            
            
            return true;
            
            
        }

    };

    
    
    class ColumnMajorOrder
    {
    public:
        
        ColumnMajorOrder()
        {
            assert( unit_test_()==true );
        }
        
        static inline AvVec3i map_linear_to_3D_idx( const uint64_t idx, const AvVec3i& global_dims )
        {
            assert( global_dims[0]>=0 && global_dims[1]>=0 && global_dims[2]>=0 );
            assert( idx<size_t(global_dims[0])*size_t(global_dims[1])*size_t(global_dims[2]) );
            
            const uint64_t nb  = global_dims[0];
            const uint64_t nb2 = uint64_t(global_dims[0])*uint64_t(global_dims[1]);
            
            uint64_t i,j,k;
            
            k = idx/(nb2);
            const uint64_t rest1 = idx%(nb2);
            j = rest1/nb;
            i = rest1%nb;
            
            AD_ASSERT( int(i)<global_dims[0] && int(j)<global_dims[1] && int(k)<global_dims[2], "bug" );
            
            return AvVec3i(i,j,k);
        }
        
        static inline uint64_t map_3D_to_linear_idx( const AvVec3i& idx_3d, const AvVec3i& dims )
        {
            assert( idx_3d[0]>=0 && idx_3d[0]<dims[0] );
            assert( idx_3d[1]>=0 && idx_3d[1]<dims[1] );
            assert( idx_3d[2]>=0 && idx_3d[2]<dims[2] );
            
            return uint64_t(idx_3d[0]) + uint64_t(dims[0]) * ( uint64_t(idx_3d[1]) + uint64_t(idx_3d[2])*uint64_t(dims[1]) ) ;
        }
        
        template<typename T>
        static inline bool increment_3D_index( const AdVec3D<T>& min_idx, const AdVec3D<T>& max_idx, AdVec3D<T>& current_idx )
        {
            assert( current_idx[0]>=min_idx[0] && current_idx[0]<=max_idx[0] );
            assert( current_idx[1]>=min_idx[1] && current_idx[1]<=max_idx[1] );
            assert( current_idx[2]>=min_idx[2] && current_idx[2]<=max_idx[2] );
            
            if ( current_idx[0]<max_idx[0] )
            {
                ++current_idx[0];
                return true;
            }
            else if ( current_idx[1]<max_idx[1] )
            {
                current_idx[0]=min_idx[0];
                ++current_idx[1];
                return true;
            }
            else if ( current_idx[2]<max_idx[2] )
            {
                current_idx[0]=min_idx[0];
                current_idx[1]=min_idx[1];
                ++current_idx[2];
                return true;
            }

            return false;
        }
        
        
        
        template<typename T>
        static inline void get_cell_ids( const size_t offset, const AdVec3D<T>& add, const AdVec3D<T>& dims, size_t cell_ids[8] )
        {
            
            assert( dims[0]>=2 && dims[1]>=2 && dims[2]>=2 );
            //assert( offset< (dims[0]-1)*(dims[1]-1)*(dims[2]-1) );
            
            size_t dp = offset;
            
            const size_t add_x = add[0];
            const size_t add_y = add[1];
            // we could get rid of this and pre-compute it ...
            const size_t add_z = add[2];
            
            
            //cell_ids[0] = id_functor::map_3D_to_linear_idx( AvVec3i( i  , j  , k   ), tets_.get_patch_dims() );
            cell_ids[0] = dp;
            
            // cell_ids[4] = id_functor::map_3D_to_linear_idx( AvVec3i( i+1, j  , k   ), tets_.get_patch_dims() );
            cell_ids[4] = dp+add_x;
            dp += add_y;
            
            // cell_ids[2] = id_functor::map_3D_to_linear_idx( AvVec3i( i  , j+1, k   ), tets_.get_patch_dims() );
            cell_ids[2] = dp;
            
            //cell_ids[6] = id_functor::map_3D_to_linear_idx( AvVec3i( i+1, j+1, k   ), tets_.get_patch_dims() );
            cell_ids[6] = dp+add_x;
            dp += add_z;
            
            //cell_ids[3] = id_functor::map_3D_to_linear_idx( AvVec3i( i  , j+1, k+1 ), tets_.get_patch_dims() );
            cell_ids[3] = dp;
            
            //  cell_ids[7] = id_functor::map_3D_to_linear_idx( AvVec3i( i+1, j+1, k+1 ), tets_.get_patch_dims() );
            cell_ids[7] = dp+add_x;
            dp -= add_y;
            
            //cell_ids[1] = id_functor::map_3D_to_linear_idx( AvVec3i( i  , j,   k+1 ), tets_.get_patch_dims() );
            cell_ids[1] = dp;
            
            //cell_ids[5] = id_functor::map_3D_to_linear_idx( AvVec3i( i+1, j,   k+1 ), tets_.get_patch_dims() );
            cell_ids[5] = dp + add_x;
            
            
        }
        
        
        
    private:
        bool unit_test_()
        {
            
            const AvVec3i dims(3, 5, 2 );
            
            size_t linear_id = 0;
            
            AdVec3D<int> idx(0);
            do
            {
                
                const AvVec3i index_3d = map_linear_to_3D_idx( linear_id, dims );
            
                if ( index_3d[0]!=idx[0] || index_3d[1]!=idx[1] || index_3d[2]!=idx[2] )
                {
                    assert(0);
                    return false;
                }
                        
                if ( map_3D_to_linear_idx( idx, dims) != linear_id )
                {
                    assert(0);
                    return false;
                }
                    
                ++linear_id;
            } while (increment_3D_index( AdVec3D<int>(0,0,0), AdVec3D<int>(dims[0]-1,dims[1]-1,dims[2]-1), idx ));
                
            
            return true;
            
            
        }
        
        
    };

    
    static inline bool intersection(const AABBox& b1, const AABBox& b2)
    {
        return (b1.min[0] < b2.max[0] && b2.min[0] < b1.max[0] &&
                b1.min[1] < b2.max[1] && b2.min[1] < b1.max[1] &&
                b1.min[2] < b2.max[2] && b2.min[2] < b1.max[2]);
    }
    
    
    static inline bool get_intersection( const AABBox& region1, const AABBox& region2, AABBox& result )
    {
        
        if ( intersection(region1, region2)==false )
        {
            return false;
        }
        else
        {
            PosVec& ll = result.min;
            PosVec& ur = result.max;
            for ( int i=0; i<3; i++  )
            {
                ll[i] = ( region1.min[i] < region2.min[i] )  ? region2.min[i] : region1.min[i];
                ur[i] = ( region1.max[i] < region2.max[i] )  ? region1.max[i] : region2.max[i];
            }
        }
        
        return true;
        
    }
    
    
    // to-do: this struct wastes 4 bytes due to padding/alignment issues ...
    struct particle_with_id
    {
        
        particle_with_id( ) : id(0)
        {
            
        }
        
        inline void print() const
        {
            std::cout << pos[0] << "  " << pos[1] << "  " << pos[2] << "  " /* << id << " " */ << std::endl;
        }
        
        PosVec pos;
        //vel_t vx, vy, vz;
        //mass_t mass;
        ids_t id;
    };

    
    
    
#if 1
    class Particles
    {
        
    public:
        typedef std::vector<pos_t> pos_t_vec;
        typedef std::vector<ids_t> ids_t_vec;
        
        typedef std::shared_ptr<pos_t_vec> pos_t_ptr;
        typedef std::shared_ptr<ids_t_vec> ids_t_ptr;
        
        
    public:
        Particles() :
        pos_x_( new pos_t_vec() ),
        pos_y_( new pos_t_vec() ),
        pos_z_( new pos_t_vec() ),
        ids_( new ids_t_vec() )
        {
            if( check_invariant_()==false )
            {
                throw AdRuntimeException("ERROR: Particles: failed to construct class.");
            }
        }

        inline size_t get_num() const
        {
            assert( check_invariant_() );
            return pos_x_->size();
        }
        
        inline void clear()
        {
            pos_x_.reset(new pos_t_vec());
            pos_y_.reset(new pos_t_vec());
            pos_z_.reset(new pos_t_vec());
            ids_.reset(new ids_t_vec());
        }
        
        inline void set_num_positions( const size_t num )
        {
            pos_x_->resize(num);
            pos_y_->resize(num);
            pos_z_->resize(num);
        }
        
        inline void set_num_ids( const size_t num)
        {
            assert( check_invariant_() );
            ids_->resize(num);
            assert( check_invariant_() );
        }
        
        inline void set_position( const size_t num, const pos_t x,  const pos_t y, const pos_t z )
        {
            assert( pos_x_->size()>num );
            assert( check_invariant_() );
            
            (*pos_x_)[num] = x;
            (*pos_y_)[num] = y;
            (*pos_z_)[num] = z;
            
            assert( check_invariant_() );
            
        }
        
        inline void push_back( const pos_t x,  const pos_t y, const pos_t z, const ids_t id )
        {
            assert( check_invariant_() );
          
            const size_t num = get_num();
            
            pos_x_->resize(num+1);
            pos_y_->resize(num+1);
            pos_z_->resize(num+1);
            
            (*pos_x_)[num] = x;
            (*pos_y_)[num] = y;
            (*pos_z_)[num] = z;
            
            ids_->resize(num+1);
            (*ids_)[num] = id;
            
            assert( check_invariant_() );
            
        }
        
        inline void set_id ( const size_t num, const ids_t id )
        {
            assert( check_invariant_() );
            assert( ids_.get() && ids_->size()>num );
            (*ids_)[num] = id;
            assert( check_invariant_() );
        }
        
        
        inline void set_position( const size_t num, const pos_t* pos )
        {
            assert( check_invariant_() );
            set_position( num, pos[0], pos[1], pos[2]);
            assert( check_invariant_() );
        }
        
        inline void set_ids( ids_t_ptr ids )
        {
            assert( check_invariant_() );
            if ( ids.get()==0 )
            {
                throw AdRuntimeException("ERROR: Positions::set_ids(): invalid ids pointer.");
            }
            
            if ( !pos_x_->empty() && pos_x_->size() != ids->size() )
            {
                throw AdRuntimeException("ERROR: Positions::set_ids(): inconsistent ids size.");
            }
            
            ids_ = ids;
            assert( check_invariant_() );
        }
        
        inline ids_t get_id ( const size_t num ) const
        {
            assert( check_invariant_() );
            assert( ids_->size()>num );
            return (*ids_)[num];
        }
        
        inline void set_positions( pos_t_ptr px, pos_t_ptr py, pos_t_ptr pz )
        {
            
            if ( px.get()==0 || py.get()==0 || pz.get()==0 )
            {
                throw AdRuntimeException("ERROR: Positions::set_positions(): invalid positions pointers.");
            }
            
            if ( px->size()!=py->size() || px->size()!=pz->size() || (!ids_->empty() && ids_->size()!=px->size()) )
            {
                throw AdRuntimeException("ERROR: Positions::set_positions(): invalid position arrays sizes");
            }
            
            
            pos_x_ = px;
            pos_y_ = py;
            pos_z_ = pz;
            
        }
        
        inline void get_position( const size_t num, pos_t& x, pos_t& y, pos_t& z ) const
        {
            assert( pos_x_->size()>num );
            assert( check_invariant_() );
            x = (*pos_x_)[num];
            y = (*pos_y_)[num];
            z = (*pos_z_)[num];
        }
        
        inline void get_position( const size_t num, pos_t* pos ) const
        {
            get_position( num, pos[0], pos[1], pos[2] );
            
        }
      
        inline  pos_t_ptr get_positions( const unsigned int num ) const
        {
            assert( pos_x_->size()== pos_y_->size() && pos_y_->size()==pos_z_->size() );
            
            if ( num==0 )
                return pos_x_;
            else if (num==1)
                return pos_y_;
            return pos_z_;
        }
        
        inline ids_t_ptr get_ids( ) const
        {
            return ids_;
        }
        
        
    private:
        inline bool check_invariant_() const
        {
            if ( pos_x_.get()==0 || pos_y_.get()==0 || pos_z_.get()==0  || ids_.get()==0 )
            {
                assert(0);
                return false;
            }

            
            if ( pos_x_->size() != pos_y_->size() || pos_y_->size() != pos_z_->size() )
            {
                assert(0);
                return false;
            }
            
            //if ( !masses_.empty() && masses_.size()!=pos_x_.size() )
            //{
            //    return false;
            //}
            
            if ( !ids_->empty() && ids_->size()!=pos_x_->size() )
            {
                assert(0);
                return false;
            }
            
            return true;
            
        }
        
        std::shared_ptr< std::vector<pos_t> > pos_x_;
        std::shared_ptr< std::vector<pos_t> > pos_y_;
        std::shared_ptr< std::vector<pos_t> > pos_z_;
        std::shared_ptr< std::vector<ids_t> > ids_;
        //std::vector<mass_t> masses_;
        //std::vector<ids_t> ids_;
        
    };
    
    typedef Particles particles_with_ids_chunk;
    
    
    
#endif
    
    template <class id_functor> class Tetrahedra
    {
    
                
    public:
        
        Tetrahedra() : patch_dims_(0), num_cells_(0)
        {
        }
        
        inline size_t get_num_tets() const
        {
            return 6*num_cells_;
        }
        
        inline void clear()
        {
            particles.clear();
        }
        
        inline const AvVec3i& get_patch_dims() const
        {
            return patch_dims_;
        }
        
        inline void set_patch_dims( const unsigned int dx, const unsigned int dy, const unsigned int dz )
        {
            
            if ( particles.get_num()!= (size_t(dx)*size_t(dy)*size_t(dz)) ) // || dx<2 || dy<2 || dz<2 )
            {
                std::cout << particles.get_num() << " " << dx << " " << dy << " " << dz << std::endl;
                throw AdRuntimeException( "ERROR: Tetrahedra::set_patch_dims(): Inconsistent patch dimensions.", true );
            }
            
            patch_dims_[0] = dx;
            patch_dims_[1] = dy;
            patch_dims_[2] = dz;
            
            num_cells_ = (dx==0 || dy==0 || dz==0) ? 0 : (size_t(dx-1)*size_t(dy-1)*size_t(dz-1));
            
        }
        
        Particles particles;
  
    private:
        
       
        AvVec3i patch_dims_;
        uint64_t num_cells_;
        
    
    };
    
    
    static inline AABBox get_bbox( const size_t num_points, const pos_t* interleaved_coords )
    {
        AD_ASSERT( num_points>0, "" );
        
        AABBox res;
        
        if ( interleaved_coords==0 || num_points==0 )
        {
            return res;
        }
        
        
        res.min = PosVec( interleaved_coords[0],interleaved_coords[1],interleaved_coords[2] ); //[0],points[ids[0]].y,points[ids[0]].z);
        res.max = res.min;
        
        size_t pc = 3;
        for ( size_t i=1; i<num_points; ++i )
        {
            
            const pos_t* p = &interleaved_coords[pc];
            
            if      ( p[0]<res.min[0] ) res.min[0] = p[0];
            else if ( p[0]>res.max[0] ) res.max[0] = p[0];
            
            if      ( p[1]<res.min[1] ) res.min[1] = p[1];
            else if ( p[1]>res.max[1] ) res.max[1] = p[1];
            
            if      ( p[2]<res.min[2] ) res.min[2] = p[2];
            else if ( p[2]>res.max[2] ) res.max[2] = p[2];

            pc += 3;
        
        }
        
        assert( pc/3==num_points );
        
        return res;

    }
    
    
    
    
    template <class id_functor>
    class MetaMesh
    {
        
    public:
        
        MetaMesh(const Tetrahedra<id_functor>& tets) :
        tets_(tets)
        {

            if ( tets.get_num_tets()==0 )
            {
                cell_dims_ = AvVec3i( 0, 0, 0 );
            }
            else
            {
                assert( tets.get_patch_dims()[0]>=2 && tets.get_patch_dims()[1]>=2 && tets.get_patch_dims()[2]>=2 );
                cell_dims_ =  AvVec3i(ceil_div(tets.get_patch_dims()[0]-1, get_number_of_cells_per_dimension() ),
                                      ceil_div(tets.get_patch_dims()[1]-1, get_number_of_cells_per_dimension() ),
                                      ceil_div(tets.get_patch_dims()[2]-1, get_number_of_cells_per_dimension() ));
            }
            
            generate_meta_cells_();

            assert( unit_tests_passed_() );
        
        }
        
        static inline int get_number_of_cells_per_dimension()
        {
            // to-do: this should be a user-defined parameter
            return 7;
        }
        
        static inline int get_number_of_vertices_per_meta_block()
        {
            return (get_number_of_cells_per_dimension()+1)*(get_number_of_cells_per_dimension()+1)*(get_number_of_cells_per_dimension()+1);
        }
        
        
        
        inline const AABBox& get_bbox( const uint64_t meta_cell_id ) const
        {
            return meta_cells_[meta_cell_id].bbox;
        }
        
        
        inline void get_tets( const AABBox& box, const uint64_t meta_cells_id, std::vector<pos_t>& interleaved_positions ) const
        {
            
            interleaved_positions.clear();
            interleaved_positions.resize( 3*get_number_of_vertices_per_meta_block(), 666.f ); // 8*8*8*3
            
            
            AvVec3i min_idx;
            AvVec3i dims;
            
            get_meta_cell_index_range_( meta_cells_id, min_idx, dims );
            
            
            const AvVec3i max_idx(min_idx[0]+dims[0]-1,min_idx[1]+dims[1]-1,min_idx[2]+dims[2]-1);
            
            AdVec3D<int> idx = min_idx;
            
            assert( max_idx[0]-min_idx[0] <= get_number_of_cells_per_dimension()+1);
            assert( max_idx[1]-min_idx[1] <= get_number_of_cells_per_dimension()+1);
            assert( max_idx[2]-min_idx[2] <= get_number_of_cells_per_dimension()+1);
            
            for ( size_t k=min_idx[2]; k<=max_idx[2]; ++k )
            {
                for ( size_t j=min_idx[1]; j<=max_idx[1]; ++j )
                {
                    size_t offset_src = id_functor::map_3D_to_linear_idx( AdVec3D<int>(min_idx[0],j,k), tets_.get_patch_dims() );
                    size_t offset_dst = 3*id_functor::map_3D_to_linear_idx(AdVec3D<int>(0,j-min_idx[1],k-min_idx[2]),
                                                                           AdVec3D<int>((get_number_of_cells_per_dimension()+1),
                                                                                                  (get_number_of_cells_per_dimension()+1),
                                                                                                  (get_number_of_cells_per_dimension()+1) ));
                    for ( size_t i=min_idx[0]; i<=max_idx[0]; ++i )
                    {
                        
                        assert( i<tets_.get_patch_dims()[0] );
                        assert( j<tets_.get_patch_dims()[1] );
                        assert( k<tets_.get_patch_dims()[2] );
                        
                        assert( (offset_dst+3) <= interleaved_positions.size() );
                   
                        tets_.particles.get_position( offset_src, &interleaved_positions[0] + offset_dst );
                        
                        ++offset_src;
                        offset_dst+=3;
                        
                    }
                }
            }
            
        }
        
        inline uint64_t get_estimate_num_tets( const AABBox& box, const uint64_t meta_cell_id ) const
        {
            AvVec3i min_idx;
            AvVec3i dims;
            get_meta_cell_index_range_( meta_cell_id, min_idx, dims );
            
            const size_t total_cells_in_meta_cell = size_t(dims[0]-1)*size_t(dims[1]-1)*size_t(dims[2]-1);
            
            AABBox intersection_box;
            get_intersection( box, meta_cells_[meta_cell_id].bbox, intersection_box );
            
            return 6*total_cells_in_meta_cell*( intersection_box.getVolume())/(meta_cells_[meta_cell_id].bbox.getVolume() );
            
        }

        inline void get_cell_ids ( std::vector<uint64_t>& ids ) const
        {
            ids.resize(meta_cells_.size());
            for ( size_t i=0; i<meta_cells_.size(); ++i )
            {
                // currently we are using trivial ids
                ids[i] = i;
            }
        }
        
        
        
    private:
        
        
        bool unit_tests_passed_()
        {
            {
                
                AvVec3i min_idx;
                AvVec3i dims;
                
                if ( cell_dims_[0]>0 )
                {
                    get_meta_cell_index_range_( cell_dims_[0]-1, cell_dims_[1]-1, cell_dims_[2]-1, min_idx, dims );
                
                    if (min_idx[0]+dims[0]!=tets_.get_patch_dims()[0] ||
                        min_idx[1]+dims[1]!=tets_.get_patch_dims()[1] ||
                        min_idx[2]+dims[2]!=tets_.get_patch_dims()[2])
                    {
                        assert(0);
                        return false;
                    }
                }
                
                if ( meta_cells_.empty()== false )
                {
                    get_meta_cell_index_range_( meta_cells_.size()-1, min_idx, dims );
                
                    if (min_idx[0]+dims[0]!=tets_.get_patch_dims()[0] ||
                        min_idx[1]+dims[1]!=tets_.get_patch_dims()[1] ||
                        min_idx[2]+dims[2]!=tets_.get_patch_dims()[2])
                    {
                        assert(0);
                        return false;
                    }
                }
                
                for ( int k=0; k<cell_dims_[2]; ++k )
                {
                    get_meta_cell_index_range_( 0, 0, k, min_idx, dims );
                    
                    min_idx.print();
                    dims.print();
                    std::cout << std::endl;
                    
                }
                
                for ( int k=0; k<cell_dims_[2]; ++k )
                {
                    get_meta_cell_index_range_( k, min_idx, dims );
                    
                    min_idx.print();
                    dims.print();
                    std::cout << std::endl;
                    
                }
                
            }
            
            return true;
        }
        
        struct MetaCell
        {
            AABBox bbox;
        };
        
        inline void get_meta_cell_index_range_( const uint64_t meta_cell_index, AvVec3i& min_idx, AvVec3i& dims ) const
        {
            
            assert( meta_cell_index<meta_cells_.size() );
            
            const AvVec3i idx_3d = id_functor::map_linear_to_3D_idx( meta_cell_index, cell_dims_ );
        
            get_meta_cell_index_range_( idx_3d[0], idx_3d[1], idx_3d[2], min_idx, dims );
            
        }
        
        
        void get_meta_cell_index_range_( const unsigned int cx, const unsigned int cy, const unsigned int cz, AvVec3i& min_idx, AvVec3i& dims ) const
        {
            
            if ( cx>=cell_dims_[0] || cy>=cell_dims_[1] || cz>=cell_dims_[2] )
            {
                min_idx = AvVec3i( 0, 0, 0 );
                dims = AvVec3i( 0, 0, 0 );
                return;
            }
            
            assert( tets_.get_num_tets()>0 );
            
            const AvVec3i& pd = tets_.get_patch_dims();
            
            assert( pd[0]>1 && pd[1]>1 && pd[2]>1 );
            
            const AvVec3i cells_per_meta_cell(get_number_of_cells_per_dimension(),get_number_of_cells_per_dimension(),get_number_of_cells_per_dimension());
           
            
            min_idx = AvVec3i( cx*cells_per_meta_cell[0], cy*cells_per_meta_cell[1], cz*cells_per_meta_cell[2] );
            
            assert( min_idx[0]<(pd[0]-1) && min_idx[1]<(pd[1]-1) && min_idx[2]<(pd[2]-1) );
            
            
            dims[0] = (cx+1==cell_dims_[0]) ?  (pd[0]-min_idx[0]) : cells_per_meta_cell[0]+1;
            dims[1] = (cy+1==cell_dims_[1]) ?  (pd[1]-min_idx[1]) : cells_per_meta_cell[1]+1;
            dims[2] = (cz+1==cell_dims_[2]) ?  (pd[2]-min_idx[2]) : cells_per_meta_cell[2]+1;

            assert( dims[0]<=get_number_of_cells_per_dimension()+1);
            assert( dims[1]<=get_number_of_cells_per_dimension()+1);
            assert( dims[2]<=get_number_of_cells_per_dimension()+1);
            assert( dims[0]>1 && dims[1]>1 && dims[2]>1 );
            assert( (min_idx[0]+dims[0])<=tets_.get_patch_dims()[0] );
            assert( (min_idx[1]+dims[1])<=tets_.get_patch_dims()[1] );
            assert( (min_idx[2]+dims[2])<=tets_.get_patch_dims()[2] );
            
        }
        
        
        void generate_meta_cells_()
        {
            
            meta_cells_.clear();
            meta_cells_.resize( size_t(cell_dims_[0])*size_t(cell_dims_[1])*size_t(cell_dims_[2]) );
            
            
            if ( meta_cells_.empty() )
                return;
            
            AD_VERBOSE( 3, { std::cout << "INFO: generate_meta_cells_(): generating " << meta_cells_.size() << " meta cells"  << std::endl; } );
            
            size_t mc_counter = 0;
            
            
            AdVec3D<int> idx(0);
            do
            {
                
                const int mci = idx[0];
                const int mcj = idx[1];
                const int mck = idx[2];
                
                AvVec3i min_idx;
                AvVec3i dims;
                
                get_meta_cell_index_range_( mci, mcj, mck, min_idx, dims );
                
                
                // initialize bbox
                pos_t particle[3];
                tets_.particles.get_position( id_functor::map_3D_to_linear_idx( min_idx, tets_.get_patch_dims() ), particle );
                
                MetaCell& mc = meta_cells_[mc_counter];
                
                mc.bbox.min = PosVec(particle[0], particle[1], particle[2] );
                mc.bbox.max = mc.bbox.min;
                
                
                AdVec3D<int> p(0);
                do
                {
                    tets_.particles.get_position( id_functor::map_3D_to_linear_idx( min_idx+p, tets_.get_patch_dims() ), particle );
                    for ( int b=0; b<3; ++b )
                    {
                        mc.bbox.min[b] = std::min(mc.bbox.min[b],particle[b] );
                        mc.bbox.max[b] = std::max(mc.bbox.max[b],particle[b] );
                    }
                } while ( id_functor::increment_3D_index( AvVec3i(0), AvVec3i(dims[0]-1,dims[1]-1,dims[2]-1), p ) );
                
                
                // increase meta_cell counter;
                ++mc_counter;
                
            }  while ( id_functor::increment_3D_index( AvVec3i(0), AvVec3i(cell_dims_[0]-1,cell_dims_[1]-1,cell_dims_[2]-1), idx ) );
            
            
            
        }
        
        const Tetrahedra<id_functor>& tets_;
        
        AvVec3i cell_dims_;
        
        std::vector<MetaCell> meta_cells_;
        
    };
    
    
    
    class OctreeDataType
    {
    public:
        OctreeDataType() :
        num_tets_covering_node_estimate(0)
        //cost_factor(0)
        {}
        
        
        inline void print_info() const
        {
            std::cout << "INFO: OctreeDataType::print_info(): Node: covered tets (estimate) = " << num_tets_covering_node_estimate << " and " << local_meta_cell_ids.size() << " meta cells" << std::endl;
        }

        /*
         this array stores (an estimate of) how many tets each proc owns that cover this node
         required to minimized data transfer during load balancing
         */
        // number of tets assigned to this node
        uint64_t num_tets_covering_node_estimate;
        std::vector<uint64_t> local_meta_cell_ids;
        std::vector< uint64_t > num_tets_on_proc_estimate;
        
        
    };
    
    
    typedef Octree< pos_t, OctreeDataType > TetOctree;
    
    
    
    class FindCommunicationPartner
    {
        
    public:
        FindCommunicationPartner( const int my_rank, const int num_ranks ) :
        my_rank_(my_rank), num_ranks_(num_ranks), counter_(0)
        {
            if ( num_ranks_<=0 || num_ranks_<=my_rank_ /* || (num_ranks_>1 && num_ranks_%2!=0)*/ )
            {
                throw AdRuntimeException("ERROR: unsupported number of ranks.");
            }
            
            // initialize communication mask
            reset();
            
            assert( unit_test( ) );
            
        }
        
        void reset()
        {
            // initialize communication mask
            communicated.clear();
            communicated.resize(num_ranks_);
            for ( size_t i=0; i<communicated.size(); ++i )
            {
                communicated[i].resize( num_ranks_, false );
            }
            counter_ = 0;
            
        }
        
        int get_next_communication_partner(  bool& done )
        {
            
            assert( counter_<num_ranks_ );
            
            if ( counter_ == num_ranks_-1 )
            {
                done = true;
                return -1;
            }
            else
            {
                done = false;
            }
            
            
            int my_partner = -1;
            
            assert( communicated.size()==num_ranks_ );
            
            
            std::vector< bool > available(num_ranks_,true);
            
            for ( int i=0; i<num_ranks_; ++i )
            {
                if ( !available[i] ) continue;
                
                for ( int j=i+1; j<num_ranks_; ++j )
                {
                    
                    if ( !available[j] ) continue;
                    
                    if ( communicated[i][j]==false && available[i] && available[j] )
                    {
                        assert( communicated[j][i]==false );
                        
                        communicated[i][j] = true;
                        communicated[j][i] = true;
                        available[i] = available[j] = false;
                        
                        if ( i==my_rank_ )
                        {
                            ++counter_;
                            my_partner = j;
                        }
                        else if ( j==my_rank_ )
                        {
                            ++counter_;
                            my_partner = i;
                        }
                        
                        break;
                        
                    }
                    
                }
            }
            //assert( my_partner>=0 && my_partner!=my_rank_ );
            
            return my_partner;
            
        }
        
        bool unit_test( )
        {
            
            reset();
            
            std::vector<bool> communication_partner( num_ranks_, false );
            
            bool done = false;
            while (!done)
            {
                const int next_partner = get_next_communication_partner(done);
                
                if ( next_partner>=0 )
                {
                    assert( communication_partner[next_partner] == false );
                    communication_partner[next_partner] = true;
                }
                //print_();
                
            }
            
            for ( int i=0; i<num_ranks_; ++i )
            {
                if ( i!=my_rank_ && communication_partner[i]==false )
                {
                    assert(0);
                    return false;
                }
            }
            
            reset();
            
            return true;
            
            
            
        }
        
        
        void print_() const
        {
            for ( int i=0; i<communicated.size(); ++i )
            {
                for ( int j=0; j<communicated[i].size(); ++j )
                {
                    std::cout << communicated[i][j] << "  ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        
        
        
    private:
        
        std::vector< std::vector< bool > > communicated;
        
        int my_rank_;
        int num_ranks_;
        int counter_;
        
    };

    
    
    class AMD_Parameters
    {
    public:
        
        AMD_Parameters() :
        use_regions_of_interest(false),
        max_refinement_level(5),
        stride(1),
        max_tets_per_octree_node(-1),
        linear_patch_resolution(64),
        num_pthreads(1),
        num_mpi_ranks_per_node(1),
        periodic_boundaries(true)
        {
        }
        
        std::string input_file;
        std::string output_directory;
        std::string id_type;
        bool use_regions_of_interest;
        AABBox region_of_interest;
        int max_refinement_level;
        int stride;
        uint64_t max_tets_per_octree_node;
        int linear_patch_resolution;
        int num_pthreads;
        int num_mpi_ranks_per_node;
        std::string restart_path;
        bool periodic_boundaries;
        
        inline void print() const
        {
            std::cout << "INFO: AMD_Parameters(): PARAMETERS INFO:\n";
            std::cout << "    input_file               = " << input_file << std::endl;  //INPUT
            std::cout << "    output_directory         = " << output_directory << std::endl;; //OUTPUT
            std::cout << "    id_type                  = " << id_type << std::endl;; //DARK_SKY
            if ( 	  use_regions_of_interest==true )
            {
                std::cout << "    use_region_of_interest   = true" << std::endl;; //true
            }
            else
            {
                std::cout << "    use_region_of_interest   = false" << std::endl;; //true
            }
            std::cout << "    region_of_interest       = " << std::endl;; //1. 2. 3. 4. 5. 6.
            region_of_interest.print();
            std::cout << "    max_refinement_level     = " << max_refinement_level << std::endl;; //10
            std::cout << "    stride                   = " << stride << std::endl;; // 1
            std::cout << "    max_tets_per_octree_node = " << max_tets_per_octree_node << std::endl;; //54321
            std::cout << "    linear_patch_resolution  = " << linear_patch_resolution << std::endl;; //64
            std::cout << "    num_pthreads             = " << num_pthreads << std::endl;
            std::cout << "    num_mpi_ranks_per_node   = " << num_mpi_ranks_per_node << std::endl;
            std::cout << "    restart_path             = " << restart_path << std::endl;
            std::cout << "    periodic_boundaries      = " << periodic_boundaries << std::endl;
            std::cout << "------------------\n";
        }
        
    };
    

    
};

#endif


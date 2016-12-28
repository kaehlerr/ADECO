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



#ifndef _AD_POINT_DATA_READERS_
#define _AD_POINT_DATA_READERS_

#include <cfloat>
#include <string>

#include "AdMeasureMPIWallClockTime.h"

#include "AdUtils.h"
#include "AdAssert.h"

#include "AdLagrangianRedistribution.h"


namespace AdaptiveMassDeposit
{



static inline AABBox get_global_bbox_mpi( const bool use_local_bbox, const AABBox& local_bbox )
{
    float loc_min[3] = {local_bbox.min[0],local_bbox.min[1],local_bbox.min[2]};
    float loc_max[3] = {local_bbox.max[0],local_bbox.max[1],local_bbox.max[2]};
    
    float global_bbox_min[3];
    float global_bbox_max[3];
    
    if ( use_local_bbox==false )
    {
        loc_min[0] = loc_min[1] = loc_min[2] = FLT_MAX;
        loc_max[0] = loc_max[1] = loc_max[2] = FLT_MIN;
    }
    
    CHECK_MPI_ERROR( MPI_Allreduce(loc_min, global_bbox_min, 3, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD) );
    CHECK_MPI_ERROR( MPI_Allreduce(loc_max, global_bbox_max, 3, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD) );
    
    AABBox res;
    res.min = PosVec(global_bbox_min[0],global_bbox_min[1],global_bbox_min[2]);
    res.max = PosVec(global_bbox_max[0],global_bbox_max[1],global_bbox_max[2]);
    
    return res;
    
}


static AABBox get_local_bbox( const std::vector<std::shared_ptr<particles_with_ids_chunk> >& particles_by_proc )
{
    
    AABBox box;
    
    bool has_local_particles = false;
    // first initialize bbox
    for ( size_t p=0; p<particles_by_proc.size(); ++p )
    {
        if ( particles_by_proc[p]->get_num()>0 )
        {
            pos_t pos[3];
            
            particles_by_proc[p]->get_position( 0, pos );
            box.min = PosVec( pos[0],pos[1],pos[2] );
            box.max = PosVec( pos[0],pos[1],pos[2] );
            
            has_local_particles = true;
            break;
        }
    }
    
    if ( has_local_particles==false )
    {
        throw AdRuntimeException("WARNING: get_local_bbox(): have no local particles.",false);
    }
    
    // to-do: try to optimize this using OpemMP
    for ( size_t p=0; p<particles_by_proc.size(); ++p )
    {
        const Particles::pos_t_vec pos_x_ptr = *particles_by_proc[p]->get_positions( 0 );
        const Particles::pos_t_vec pos_y_ptr = *particles_by_proc[p]->get_positions( 1 );
        const Particles::pos_t_vec pos_z_ptr = *particles_by_proc[p]->get_positions( 2 );
        
        for ( size_t i=0; i<pos_x_ptr.size(); ++i )
        {
            box.min = PosVec( std::min(pos_x_ptr[i],box.min[0]), std::min(pos_y_ptr[i],box.min[1]), std::min(pos_z_ptr[i],box.min[2]) );
            box.max = PosVec( std::max(pos_x_ptr[i],box.max[0]), std::max(pos_y_ptr[i],box.max[1]), std::max(pos_z_ptr[i],box.max[2]) );
        }
    }
    
    return box;
    
}


static AABBox get_global_bbox_mpi( const std::vector< std::shared_ptr<particles_with_ids_chunk> >& particles_by_proc )
{
    
    try
    {
        const AABBox local_bbox = get_local_bbox( particles_by_proc );
        return get_global_bbox_mpi( true, local_bbox );
    }
    catch( std::exception& ex )
    {
        AABBox local_bbox;
        std::cout << "WARNING: get_global_bbox_mpi(): Local process does not own any particles yet. Computing global bbox without it." << std::endl;
        return get_global_bbox_mpi( false, local_bbox );
    }
    
}


template <class ID_MAPPING_ORDER>
static void apply_boundary_corrections(const AABBox& bbox, const AvVec3i& grid_dims, const ID_MAPPING_ORDER& id_functor, particles_with_ids_chunk& particles )
{
    
    const pos_t box_ext[3]    = { bbox.get_extension(0),                    bbox.get_extension(1),                  bbox.get_extension(2)                   };
    const pos_t box_ext_h1[3] = { bbox.min[0]+box_ext[0]/pos_t(4.0),        bbox.min[1]+box_ext[1]/pos_t(4.0),      bbox.min[2]+box_ext[2]/pos_t(4.0)       };
    const pos_t box_ext_h2[3] = { bbox.min[0]+pos_t(3./4.)*box_ext[0],      bbox.min[1]+pos_t(3./4.)*box_ext[1],    bbox.min[2]+pos_t(3./4.)*box_ext[2]     };
    const pos_t td_h[3]       = { pos_t(grid_dims[0])/pos_t(2.0),           pos_t(grid_dims[1])/pos_t(2.0),         pos_t(grid_dims[2])/pos_t(2.0)          };
    
    
    Particles::pos_t_vec& px = *particles.get_positions(0);
    Particles::pos_t_vec& py = *particles.get_positions(1);
    Particles::pos_t_vec& pz = *particles.get_positions(2);
    Particles::ids_t_vec& ids = *particles.get_ids();
    
    // to-do: should/could parallelize this using OpenMP
    for ( size_t p=0; p<particles.get_num(); ++p )
    {
        
        // use id_functor to translate linear index into 3D index
        const AvVec3i idx = id_functor.map_linear_to_3D_idx( ids[p], grid_dims );
        
        pos_t pos[3] = { px[p], py[p], pz[p] };
        //const AvVislib::AvVec3D<T> pos(positions[idx],positions[idx+1],positions[idx+2]);
        
        bool need_update = false;
        for ( int i=0; i<3; ++i )
        {
            
            if ( idx[i]>td_h[i] && pos[i]<box_ext_h1[i] )
            {
                pos[i] += box_ext[i];
                need_update = true;
            }
            else if ( idx[i]<td_h[i] && pos[i]>box_ext_h2[i])
            {
                pos[i] -= box_ext[i];
                need_update = true;
            }
            
        }
        
        if ( need_update )
        {
            px[p] = pos[0];
            py[p] = pos[1];
            pz[p] = pos[2];
        }
        
        
    } // end loop over p
    
}

class AdPointDataReader
{
    
public:
    
    
    inline AvVec3i get_rank_dims() const
    {
        return procs_dims_;
    }
    
    inline AvVec3i get_global_particle_dims() const
    {
        return global_particle_dims_;
    }
    
    inline std::vector< std::shared_ptr<particles_with_ids_chunk> >& get_particles_per_rank( )
    {
        return particles_per_proc_;
    }
    
    inline size_t load_data(const std::string& filename,
                            const int my_rank,
                            const int num_procs,
                            const int stride )
    {
        if ( filename.empty() )
        {
            throw AdRuntimeException( "ERROR: AdPointDataReader::load_data(): missing filename." );
        }
        
        if (my_rank>=num_procs || my_rank<0 || num_procs<1 )
        {
            throw AdRuntimeException( "ERROR: AdPointDataReader::load_data(): invalid MPI rank parameters." );
        }
        
        if ( stride<1 )
        {
            throw AdRuntimeException( "ERROR: AdPointDataReader::load_data(): invalid 'stride' parameters." );
        }
        
        // allocate bins to store the data
        particles_per_proc_.resize( num_procs );
        for ( size_t i=0; i<particles_per_proc_.size(); ++i )
        {
            particles_per_proc_[i] = std::shared_ptr<particles_with_ids_chunk>(new particles_with_ids_chunk);
        }
        
        
        const size_t num_particles = load_data_(filename,
                                                my_rank,
                                                num_procs,
                                                stride,
                                                procs_dims_,
                                                global_particle_dims_,
                                                particles_per_proc_ );
        
        if ( num_particles == 0 )
        {
            std::cerr << "WARNING: AdPointDataReaders()::load_data(): call returned 0 particles on MPI rank == " << my_rank << std::endl;
        }
        
        AD_ASSERT( checkInvariant_( my_rank, num_procs ), "ERROR: AdPointDataReaders(): checkInvariants_() failed." );
        
        return num_particles;
        
    }
    
    
    
    
protected:
    virtual size_t load_data_(const std::string& filename,
                              const int my_rank,
                              const int num_procs,
                              const int stride,
                              AvVec3i& procs_dims,
                              AvVec3i& global_particle_dims,
                              std::vector< std::shared_ptr<particles_with_ids_chunk> >& particles_per_proc ) = 0;
private:
    
    
    bool checkInvariant_( const int my_rank, const int num_procs ) const
    {
        
        assert( procs_dims_[0]*procs_dims_[1]*procs_dims_[2] == num_procs );
        
        size_t particle_count = 0;
        
        for ( int p=0; p<particles_per_proc_.size(); ++p )
        {
            
            particle_count += particles_per_proc_[p]->get_num();
            
            for ( int i=0; i<particles_per_proc_[p]->get_num(); ++i )
            {
                // we are assuming column-major array order in the reaminder of the code, so readers have to remap if necessary
                const size_t lin_id = particles_per_proc_[p]->get_id(i);
                const AvVec3i lid = AdaptiveMassDeposit::ColumnMajorOrder::map_linear_to_3D_idx( lin_id, global_particle_dims_ );
                
                
                AvVec3i tmp_block_offset;
                AvVec3i tmp_block_dims;
                // check if point is really assigned to the correct process
                AdaptiveMassDeposit::LagrangianRedistribution::get_patch_info<ColumnMajorOrder>(num_procs,
                                                                                                p,
                                                                                                procs_dims_,
                                                                                                global_particle_dims_,
                                                                                                tmp_block_offset,
                                                                                                tmp_block_dims );
                
                AD_ASSERT_C( lid[0]>=tmp_block_offset[0] && lid[0]<(tmp_block_offset[0]+tmp_block_dims[0]),
                            { AvVec3i(lid[0],lid[1],lid[2]).print(); tmp_block_offset.print(); tmp_block_dims.print();} );
                assert( lid[1]>=tmp_block_offset[1] && lid[1]<(tmp_block_offset[1]+tmp_block_dims[1]) );
                assert( lid[2]>=tmp_block_offset[2] && lid[2]<(tmp_block_offset[2]+tmp_block_dims[2]) );
            }
            
            
        }
        
        
        return true;
        
    }
    
private:
    AvVec3i procs_dims_;
    AvVec3i global_particle_dims_;
    std::vector< std::shared_ptr<particles_with_ids_chunk> > particles_per_proc_;
    
};



class AdPointDataReadersFactory
{
public:
    virtual std::shared_ptr<AdPointDataReader> create_reader() = 0;
    
};


template< class ReaderClass >
class AdPointDataReadersCreator : public AdPointDataReadersFactory
{
public:
    virtual std::shared_ptr<AdPointDataReader> create_reader()
    {
        return std::shared_ptr<ReaderClass> ( new ReaderClass() );
    }
    
};

class AdDarkSkyDataReader : public AdPointDataReader
{
    
protected:
    
    static inline void morton_2_grid( const int64_t morton, int64_t grid_id[3])
    {
        
        static const int64_t mask = (1LL<<48)-1;
        
        int64_t key = morton & mask;
        
        int level = 0;
        
        grid_id[0] = 0;
        grid_id[1] = 0;
        grid_id[2] = 0;
        
        while ( key>0 )
        {
            grid_id[2] += (key & 1) << level;
            key = key >> 1;
            
            grid_id[1] += (key & 1) << level;
            key = key >> 1;
            
            grid_id[0] += (key & 1) << level;
            key = key >> 1;
            
            level += 1;
        }
        
        std::swap( grid_id[0], grid_id[2] );
        
    }
    
    
    
    static inline uint64_t grid_2_morton( const unsigned int x, const unsigned int y, const unsigned int z)
    {
        uint64_t morton = 0;
        
        static const uint64_t mb = (sizeof(uint64_t)*CHAR_BIT)/3;
        
        for ( uint64_t i=0; i<mb; ++i )
        {
            const uint64_t i2 = 2*i;
            const uint64_t is = uint64_t(1) << i;
            morton |= ( (x&is)<<i2) | ( (y&is)<<(i2+1)) | ( (z&is)<<(i2+2)) ;
        }
        
        return morton;
        
    }

    
    
    static size_t get_header_size( const std::string& filename )
    {
        
        std::filebuf fb;
        if ( !fb.open (filename.c_str(),std::ios::in|std::ios_base::binary) )
        {
            throw AdRuntimeException( "ERROR: get_header_size(): failed to open file: " + filename );
        }
        
        std::istream infile(&fb);
        
        char buffer[1024];
        size_t security_c = 0;
        int sha1_chunks = -1;
        
        bool found_eoh = false;
        
        while ( infile.good() && security_c++<100000 )
        {
            infile.getline (buffer, 1024 );
            std::string line(buffer);
            if ( line.compare(0, 17, "int sha1_chunks =")==0  )
            {
                std::string tmp = line.substr(line.find("=")+1, line.find(";")-line.find("=")-1);
                sha1_chunks = atoi(tmp.c_str());
            }
            
            if ( line.compare(0, 9, "# SDF-EOH") == 0 )
            {
                found_eoh = true;
                break;
            }
        }
        /*
         struct {
         unsigned int sha1_len;
         unsigned char sha1[20];
         }[16];
         */
        // header size == current file pos + offet for sha1
        const size_t header_size = size_t(infile.tellg()) + sha1_chunks*(sizeof(int)+20);
        
        fb.close();
        
        if ( found_eoh==false || sha1_chunks<0 || security_c++>=100000 )
        {
            throw AdRuntimeException("Failed to parse header of file: " + filename );
        }
        
        return header_size;
        
    }
    
    
    virtual size_t load_data_(const std::string& filename,
                              const int my_rank,
                              const int num_procs,
                              const int stride,
                              AvVec3i& procs_dims,
                              AvVec3i& global_particle_dims,
                              std::vector< std::shared_ptr<particles_with_ids_chunk> >& particles_per_rank )
    {
        
#if 1
        
        // first get header size of SDF file
        unsigned long header_size = 0;
        
        if ( my_rank== 0 )
        {
            header_size = get_header_size(filename);
        }
        
        CHECK_MPI_ERROR( MPI_Bcast( &header_size, 1, MPI_LONG_INT, 0, MPI_COMM_WORLD) );
        
        if ( header_size==0 )
        {
            throw AdRuntimeException( "ERROR: AdDarkSkyDataReader::load_data_(): invalid header size" );
        }
        
        
        MPI_File file_handle;
        
        CHECK_MPI_ERROR( MPI_File_open( MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &file_handle) );
        
        //size_t file_size = get_file_size(filename);
        MPI_Offset file_size = 0;
        CHECK_MPI_ERROR( MPI_File_get_size( file_handle, &file_size ) );
        
        /*
         struct darksky_particle
         {
         float x, y, z;
         float vx, vy, vz;
         int64_t id;
         } ds_part_buffer;
         */
        static const size_t bytes_per_particle = 6*sizeof(float)+sizeof(int64_t);
        static const int ids_offset = 6*sizeof(float);
        
        // hack: assuming that number of particle structs is divisible by number of processors
        const uint64_t total_num_particles = (file_size-header_size)/(bytes_per_particle);
        
        if ( (file_size-header_size)%bytes_per_particle!=0 )
        {
            throw AdRuntimeException("ERROR: AdDarkSkyDataReader::load_data_(): inconsistent file size ");
        }
        
        const size_t linear_particle_dims = rint( cbrt(total_num_particles) );
        
        if ( linear_particle_dims*linear_particle_dims*linear_particle_dims!= total_num_particles )
        {
            throw AdRuntimeException("ERROR: AdDarkSkyDataReader::load_data_(): lagrangian grid not cubical ");
        }
        else
        {
            std::cout << "INFO: AdDarkSkyDataReader::load_data_(): linear lagrangian grid dimension: " << linear_particle_dims << std::endl;
        }
        
        const size_t particles_per_proc = ceil( float(total_num_particles)/num_procs);
        
        const unsigned int src_dims[3] = {
            static_cast<unsigned int>(linear_particle_dims/stride),
            static_cast<unsigned int>(linear_particle_dims/stride),
            static_cast<unsigned int>(linear_particle_dims/stride)
        };
        unsigned  int dst_dims[3];
        
        ProcessLayout::find_best_match( num_procs, src_dims, dst_dims);
        
        procs_dims = AvVec3i( dst_dims[0],dst_dims[1],dst_dims[2] );
        
        const AvVec3i process_3D_idx = ColumnMajorOrder::map_linear_to_3D_idx( uint64_t(my_rank), procs_dims );;
        
        int effective_rank = -1;
        
        const bool power_of_two = (procs_dims[0]>0) && ( !( procs_dims[0] & (procs_dims[0]-1)) );
        if ( power_of_two && procs_dims[1] == procs_dims[0] && procs_dims[2]==procs_dims[0] )
        {
            for ( int64_t rank=0; rank<num_procs; ++rank )
            {
                int64_t lid[3] = {0,0,0};
                morton_2_grid(int64_t( rank ),lid); AD_ASSERT( grid_2_morton(lid[0],lid[1],lid[2])==uint64_t(rank), "" );
                
                if ( AvVec3i(lid[0],lid[1],lid[2])==process_3D_idx )
                {
                    effective_rank = rank;
                    break;
                }
            }
        }
        else
        {
            effective_rank = my_rank;
        }
        
        if ( effective_rank<0 || effective_rank>=num_procs )
        {
            throw AdRuntimeException("ERROR: AdDarkSkyDataReader::load_data_(): inconsistent effective process id. ");
        }
        
        
        global_particle_dims = AvVec3i(linear_particle_dims,linear_particle_dims,linear_particle_dims);
        
        if ( global_particle_dims[0]%stride!=0 || global_particle_dims[1]%stride!=0 || global_particle_dims[2]%stride!=0 )
        {
            throw AdRuntimeException("ERROR: AdDarkSkyDataReader::load_data_(): invalid stride for data dimensions ");
        }
        
        global_particle_dims[0] /= stride;
        global_particle_dims[1] /= stride;
        global_particle_dims[2] /= stride;
        
        AD_VERBOSE( 1, { std::cout << "INFO: AdDarkSkyDataReader::load_data_(): (MPI_rank==" << my_rank << "): Starting to load " << particles_per_proc << " of the total " << total_num_particles << " particles. " << std::endl; } );
        
        // compute number of cells per processor (upper rows could contain fewer particles)
        const AvVec3i cells_per_proc((global_particle_dims[0]-1 + (procs_dims[0]-1))/procs_dims[0],
                                     (global_particle_dims[1]-1 + (procs_dims[1]-1))/procs_dims[1],
                                     (global_particle_dims[2]-1 + (procs_dims[2]-1))/procs_dims[2] );
        
        
        const size_t start = effective_rank*particles_per_proc;
        //const size_t start = my_rank*particles_per_proc;
        
        
        // skip to start of our piece of the cake
        CHECK_MPI_ERROR( MPI_File_set_view( file_handle, header_size+start*bytes_per_particle, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL ) );
        
        static const size_t particles_per_pass = 50000000;
        static const size_t buffer_size = particles_per_pass*( bytes_per_particle );
        
        std::vector<char> buffer(buffer_size,0);
        
        size_t c = start;
        size_t loc_c = 0;
        bool file_good = true;
        
        size_t num_effective_particles = 0;
        
        while ( file_good && loc_c<particles_per_proc && c<total_num_particles )
        {
            
            const size_t new_size = std::min(buffer.size()/bytes_per_particle, std::min( particles_per_proc-loc_c,size_t(total_num_particles-c)) );
            buffer.resize( new_size*bytes_per_particle );
            
            if ( buffer.empty() )
            {
                std::cout << "INFO: AdDarkSkyDataReader::load_data_(): BREAK. " << std::endl;
                break;
            }
            
            MPI_Status status;
            
            if ( MPI_File_read( file_handle, &buffer[0], buffer.size(), MPI_CHAR, &status ) != MPI_SUCCESS )
            {
                std::cout << "WARNING: AdDarkSkyDataReader::load_data_(): file_handle = " << file_handle << " buffer.size()=" << buffer.size() << std::endl;
                throw AdRuntimeException("ERROR: AdDarkSkyDataReader::load_data_(): Failed to read all particles for lagrangian input grid.");
            }
            
            AD_ASSERT( buffer.size()%bytes_per_particle==0, "ERROR: AdDarkSkyDataReader::load_data_(): invalid number of read bytes" );
            
            const size_t num_read_particles = buffer.size()/bytes_per_particle;
            
            
            
#pragma omp parallel for reduction(+:loc_c) reduction(+:c) //schedule(dynamic)
            for ( size_t i=0; i<num_read_particles; ++i )
            {
                
                const size_t bytes_counter = i*bytes_per_particle;
                
                const float* float_ptr = reinterpret_cast<const float*>(&buffer.front()+bytes_counter);
                const float pos[3] = { float_ptr[0],float_ptr[1], float_ptr[2] };
                const int64_t id = *(reinterpret_cast<int64_t*>( &buffer.front()+bytes_counter+ids_offset ));
                
                int64_t lid[3];
                morton_2_grid(id, lid);
                
                AD_ASSERT( lid[0]>=0                 || lid[1]>=0                || lid[2]>=0       , "ERROR: AdDarkSkyDataReader::load_data_(): invalid dark sky id"  );
                AD_ASSERT( lid[0]<200000             || lid[1]<200000            || lid[2]<200000   , "ERROR: AdDarkSkyDataReader::load_data_(): invalid dark sky id"  );
                
                const bool skip_particle = ( stride!=1 && (lid[0]%stride!=0 || lid[1]%stride!=0 || lid[2]%stride!=0) );
                
                if ( !skip_particle )
                {
                    // compute effective particles ID
                    lid[0] /= stride;
                    lid[1] /= stride;
                    lid[2] /= stride;
                    
                    
                    particle_with_id new_part;
                    //const pos_t pos[3] = { pos[0],pos[1],pos[2]);
                    const ids_t lin_id = lid[0]+(global_particle_dims[0])*(lid[1]+lid[2]*(global_particle_dims[1]));
                    
                    
                    int target_ranks[8];
                    const int num_results = LagrangianRedistribution::get_target_processes<ColumnMajorOrder>(AvVec3i(lid[0],lid[1],lid[2]),
                                                                                                             procs_dims, global_particle_dims ,
                                                                                                             AvVec3i(cells_per_proc), target_ranks );
                    // to-do: get rid of critial section using private data and reduction ....
#pragma omp critical
                    {
                        for ( int n=0; n<num_results; ++n )
                        {
                            
#ifndef NDEBUG
                            {
                                
                                const AvVec3i idx_3D = ColumnMajorOrder::map_linear_to_3D_idx( lin_id, global_particle_dims );
                                
                                assert( idx_3D==AvVec3i(lid[0],lid[1],lid[2]) );
                                
                                AvVec3i tmp_block_offset;
                                AvVec3i tmp_block_dims;
                                // check if point is really assigned to the correct process
                                AdaptiveMassDeposit::LagrangianRedistribution::get_patch_info<ColumnMajorOrder>(num_procs,
                                                                                                                target_ranks[n],
                                                                                                                procs_dims,
                                                                                                                global_particle_dims,
                                                                                                                tmp_block_offset,
                                                                                                                tmp_block_dims );
                                
                                AD_ASSERT_C( lid[0]>=tmp_block_offset[0] && lid[0]<(tmp_block_offset[0]+tmp_block_dims[0]),
                                            { AvVec3i(lid[0],lid[1],lid[2]).print(); tmp_block_offset.print(); tmp_block_dims.print();} );
                                assert( lid[1]>=tmp_block_offset[1] && lid[1]<(tmp_block_offset[1]+tmp_block_dims[1]) );
                                assert( lid[2]>=tmp_block_offset[2] && lid[2]<(tmp_block_offset[2]+tmp_block_dims[2]) );
                            }
#endif
                            
                            particles_per_rank[target_ranks[n]]->push_back( pos[0],pos[1],pos[2], lin_id );
                        }
                        
                        
                        ++num_effective_particles;
                    }
                    
                } // end if (!skip_particle)
                
                // save using reduction pragma
                ++loc_c;
                ++c;
                
                
                
            } // end for i<num_read_particles
            
            AD_VERBOSE( 1, {std::cout << "INFO: AdDarkSkyDataReader::load_data_(): (MPI_rank==" << my_rank << "): Read " << 100.*loc_c/float(particles_per_proc) << "% of its particles." << std::endl;} );
            
        } // end while
        
        
        
        CHECK_MPI_ERROR( MPI_File_close( &file_handle ) );
        
        
        AD_VERBOSE( 1, { std::cout << "INFO: AdDarkSkyDataReader::load_data_(): (MPI_rank==" << my_rank << ") Finished reading " << loc_c << " particles." << std::endl; } );
 
        {
            long long int global_num_particles = 0;
            long long int local_num_particles = num_effective_particles;
            
            CHECK_MPI_ERROR( MPI_Allreduce(&local_num_particles, &global_num_particles, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD) );
            
            
            if( global_num_particles != size_t(global_particle_dims[0])*size_t(global_particle_dims[1])*size_t(global_particle_dims[2]) )
            {
                std::cout << global_num_particles << std::endl;
                std::cout << size_t(global_particle_dims[0])*size_t(global_particle_dims[1])*size_t(global_particle_dims[2]) << std::endl;
                global_particle_dims.print();
                throw AdRuntimeException("ERROR: AdDarkSkyDataReader::load_data_(): inconsistent global particle number");
            }
        }

        
        
        
        
        return num_effective_particles;
        
#endif
        
    }
    
    
    
    
    

};


class AdSortedByIdsPointDataReader : public AdPointDataReader
{
    
protected:
    virtual size_t load_data_(const std::string& filename,
                              const int my_rank,
                              const int num_procs,
                              const int stride,
                              AvVec3i& procs_dims,
                              AvVec3i& global_particle_dims,
                              std::vector< std::shared_ptr<particles_with_ids_chunk> >& particles_per_proc)
    {
        
        assert( particles_per_proc.size() == num_procs );
        
        if ( stride>1 )
        {
            std::cout << "WARNING: AdSortedByIdsPointDataReader::load__data(): 'stride' parameter currently not supported - will ignore it ... " << std::endl;
        }
        
        MPI_File file_handle = 0;
        
        if ( MPI_File_open( MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &file_handle )!=MPI_SUCCESS )
        {
            throw AdRuntimeException("ERROR: AdInitialization_IDS::load_particle_data(): failed to open file: " + filename );
        }
        
        MPI_Offset file_size = 0;
        
        CHECK_MPI_ERROR( MPI_File_get_size( file_handle, &file_size ) );
        
        static const size_t bytes_per_particle = 3*sizeof(float);
        
        if ( file_size%bytes_per_particle!=0 )
        {
            throw AdRuntimeException("ERROR: AdInitialization_IDS::load_particle_data(): inconsistent file size ");
        }
        
        const uint64_t global_num_particles = file_size/bytes_per_particle;
        
        global_particle_dims = AvVec3i( rint(cbrt(global_num_particles)) );
        
        if ( size_t(global_particle_dims[0])*size_t(global_particle_dims[1])*size_t(global_particle_dims[2])!= global_num_particles )
        {
            throw AdRuntimeException("ERROR: AdInitialization_IDS::load_particle_data(): lagrangian grid not cubical ");
        }
        
        {
            const unsigned int src_dims[3] = {
                static_cast<unsigned int>(global_particle_dims[0]),
                static_cast<unsigned int>(global_particle_dims[1]),
                static_cast<unsigned int>(global_particle_dims[2])
            };
            
            unsigned  int dst_dims[3];
            
            ProcessLayout::find_best_match( num_procs, src_dims, dst_dims);
            
            procs_dims = AvVec3i( dst_dims[0], dst_dims[1], dst_dims[2] );
        }
        
        AvVec3i block_offset;
        AvVec3i particle_block_dims;
        
        LagrangianRedistribution::get_patch_info< ColumnMajorOrder >( num_procs,
                                                                       my_rank,
                                                                       procs_dims,
                                                                       global_particle_dims,
                                                                       block_offset,
                                                                       particle_block_dims );
        
        
        
        
        CHECK_MPI_ERROR( MPI_File_set_view( file_handle, 0, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL ) );
        
        std::vector<float> buffer( 3*particle_block_dims[0] );
        
        size_t c = 0;
        
        
        
        for ( size_t k=0; k<particle_block_dims[2]; ++k )
        {
            
            for ( size_t j=0; j<particle_block_dims[1]; ++j )
            {
                
                const size_t linear_offset = block_offset[0] + global_particle_dims[0]*( (block_offset[1]+j) + global_particle_dims[1]*(block_offset[2]+k) ) ;
                
                // are we still inside the grid of active particles ?
                AD_ASSERT( linear_offset<global_num_particles, "" );
                AD_ASSERT_C( (linear_offset+particle_block_dims[0])<=global_num_particles, { printf("ERROR: %lu, %i, %llu\n",linear_offset,particle_block_dims[0], (long long unsigned int)global_num_particles);} );
                AD_ASSERT( bytes_per_particle*(linear_offset+particle_block_dims[0])<=size_t(file_size), "bug");
                
                MPI_Status status;
                
                CHECK_MPI_ERROR( MPI_File_read_at( file_handle, 3*linear_offset, &buffer[0], buffer.size(), MPI_FLOAT, &status ) );
                
                {
                    int count = 0;
                    CHECK_MPI_ERROR( MPI_Get_count( &status, MPI_FLOAT, &count ));
                    AD_ASSERT_C( size_t(count)==buffer.size() && size_t(count)==3*particle_block_dims[0], { printf("ERROR: file I/O: invalid number of read items(%i) vs particle_lock_dims[0](%i). index=[%lu,%lu], offset==%lu\n",count,particle_block_dims[0],j,k, linear_offset);} );
                }
                
                for ( size_t i=0; i<particle_block_dims[0]; ++i )
                {
                    
                    const size_t idx = 3*i;
                    AD_ASSERT(  (idx+2) < buffer.size(), "bug" );
                    
                    
                    const pos_t pos[3] = { buffer[idx], buffer[idx+1], buffer[idx+2] };
                    const ids_t id = linear_offset+i;
                    
                    assert( (block_offset[0]+i) < global_particle_dims[0] );
                    assert( (block_offset[1]+j) < global_particle_dims[1] );
                    assert( (block_offset[2]+k) < global_particle_dims[2] );
                    
                    // per construction each particle read by this rank is for this rank ...
                    particles_per_proc[my_rank]->push_back( pos[0],pos[1],pos[2],id );
                    
                    ++c;
                }
                
            }
            
        }
        
        
        AD_ASSERT_C( particles_per_proc[my_rank]->get_num() == (particle_block_dims[0]*particle_block_dims[1]*particle_block_dims[2]),
                    { printf("%lu, %i, %i, %i\n", particles_per_proc[my_rank]->get_num(), particle_block_dims[0], particle_block_dims[1], particle_block_dims[2]);});
        
        AD_VERBOSE(0, { std::cout << "INFO: AdInitialization_IDS::load_particle_data(): Process " << my_rank << " read " << c << " particles. Closing file." << std::endl; } );
        
        CHECK_MPI_ERROR( MPI_File_close( &file_handle ) );
        
        return c;
        
    }
    
};


static std::shared_ptr<AdPointDataReader> get_reader( const std::string& format )
{
    if ( format.find("SORTED_BY_ID")!=std::string::npos )
    {
        return std::shared_ptr<AdPointDataReader>( new AdSortedByIdsPointDataReader() );
    }
    else if ( format.find("DARK_SKY")!=std::string::npos )
    {
        return std::shared_ptr<AdPointDataReader>( new AdDarkSkyDataReader() );
    }
    else
    {
        throw AdRuntimeException("ERROR: id_type " + format + " not supported.");
    }
    
}
    
}


#endif
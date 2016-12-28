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



#ifndef _AMD_LAGRANGIAN_REDISTRIBUTION_
#define _AMD_LAGRANGIAN_REDISTRIBUTION_


namespace AdaptiveMassDeposit
{

class LagrangianRedistribution
{
    
public:
    
    template <class ID_MAPPING_ORDER>
    static int get_target_processes(const AvVec3i& particle_idx_3d, const AvVec3i& proc_dims,
                                    const AvVec3i& lagrangian_grid_dims, const AvVec3i& cells_per_proc,
                                    int* target_ranks )
    {
        assert( proc_dims[0]>0 && proc_dims[1]>0 && proc_dims[2]>0 );
        assert( cells_per_proc[0]>0 && cells_per_proc[1]>0 && cells_per_proc[2]>0 );
        assert( cells_per_proc[0]<lagrangian_grid_dims[0] && cells_per_proc[1]<lagrangian_grid_dims[1] && cells_per_proc[2]<lagrangian_grid_dims[2] );
        
        assert( particle_idx_3d[0]<lagrangian_grid_dims[0] && particle_idx_3d[1]<lagrangian_grid_dims[1] && particle_idx_3d[2]<lagrangian_grid_dims[2] );
        assert( particle_idx_3d[0]>=0 && particle_idx_3d[1]>=0 && particle_idx_3d[2]>=0 );
        
        
        int id[3][2] = { {-1,-1},{-1,-1},{-1,-1} };
        for ( int i=0; i<3; ++i )
        {
            int* tmp_id = id[i];
            
            // first check if this node is on a patch boundary
            if ( particle_idx_3d[i]%cells_per_proc[i]==0)
            {
                
                // check if it is on outer boundaries
                if ( particle_idx_3d[i]==0 )
                {
                    tmp_id[0] = 0;
                }
                else if ( particle_idx_3d[i]==(lagrangian_grid_dims[i]-1) )
                {
                    tmp_id[0] = proc_dims[i]-1;
                }
                else // this vertex belongs to several patches
                {
                    // this is what we call (interior) boundary particles in this routine
                    tmp_id[1] = particle_idx_3d[i]/cells_per_proc[i];
                    tmp_id[0] = tmp_id[1]-1;
                }
            }
            else // this node is inside one patch only
            {
                tmp_id[0] = particle_idx_3d[i]/cells_per_proc[i];
            }
            
            
        }
        
        
        int num_results = 0;
        for ( int k=0; k<=1; ++k )
        {
            if (id[2][k]==-1) continue;
            for ( int j=0; j<=1; ++j )
            {
                if (id[1][j]==-1) continue;
                for ( int i=0; i<=1; ++i )
                {
                    if (id[0][i]==-1) continue;
                    
                    assert( id[0][i]<proc_dims[0]   && id[1][j]<proc_dims[1]    && id[2][k]<proc_dims[2]    );
                    assert( id[0][i]>=0             && id[1][j]>=0              && id[2][k]>=0              );
                    
                    target_ranks[num_results] = ID_MAPPING_ORDER::map_3D_to_linear_idx( AvVec3i(id[0][i], id[1][j], id[2][k]), proc_dims );
                    ++num_results;
                    
                }
            }
        }
        
        return num_results;
        
    }
    
      
    
    
    template <class ID_MAPPING_ORDER>
    static void sort_by_id( const AvVec3i& patch_index_min, const AvVec3i& patch_dims, const AvVec3i& grid_dims, particles_with_ids_chunk& lagrangian_chunk )
    {
        
        assert( size_t(patch_dims[0])*size_t(patch_dims[1])*size_t(patch_dims[2])==lagrangian_chunk.get_num() );
        
        Particles::ids_t_vec& ids = *lagrangian_chunk.get_ids();
        
        // to-do: parallize this via OpenMP
        for ( size_t i=0; i<lagrangian_chunk.get_num(); ++i )
        {
            // first loop over all particles and replace ids by their absolute array positions
            const AvVec3i idx_3D = ID_MAPPING_ORDER::map_linear_to_3D_idx( ids[i], grid_dims );
            
            assert( idx_3D[0]>=patch_index_min[0] );
            assert( idx_3D[1]>=patch_index_min[1] );
            assert( idx_3D[2]>=patch_index_min[2] );
            
            AD_ASSERT_C( idx_3D[0]>=patch_index_min[0] && idx_3D[1]>=patch_index_min[1] && idx_3D[2]>=patch_index_min[2],
                        { idx_3D.print(); patch_index_min.print(); patch_dims.print(); } );
            
            AD_ASSERT_C( (idx_3D[0]-patch_index_min[0])<patch_dims[0] &&
                        (idx_3D[1]-patch_index_min[1])<patch_dims[1] &&
                        (idx_3D[2]-patch_index_min[2])<patch_dims[2],
                        { idx_3D.print(); patch_index_min.print(); patch_dims.print(); } );
            
            const ids_t idx_lin =ID_MAPPING_ORDER::map_3D_to_linear_idx( idx_3D-patch_index_min, patch_dims );
            
            AD_ASSERT_C( idx_lin<lagrangian_chunk.get_num(), { std::cout << " " << idx_lin << "   " << lagrangian_chunk.get_num()  << std::endl; } );
            
            ids[i] = idx_lin;
        }
        
        
        Particles::pos_t_vec& px = *lagrangian_chunk.get_positions(0);
        Particles::pos_t_vec& py = *lagrangian_chunk.get_positions(1);
        Particles::pos_t_vec& pz = *lagrangian_chunk.get_positions(2);
        
        
        bool need_decrease = false;
        for ( size_t i=0; i<lagrangian_chunk.get_num(); ++i )
        {
            // do we need to check the last particle again due to swapping ?
            if ( need_decrease )
            {
                assert(i>0);
                --i;
                assert(i<lagrangian_chunk.get_num());
            }
            
            // particle already at the correct position ?
            if ( ids[i]==i )
            {
                need_decrease = false;
            }
            else
            {
                need_decrease = true;
                // else we need to swap the particle to its correct positions
                std::swap( px[i], px[ids[i]] );
                std::swap( py[i], py[ids[i]] );
                std::swap( pz[i], pz[ids[i]] );
                
                std::swap( ids[i], ids[ids[i]] );
            }
            
            
        }
        
    }
    
    
    template < class INDEX_MAPPING_ORDER >
    static void get_patch_info(const int num_ranks,
                               const int my_rank,
                               const AvVec3i procs_dims,
                               const AvVec3i particle_dims,
                               AvVec3i& block_offset,
                               AvVec3i& block_dims )
    {
        if ( size_t(procs_dims[0])*size_t(procs_dims[1])*size_t(procs_dims[2])!=size_t(num_ranks) )
        {
            throw AdRuntimeException("ERROR: get_patch_info(): invalid process layout. ");
        }
        
        const AvVec3i offset_3D = INDEX_MAPPING_ORDER::map_linear_to_3D_idx( uint64_t(my_rank), procs_dims );
        
        
        AD_ASSERT( offset_3D[0]<procs_dims[0] && offset_3D[1]<procs_dims[1] && offset_3D[2]<procs_dims[2], "bug" );
        
        const AdVec3D<size_t> intertior_particle_block_dims(1+(particle_dims[0]-1 + (procs_dims[0]-1))/procs_dims[0],
                                                                      1+(particle_dims[1]-1 + (procs_dims[1]-1))/procs_dims[1],
                                                                      1+(particle_dims[2]-1 + (procs_dims[2]-1))/procs_dims[2] );
        
        block_offset = AvVec3i(std::min( size_t(particle_dims[0]-1), offset_3D[0]*(intertior_particle_block_dims[0]-1) ),
                               std::min( size_t(particle_dims[1]-1), offset_3D[1]*(intertior_particle_block_dims[1]-1) ),
                               std::min( size_t(particle_dims[2]-1), offset_3D[2]*(intertior_particle_block_dims[2]-1) ) );
        
        AD_ASSERT((size_t(particle_dims[0])>offset_3D[0]*intertior_particle_block_dims[0])||
                  (size_t(particle_dims[1])>offset_3D[1]*intertior_particle_block_dims[1])||
                  (size_t(particle_dims[2])>offset_3D[2]*intertior_particle_block_dims[2]), "get_patch_info(): internal error 01" );
        
        // particles per dimesions for this processes' block: might be smaller than intertior_particle_block_dims at the edges ...
        block_dims = AvVec3i( std::min( intertior_particle_block_dims[0], size_t(particle_dims[0]-block_offset[0]) ),
                             std::min( intertior_particle_block_dims[1], size_t(particle_dims[1]-block_offset[1]) ),
                             std::min( intertior_particle_block_dims[2], size_t(particle_dims[2]-block_offset[2]) ) );
        
        if ( block_offset[0]==(particle_dims[0]-1) ) block_dims[0]=0;
        if ( block_offset[1]==(particle_dims[1]-1) ) block_dims[1]=0;
        if ( block_offset[2]==(particle_dims[2]-1) ) block_dims[2]=0;
        
        
        
        assert( (block_offset[0]+block_dims[0]) <= particle_dims[0] );
        assert( (block_offset[1]+block_dims[1]) <= particle_dims[1] );
        assert( (block_offset[2]+block_dims[2]) <= particle_dims[2] );
        
    }
    
    
    // re-destributed the particles based on their lagrangian position on original grid
    // the initial chunk will be replaced by the lagrangian chunk
    template <class ID_MAPPING_ORDER>
    static void redistribute_particles(const int num_ranks,
                                       const int my_rank,
                                       const AvVec3i& initial_grid_dims,
                                       AvVec3i& patch_dims,
                                       std::vector< std::shared_ptr<particles_with_ids_chunk> >& particles_per_proc,
                                       Particles& output_particles )
    {
        
        AD_VERBOSE( 3, { std::cout << "INFO: redistribute_particles(my_rank==" <<  my_rank << "): entered routine. " << std::endl; } );
        
        output_particles.clear();
        
        // to-do: this is redundant computation that should be done once and stored as private member variables .
        //        (right it is computed also in reader routine  )
        AvVec3i proc_dims;
        {
            const unsigned int src_dims[3] = {
                static_cast<unsigned int>(initial_grid_dims[0]),
                static_cast<unsigned int>(initial_grid_dims[1]),
                static_cast<unsigned int>(initial_grid_dims[2])
            };
            unsigned  int dst_dims[3];
            
            ProcessLayout::find_best_match( num_ranks, src_dims, dst_dims);
            
            proc_dims = AvVec3i( dst_dims[0],dst_dims[1],dst_dims[2] );
        }
        
        const uint64_t num_procs = uint64_t(proc_dims[0])*uint64_t(proc_dims[1])*uint64_t(proc_dims[2]);
        
        
        if ( particles_per_proc.size() != num_procs )
        {
            throw AdRuntimeException("ERROR: redistribute_particles(): invalid 'particles_for_procs' size ");
        }
        
        
        
        
        
        {
#ifndef NDEBUG
            
            for ( int p=0; p<particles_per_proc.size(); ++p )
            {
                for ( int i=0; i<particles_per_proc[p]->get_num(); ++i )
                {
                    
                    const size_t lin_id = particles_per_proc[p]->get_id(i);
                    const AvVec3i lid = ID_MAPPING_ORDER::map_linear_to_3D_idx( lin_id, initial_grid_dims );
                    
                    
                    AvVec3i tmp_block_offset;
                    AvVec3i tmp_block_dims;
                    // check if point is really assigned to the correct process
                    AdaptiveMassDeposit::LagrangianRedistribution::get_patch_info<ID_MAPPING_ORDER>(num_procs,
                                                                                                    p,
                                                                                                    proc_dims,
                                                                                                    initial_grid_dims,
                                                                                                    tmp_block_offset,
                                                                                                    tmp_block_dims );
                    
                    AD_ASSERT_C( lid[0]>=tmp_block_offset[0] && lid[0]<(tmp_block_offset[0]+tmp_block_dims[0]),
                                { AvVec3i(lid[0],lid[1],lid[2]).print(); tmp_block_offset.print(); tmp_block_dims.print();} );
                    assert( lid[1]>=tmp_block_offset[1] && lid[1]<(tmp_block_offset[1]+tmp_block_dims[1]) );
                    assert( lid[2]>=tmp_block_offset[2] && lid[2]<(tmp_block_offset[2]+tmp_block_dims[2]) );
                }
            }
#endif
            
        }
        
        
        particles_with_ids_chunk& lagrangian_chunk = *particles_per_proc[my_rank];
        
        
        {
            size_t num_for_remote = 0;
            for ( size_t i=0; i<particles_per_proc.size(); ++i )
            {
                num_for_remote  += i==my_rank ? 0 : particles_per_proc[i]->get_num();
            }
            std::cout << "INFO: redistribute_particles(): remote_particles.size() == " << num_for_remote;
            std::cout << "; lagrangian_chunk.size()==" << lagrangian_chunk.get_num() << std::endl;
        }
        
        
        AD_ASSERT( my_rank<int(num_procs), "parameter error" );
        
        long long int total_num_sent = 0;
        long long int total_num_received = 0;
        long long int total_num_owned = 0;
        
        MeasureTime timer_gather_particles;
        
        CHECK_MPI_ERROR( MPI_Barrier(MPI_COMM_WORLD) );
        
        AD_VERBOSE( 3, { std::cout << "INFO: redistribute_particles(my_rank==" <<  my_rank << "): entered exchange loop. " << std::endl; } );
        
        
        MeasureTime send_recv_timer;
        
        FindCommunicationPartner finder( my_rank, num_procs );
        
        bool done = false;
        while ( true )
        {
            
            const int p = finder.get_next_communication_partner(done);
            
            if ( done )
            {
                break;
            }
            else if ( p<0 )
            {
                // we have no communication partner for this round
                AD_VERBOSE( 3, { std::cout << "INFO: redistribute_particles(my_rank==" <<  my_rank << "): found no communication partner. continuing while loop. " << std::endl; } );
                continue;
            }
            
            // local particles are already processed
            assert( p!=my_rank );
            
            const size_t num_sent = particles_per_proc[p]->get_num();
            
            send_recv_timer.start();
            
            // communicate the position chunks via MPI and get rid of obsolete data as soon as possible
            {
                
                
                Particles::pos_t_ptr src_px = particles_per_proc[p]->get_positions(0);
                Particles::pos_t_ptr src_py = particles_per_proc[p]->get_positions(1);
                Particles::pos_t_ptr src_pz = particles_per_proc[p]->get_positions(2);
                Particles::ids_t_ptr src_ids = particles_per_proc[p]->get_ids();
                
                Particles::pos_t_ptr dst_px = lagrangian_chunk.get_positions(0);
                Particles::pos_t_ptr dst_py = lagrangian_chunk.get_positions(1);
                Particles::pos_t_ptr dst_pz = lagrangian_chunk.get_positions(2);
                Particles::ids_t_ptr dst_ids = lagrangian_chunk.get_ids();
                
                assert( src_px->size() == num_sent );
                assert( src_py->size() == num_sent );
                assert( src_pz->size() == num_sent );
                assert( src_ids->size() == num_sent );
                
                const size_t num_recv_1 = send_receive_vector_mpi<pos_t>( p, p, *src_px, *dst_px );
                src_px.reset();
                
                const size_t num_recv_2 = send_receive_vector_mpi<pos_t>( p, p, *src_py, *dst_py );
                src_py.reset();
                
                const size_t num_recv_3 = send_receive_vector_mpi<pos_t>( p, p, *src_pz, *dst_pz );
                src_pz.reset();
                
                const size_t num_recv_4 = send_receive_vector_mpi<ids_t>( p, p, *src_ids, *dst_ids );
                src_ids.reset();
                
                particles_per_proc[p].reset();
                
                if ( num_recv_1!=num_recv_2 || num_recv_1!=num_recv_3 ||  num_recv_1!=num_recv_4 )
                {
                    throw AdRuntimeException( "ERROR: redistribute_particles(): error during send_receive_vector_mpi data transfer." );
                }
                
                total_num_received += num_recv_1;
                total_num_sent += num_sent;
                
                AD_VERBOSE( 10, { std::cout << "INFO: redistribute_particles(rank=" << my_rank << "): time to 'send_receive_vector_mpi("<<p<<")': " << send_recv_timer.get_total_time() << std::endl; } );
                AD_VERBOSE( 10, { std::cout << "INFO: redistribute_particles(my_rank==" <<  my_rank << "): exchanged data with process " << p << ". num_sent == " << num_sent << ", num_recv == " <<  num_recv_1 << std::endl; } );
                
            }
            
            send_recv_timer.add_measured_time();
            
        }
        
        
        AD_VERBOSE( 2, { std::cout << "INFO: redistribute_particles(my_rank==" <<  my_rank << "): finished exchange loop. " << std::endl; } );
        
        // get dims range for this proc
        AvVec3i particle_index_min;
        //AvVec3i patch_dims;
        get_patch_info< ID_MAPPING_ORDER >( num_procs,
                                           my_rank,
                                           proc_dims,
                                           initial_grid_dims,
                                           particle_index_min,
                                           patch_dims );
        
        // sort chunk of particles by id
        
        AD_VERBOSE( 3, { std::cout << "INFO: redistribute_particles(my_rank==" <<  my_rank << "): sorting all particles. " << std::endl; } );
        
        MeasureTime sorting_timer;
        
        sorting_timer.start();
        
        AD_ASSERT_C( size_t(patch_dims[0])*size_t(patch_dims[1])*size_t(patch_dims[2])==lagrangian_chunk.get_num(),
                    { fprintf( stderr, "%i, %i, %i, %i \n", patch_dims[0],patch_dims[1],patch_dims[2], int(lagrangian_chunk.get_num()) ); } );
        
        sort_by_id<ID_MAPPING_ORDER>( particle_index_min, patch_dims, initial_grid_dims, lagrangian_chunk );
        
        
        sorting_timer.add_measured_time();
        
        AD_VERBOSE( 3, { std::cout << "INFO: redistribute_particles(rank=" << my_rank << "): time for std::sort(): " << sorting_timer.get_total_time() << std::endl; } );
        AD_VERBOSE( 3, { std::cout << "INFO: redistribute_particles(my_rank==" <<  my_rank << "): sorted all particles. " << std::endl; } );
        
        
        
        
        // what follows is pure debugging code
#ifndef NDEBUG
        {
            
            AD_ASSERT_C( size_t(patch_dims[0])*size_t(patch_dims[1])*size_t(patch_dims[2])==lagrangian_chunk.get_num(),
                        { fprintf( stderr, "%i, %i, %i, %i \n", patch_dims[0],patch_dims[1],patch_dims[2], int(lagrangian_chunk.get_num()) ); } );
            
            // loop over all received particles
            for ( size_t i=0; i<lagrangian_chunk.get_num(); ++i )
            {
                // check if no duplicated ids
                if ( i>0 )
                {
                    if ( lagrangian_chunk.get_id(i-1)>=lagrangian_chunk.get_id(i) )
                    {
                        std::cout << lagrangian_chunk.get_id(i-1) << "   " << lagrangian_chunk.get_id(i) << std::endl;
                        std::cout.flush();
                        exit(0);
                    }
                }
                
                // get 3D id
                const AvVec3i part_idx_3d = particle_index_min + ID_MAPPING_ORDER::map_linear_to_3D_idx( lagrangian_chunk.get_id(i), patch_dims );
                
                if (! ( part_idx_3d[0]>=particle_index_min[0] && part_idx_3d[0]<=(particle_index_min[0]+patch_dims[0]) ))
                {
                    initial_grid_dims.print();
                    std::cout << lagrangian_chunk.get_id(i) << std::endl;
                    std::cout << part_idx_3d[0] << std::endl;
                    std::cout << particle_index_min[0] << std::endl;
                    std::cout << patch_dims[0] << std::endl << std::endl;
                    std::cout.flush();
                    AD_ASSERT(0,"too bad");
                }
            }
            
            std::cout << "INFO: redistribute_particles(my_rank==" << my_rank << "): passed consistency tests for local lagrangian particle chunk. " << std::endl;
            
        }
#endif
        // set the sorted chunk of lagrangian particles as result
        output_particles.set_positions( lagrangian_chunk.get_positions(0), lagrangian_chunk.get_positions(1), lagrangian_chunk.get_positions(2) );
        
        // now get rid of all obsolete data
        particles_per_proc.clear();
        
        
#ifndef NDEBUG
        long long int global_num_received = 0;
        long long int global_num_sent = 0;
        long long int global_num_owned = 0;
        
        
        CHECK_MPI_ERROR( MPI_Reduce(&total_num_received, &global_num_received, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD));
        CHECK_MPI_ERROR( MPI_Reduce(&total_num_sent, &global_num_sent, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD));
        CHECK_MPI_ERROR( MPI_Reduce(&total_num_owned, &global_num_owned, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD));
        
        
        if ( my_rank==0 )
        {
            AD_VERBOSE( 2, { std::cout << "INFO: redistribute_particles(): received ="     << std::setw(15) << global_num_received; } );
            AD_VERBOSE( 2, { std::cout << ", send = "        << std::setw(15) << global_num_sent; } );
            AD_VERBOSE( 2, { std::cout << ", owned = "       << std::setw(15) << global_num_owned; } );
            AD_VERBOSE( 2, { std::cout << " of the particles for this lagrangian patch. " << std::endl; } );
        }
        
#endif
        
        
        CHECK_MPI_ERROR( MPI_Barrier(MPI_COMM_WORLD) );
        
        
        AD_VERBOSE( 2, { std::cout << "INFO: redistribute_particles(my_rank==" <<  my_rank << "): finished routine. " << std::endl; } );
        
        
    } // end lagrangian distribution
    
    
 
    
}; // end class AMDLagrangianRedistribution


}; // end of namespace


#endif

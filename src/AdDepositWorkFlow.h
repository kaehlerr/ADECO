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


#ifndef _AD_DEPOSIT_WORKFLOW_
#define _AD_DEPOSIT_WORKFLOW_

#include <string>
#include "AdUtils.h"
#include "AdAssert.h"
#include "AdOctreeGeneration.h"
#include "AdDynamicLoadBalancing.h"

#include "AdCreateOctreeWriter.h"


namespace AdaptiveMassDeposit
{

    
    template <class ARRAY_ORDER>
    static void deposit_workflow( const AMD_Parameters& parameters, const int my_rank, const int num_procs )
    {
     
        try
        {
            
            
            if ( my_rank>=num_procs || num_procs<=0 || my_rank<0 )
            {
                throw AdRuntimeException( "ERROR: deposit_workflow(): invalid MPI rank information.");
            }
            
            // create reader object
            std::shared_ptr<AdPointDataReader> reader = get_reader( parameters.id_type );
            std::shared_ptr<AdOctreeGridWriter> writer = get_writer( "dummy" );
            
            // load data using abstract interface
            MeasureMPI_WC_Time loading_timer(true);
            {
                if ( my_rank==0)
                {
                    AD_VERBOSE( 0, { std::cout << "INFO: deposit_workflow(): STARTING TO LOAD DATA ............ " << std::endl; } );
                }
              
                const size_t num_local_particles = reader->load_data( parameters.input_file, my_rank, num_procs, parameters.stride );
                if ( num_local_particles==0 )
                {
                    std::cout << "WARNING: deposit_workflow(): rank: " + number2str(my_rank) + " does not own any particles. "  << std::endl;
                }
            
                loading_timer.measureGlobalMaxElapsed();
                
                if ( my_rank==0)
                {
                    AD_VERBOSE( 0, { std::cout << "INFO: deposit_workflow(): ............ FINISHED LOADING DATA. time == " << loading_timer.getResult() << std::endl; } );
                }
                
            }
            
            // compute global axis-aligned bounding box of computational domain
            const AABBox global_bbox = get_global_bbox_mpi( reader->get_particles_per_rank() );
                
            // apply boundary corrections
            if ( parameters.periodic_boundaries )
            {
                ARRAY_ORDER idc;
                for ( size_t p=0; p<num_procs; ++p )
                {
                    apply_boundary_corrections( global_bbox, reader->get_global_particle_dims(), idc, *(reader->get_particles_per_rank()[p]) );
                }
            }
            
            // by default the entire domain is considered our region of interest
            const AABBox region_of_interest = parameters.use_regions_of_interest ? parameters.region_of_interest : global_bbox;
            
            // and print some general information
            if ( my_rank==0 )
            {
                AD_VERBOSE( 1, {  std::cout << "INFO: deposit_workflow(): input filename      = " << parameters.input_file << std::endl; } );
                AD_VERBOSE( 1, {  std::cout << "INFO: deposit_workflow(): output path         = " << parameters.output_directory << std::endl; } );
                AD_VERBOSE( 1, {  std::cout << "INFO: deposit_workflow(): global_bbox         = "; global_bbox.min.print(); global_bbox.max.print(); } );
                AD_VERBOSE( 1, {  std::cout << "INFO: deposit_workflow(): particle_dims       = "; reader->get_global_particle_dims().print(); } );
                AD_VERBOSE( 1, {  std::cout << "INFO: deposit_workflow(): region_of_interest  = "; region_of_interest.min.print(); region_of_interest.max.print(); } );
            }
            
            // now each processor exchanges particles to generate complete patches in lagrangian space
            Tetrahedra<ARRAY_ORDER> tets;
            MeasureMPI_WC_Time lagrangian_distribution_timer(true);
            {
                if ( my_rank==0)
                {
                    AD_VERBOSE( 0, { std::cout << "INFO: deposit_workflow(): STARTING PARTICLE EXCHANGE ............ " << std::endl; } );
                }
            
                AvVec3i patch_dims;
                LagrangianRedistribution::redistribute_particles<ARRAY_ORDER>(num_procs,
                                                                              my_rank,
                                                                              reader->get_global_particle_dims(),
                                                                              patch_dims,
                                                                              reader->get_particles_per_rank(), tets.particles );
                tets.set_patch_dims( patch_dims[0], patch_dims[1], patch_dims[2] );
                
                lagrangian_distribution_timer.measureGlobalMaxElapsed();
                
                if ( my_rank==0)
                {
                    AD_VERBOSE( 0, { std::cout << "INFO: deposit_workflow(): ............ FINISHED PARTICLE EXCHANGE. time == " << lagrangian_distribution_timer.getResult() << std::endl; } );
                }
            }
          
            
            
            // construct the octree data structure and the auxiliary tet mesh
            TetOctree octree;
            MetaMesh<ARRAY_ORDER> tets_mesh( tets );
            
            MeasureMPI_WC_Time octree_construction_timer(true);
            {
                if ( my_rank==0)
                {
                    std::cout << "INFO: deposit_workflow(): STARTING OCTREE CONSTRUCTION ............ " << std::endl;
                }
                
                const long long local_num_tets = tets.get_num_tets();
                unsigned long long global_num_tets = 0;
                CHECK_MPI_ERROR( MPI_Allreduce(&local_num_tets, &global_num_tets, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD) );
                AD_VERBOSE( 3, { std::cout << "INFO: deposit_mass(rank==" << std::setw(5) << my_rank << "): num_local_tets == " << std::setw(12) << local_num_tets << " num_global_tets == " << global_num_tets << std::endl;} );
                
                const uint64_t max_tets_per_node = std::min( size_t(global_num_tets/num_procs) , size_t(parameters.max_tets_per_octree_node) );
                AdOctreeGeneration<ARRAY_ORDER>::generate_tree (num_procs,
                                                                 my_rank,
                                                                 global_bbox,
                                                                 region_of_interest,
                                                                 max_tets_per_node,
                                                                 parameters.max_refinement_level,
                                                                 tets_mesh,
                                                                 octree );
                
                octree_construction_timer.measureGlobalMaxElapsed();
                
                if ( my_rank==0)
                {
                    std::cout << "INFO: deposit_mass(): ............ FINISHED OCTREE CONSTRUCTION. time == " << octree_construction_timer.getResult() << std::endl;
                    
                    std::cout << "INFO: deposit_mass(): STARTING TO STORE OCTREE META DATA ............  " << std::endl;
                    writer->write_meta_data( parameters.output_directory, "dm_octree.a5", parameters.linear_patch_resolution, octree );
                    std::cout << "INFO: deposit_workflow(): ............ FINISHED STORING OCTREE META DATA. " << std::endl;
                }
            
            }
           
            
            // and finally enter the actual deposit loop
            MeasureMPI_WC_Time deposit_loop_timer(true);
            {
                if ( my_rank==0)
                {
                    std::cout << "INFO: deposit_workflow(): STARTING MASS DEPOSIT LOOP ............ " << std::endl;
                }
                
                deposit_leaf_nodes(num_procs,
                                   my_rank,
                                   parameters.num_pthreads,
                                   parameters.num_mpi_ranks_per_node,
                                   octree,
                                   tets_mesh,
                                   parameters.linear_patch_resolution,
                                   parameters.output_directory,
                                   writer);
                
                deposit_loop_timer.measureGlobalMaxElapsed();
                
                if ( my_rank==0)
                {
                    std::cout << "INFO: deposit_workflow(): ............ FINISHED MASS DEPOSIT LOOP. time == " << deposit_loop_timer.getResult() << std::endl;
                }
            }
            
            
            // and finally print all timing information once more
            if ( my_rank==0 )
            {
                std::cout << "INFO: RESULTS: time for loading data          == " << loading_timer.getResult() << std::endl;
                std::cout << "INFO: RESULTS: time for constructing octree   == " << octree_construction_timer.getResult() << std::endl;
                std::cout << "INFO: RESULTS: time for particle exchange     == " << lagrangian_distribution_timer.getResult() << std::endl;
                std::cout << "INFO: RESULTS: time for mass deposit          == " << deposit_loop_timer.getResult() << std::endl;
            }
            
 
            
        }
        catch ( std::exception& ex )
        {
            std::cout << "ERROR: deposit_workflow(): Caught exception: " << std::endl <<  ex.what() << std::endl;
        }
    
    }

};

#endif
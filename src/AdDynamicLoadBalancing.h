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


#ifndef _AD_DYNAMIC_LOAD_BALANCING_
#define _AD_DYNAMIC_LOAD_BALANCING_

#include <unistd.h>
#include <stdint.h>
#include <vector>
#include <iostream>
#include <list>
#include <limits>


#include "AdThreadInterface.h"
#include "AdTypeDefinitions.h"
#include "AdDeposit2Grid.h"
#include "AdOctree.h"
#include "AdOctreeGridWriter.h"


namespace AdaptiveMassDeposit
{
    
    typedef float deposit_grid_type;
    
    typedef std::vector<float> particle_array;
    
    
    struct TetDepositInfo
    {
        const particle_array* particles;
        const std::vector<size_t>* connectivity;
        const pos_t* bbox;
        int dims;
        std::vector<deposit_grid_type>* data;
    };
    
    
    
    class AdWorkerThread : public AdThreadInterface
    {
        
    public:
        
        AdWorkerThread(const size_t id,
                       const unsigned int number_of_cells_per_patch_dimension,
                       pthread_mutex_t& lock ) :
        AdThreadInterface(id),
        number_of_cells_per_patch_dimension_(number_of_cells_per_patch_dimension),
        lock_(lock),
        round_finished_(true),
        all_finished_(false),
        patch_offset_(0),
        num_patches_2_process_(0)
        {
        
            if ( number_of_cells_per_patch_dimension<1 )
            {
                AdRuntimeException( "ERROR: AdWorkerThread(): invalid number_of_cells_per_patch_dimension()" );
            }
        
        }
        
        
    public:
        
        
        inline bool round_is_finished()
        {
            return round_finished_;
        }
        
        
        inline void set_all_finished( const bool state )
        {
            pthread_mutex_lock(&lock_);
            all_finished_ = state;
            pthread_mutex_unlock(&lock_);
        }
        
        
        inline void set_next_work_item(const uint64_t offset,
                                       const uint64_t num,
                                       const particle_array* particles,
                                       const std::vector<size_t>* connectivity,
                                       const pos_t* bbox,
                                       const int dims,
                                       std::vector<deposit_grid_type>* data )
        
        {
            
            assert( connectivity->empty() );
            
            if ( !round_is_finished() )
            {
                throw AdRuntimeException("ERROR: AdWorkerThread::set_next_work_item(): not ready for new work yet ...");
            }
            
            if (! (particles!=0 && data!=0 && data->size()==(size_t(dims)*size_t(dims)*size_t(dims)) && connectivity->size()%4==0) )
            {
                throw AdRuntimeException("ERROR: AdWorkerThread::set_next_work_item(): invalid function arguments ...");
            }
            
            if ( num==0 || dims<=0 )
            {
                throw AdRuntimeException("ERROR: AdWorkerThread::set_next_work_item(): no tets to process ...");
            }
            
            // we are working with trivial connectivity at the moment ...
            if ( (3*(number_of_cells_per_patch_dimension_+1)*(number_of_cells_per_patch_dimension_+1)*(number_of_cells_per_patch_dimension_+1)*(offset+num-1))>=particles->size() )
            {
                //std::cout << offset << " ... " << num << " ... " << connectivity->size()/4 << std::endl;
                throw AdRuntimeException("ERROR: AdWorkerThread::set_next_work_item(): invalid 'offset+num' value for given number of patches ...");
            }
            
            pthread_mutex_lock(&lock_);
            
            work_item_.particles = particles;
            work_item_.connectivity = connectivity;
            work_item_.bbox =  bbox;
            work_item_.dims = dims;
            work_item_.data = data;
            patch_offset_ = offset;
            num_patches_2_process_ = num;
            round_finished_ = false;
          
            pthread_mutex_unlock(&lock_);
            
        }
        
        
        
    private:
        
        inline void set_round_finished_( const bool state )
        {
            pthread_mutex_lock(&lock_);
            round_finished_ = state;
            pthread_mutex_unlock(&lock_);
        }
        
        virtual void run_( )
        {
            size_t num_rounds = 0;
            
            while ( !all_finished_ )
            {
                // check if there is new work for us
                if ( !round_is_finished() )
                {
                    // if the round_is_finished flag is set, work_item must be initialized !
                    assert( work_item_.particles!=0 && work_item_.connectivity->size()%4==0 );
                    assert( work_item_.data->empty()==false );
#if 1
                    
                    //const size_t* connectivity_ptr = work_item_.connectivity->empty() ? 0 : &(work_item_.connectivity->front());
                    const float* positions_ptr = work_item_.particles->empty() ? 0 : &(work_item_.particles->front());
                    
                    DepositWrapper::deposit_tets(true, id_, patch_offset_, num_patches_2_process_,
                                                 positions_ptr, work_item_.particles->size(),
                                                 //connectivity_ptr,  work_item_.connectivity->size(),
                                                 number_of_cells_per_patch_dimension_,
                                                 work_item_.bbox, work_item_.dims,
                                                 &(work_item_.data->front()) );
                    
                    
#else
                    // dummy code to fake some work - useful for debugging ...
                    for ( size_t i=0; i<work_item_.data->size(); ++i )
                    {
                        (*work_item_.data)[i] += num_tets_2_process_;
                    }
#endif
                    
                    num_rounds++;
                    // indicate we are finished
                    AD_VERBOSE( 10, { std::cout << "INFO: AdWorkerThread::run_(): thread[ " << id_ << "] finished round : patch offset = " << patch_offset_ << ", number of patches = " << num_patches_2_process_ << std::endl; } );
                    
                    // and 'thread-safely' indicate that we are done for this round
                    set_round_finished_( true );
                }
            }
            
            AD_VERBOSE( 3, { std::cout << "INFO: AdWorkerThread::run_(): thread[ " << id_ << "] all finished in " << num_rounds << "  rounds." << std::endl; } );
            
            assert( round_is_finished()==true );
            
        }
        
        
    private:
        const unsigned int number_of_cells_per_patch_dimension_;
        
        pthread_mutex_t& lock_;
        
        volatile TetDepositInfo work_item_;
        
        volatile bool round_finished_;
        volatile bool all_finished_;
        volatile uint64_t patch_offset_;
        
        volatile uint64_t num_patches_2_process_;
        
        
    };
    
    
    class AdWorkSchedulerThread : public AdThreadInterface
    {
        
    public:
        AdWorkSchedulerThread(const size_t          id,
                              const unsigned int    num_threads,
                              const unsigned int    thread_id_offset,
                              const unsigned int    number_of_cells_per_patch_dimension,
                              float&                busy_time,
                              pthread_mutex_t&      lock ) :
        AdThreadInterface(id),
        number_of_cells_per_patch_dimension_(number_of_cells_per_patch_dimension),
        round_finished_(true),
        all_finished_(false),
        busy_time_(busy_time),
        lock_(lock)
        {
            
            if ( number_of_cells_per_patch_dimension_<1 )
            {
                throw AdRuntimeException( "ERROR: AdWorkSchedulerThread(): invalid 'number_of_cells_per_patch_dimension' value. ");
            }
            
            // an start the worker threads
            work_item_.particles = 0;
            work_item_.connectivity = 0;
            work_item_.bbox = 0;
            work_item_.dims = 0;
            work_item_.data = 0;
            
            worker_threads_.resize(num_threads);
            worker_deposit_grids_.resize(num_threads);
            
            for ( size_t i=0; i<worker_threads_.size(); ++i )
            {
                worker_threads_[i] = new AdWorkerThread( thread_id_offset+i, number_of_cells_per_patch_dimension_, lock_ );
                worker_threads_[i]->create();
            }
            
        }
        
        ~AdWorkSchedulerThread()
        {
            // signal all threads that we are done
            for ( size_t i=0; i<worker_threads_.size(); ++i )
            {
                worker_threads_[i]->set_all_finished(true);
                worker_threads_[i]->finish();
                delete worker_threads_[i];
                worker_threads_[i] = 0;
            }
            
            AD_VERBOSE( 5, { std::cout << "INFO: AdWorkSchedulerThread(): all threads finished - cleaned up. " << std::endl; } );
            
        }
        
        
    public:
        
        inline bool round_is_finished()
        {
            return round_finished_;
        }
        
        
        
        inline void set_all_finished( const bool state )
        {
            pthread_mutex_lock(&lock_);
            all_finished_ = state;
            pthread_mutex_unlock(&lock_);
        }
        
        
        inline void set_next_work_item( const particle_array* particles,
                                       const std::vector<size_t>* connectivity,
                                       const pos_t* bbox,
                                       const int dims,
                                       std::vector<deposit_grid_type>* data )
        
        {
            
            AD_VERBOSE( 10, { std::cout << "INFO: AdWorkSchedulerThread():set_next_work_item() called." << std::endl; } );
            
            if ( !round_is_finished() )
            {
                throw AdRuntimeException("ERROR: AdWorkSchedulerThread(): not ready for new work yet ...");
            }
            
            if ( particles==0 )
            {
                throw AdRuntimeException("ERROR: AdWorkSchedulerThread()::set_next_work_item(): particles==0 ");
            }
            
            if ( connectivity==0 )
            {
                throw AdRuntimeException("ERROR: AdWorkSchedulerThread()::set_next_work_item(): connectivity==0 ");
            }
            
            if ( connectivity->size()%4!=0 )
            {
                throw AdRuntimeException("ERROR: AdWorkSchedulerThread()::set_next_work_item(): connectivity->size()%4!=0 ");
            }
            
            if ( data==0 || bbox==0 )
            {
                throw AdRuntimeException("ERROR: AdWorkSchedulerThread()::set_next_work_item(): data==0 || bbox==0");
            }
            
            pthread_mutex_lock(&lock_);
            //tet_offset_ = offset;
            work_item_.particles = particles;
            work_item_.connectivity = connectivity;
            work_item_.bbox = bbox;
            work_item_.dims = dims;
            work_item_.data = data;
            round_finished_ = false;
            pthread_mutex_unlock(&lock_);
            
            // set_round_finished_( false );
            
        }
        
        
    private:
        
        
        inline void set_round_finished_( const bool state )
        {
            pthread_mutex_lock(&lock_);
            round_finished_ = state;
            pthread_mutex_unlock(&lock_);
        }
        
        
        virtual void run_( )
        {
            
            MeasureTime total_timer, busy_timer;
            
            total_timer.start();
            
            // calling main / IO thread is responsible to cleaning workitem.data, so we assume here that everything is set up correctly ...
            try
            {
                
                //std::vector< std::pair<size_t,size_t> > debugging_info(worker_threads_.size());
                //size_t c=0;
                while ( !all_finished_ )
                {
                    // check if there is new work for us
                    if ( !round_is_finished() )
                    {
                        
                        busy_timer.start();
                        
                        
                        AD_VERBOSE( 10, { std::cout << "INFO: run_(): round not finished " << std::endl; } );
                        
                        // if the round_is_finished flag is set, work_item must be initialized !
                        assert( work_item_.particles!=0 );
                        assert( work_item_.connectivity!=0 );
                        
                        // the code is broken for the connectivity case ...
                        assert( work_item_.connectivity->size()==0 );
                        
                        const size_t num_floats_per_patch = 3*(number_of_cells_per_patch_dimension_+1)*(number_of_cells_per_patch_dimension_+1)*(number_of_cells_per_patch_dimension_+1);
                        const size_t num_patches = work_item_.particles->size()/num_floats_per_patch;
                        
                        if ( work_item_.particles->size()%num_floats_per_patch!=0 )
                        {
                            throw AdRuntimeException( " run_(): invalid position size " );
                        }
                        
                        // clear and resize the data arrays for the local threads ( grid size can change for each round )
                        for ( size_t i=0; i<worker_threads_.size(); ++i )
                        {
                            worker_deposit_grids_[i].resize( size_t(work_item_.dims)*size_t(work_item_.dims)*size_t(work_item_.dims) );
                            std::fill( worker_deposit_grids_[i].begin(), worker_deposit_grids_[i].end(), 0. );
                        }
                        
                        // hack/to-do: factor 10 ok or too large ? ( we want something larger than 1 for adaptive load balancing in the loop below )
                        const size_t min_patches_per_thread = std::min( size_t(1000), (num_patches+worker_threads_.size()-1)/worker_threads_.size() );
                                                                
                        const uint64_t patches_per_thread = std::max( size_t(min_patches_per_thread),(num_patches+10*worker_threads_.size()-1) / (10*worker_threads_.size()) );
                        
                        assert( patches_per_thread>0 );
                        
                        // now assign work for this round to the worker threads
                        uint64_t num_processed_patches = 0;
                        
                        AD_VERBOSE( 10, { std::cout << "INFO: AdWorkSchedulerThread: waiting for round to finish: " << num_processed_patches << ", " << num_patches <<std::endl; } );
                        
                        while ( num_processed_patches<num_patches )
                        {
                            
                            //AV_VERBOSE( 10, { std::cout << "INFO: run_(" << id_ << "): num_processed_tets<num_tets: " << num_processed_tets << " --- vs. --- " << num_tets  << std::endl; } );
                            
                            // assign portion of work to next available thread
                            for ( size_t i=0; i<worker_threads_.size(); ++i )
                            {
                                if ( worker_threads_[i]->round_is_finished() )
                                {
                                    
                                    const uint64_t patches_for_this_round = std::min(patches_per_thread,num_patches-num_processed_patches );
                                    
                                    assert( num_patches>=num_processed_patches );
                                    //assert( (num_processed_tets+tets_for_this_round)<=(work_item_.connectivity->size()/4) );
                                    
                                    if ( patches_for_this_round>0 )
                                    {
                                        //debugging_info[i] = std::pair<size_t,size_t>(num_processed_tets,tets_for_this_round);
                                        worker_threads_[i]->set_next_work_item(num_processed_patches,
                                                                               patches_for_this_round,
                                                                               work_item_.particles,
                                                                               work_item_.connectivity,
                                                                               work_item_.bbox,
                                                                               work_item_.dims,
                                                                               &(worker_deposit_grids_[i]) );
                     
                                        num_processed_patches += patches_for_this_round;
                                    }
                                    else
                                    {
                                        std::cout << "WARNING: run_(" << id_ << "): patches_for_this_round<=0: num_processed_patches<num_patches: " << num_processed_patches << " --- vs. --- " << num_patches  << std::endl;
                                        // tets_for_this_round==0 only possible if this was the last round and all work items are distributed
                                        break;
                                    }
                                }
                                
                                if ( num_processed_patches>=num_patches )
                                {
                                    break;
                                }
                                
                            }
                            
                        } // end 'while ( num_processed_tets<num_tets )'
                        
                        
                        // wait for worker_threads to finish and reduce the individual worker_deposit_grids_ to work_item.grid
                        size_t finished_counter = 0;
                        std::vector<bool> finished_flags( worker_threads_.size(), false );
                        do
                        {
                            finished_counter = 0;
                            for ( size_t i=0; i<worker_threads_.size(); ++i )
                            {
                                if ( worker_threads_[i]->round_is_finished() )
                                {
                                    ++finished_counter;
                                    if (finished_flags[i]==false )
                                    {
                                        finished_flags[i]=true;
                                        // reducing to work_item here allows to overlap the reductions with the computations in the worker grids
                                        if ( !(work_item_.data!=0 &&
                                               work_item_.data->empty()==false &&
                                               worker_deposit_grids_[i].size()==work_item_.data->size() ) )
                                        {
                                            throw AdRuntimeException("ERROR:  AdWorkSchedulerThread::run_(): Internal error 1.");
                                        }
                                        
                                        for ( size_t a=0; a<work_item_.data->size(); ++a )
                                        {
                                            (*work_item_.data)[a] += worker_deposit_grids_[i][a];
                                        }
                                        
                                    }
                                }
                            }
                            
                            // hack/todo: make sure not to burn too many cycles ... (better use conditional variables instead ... )
                            usleep( 1000 );
                            
                        }
                        while ( finished_counter != worker_threads_.size() );
                        
                        // and 'thread-safely' indicate that all worker threads are done for this round
                        set_round_finished_( true );
                        
                        busy_timer.add_measured_time();
                       
                        
                        AD_VERBOSE( 10, { std::cout << "INFO: AdWorkSchedulerThread: current busy time == " << busy_timer.get_total_time() << std::endl; } );
                        
                        
                        AD_VERBOSE( 10, { std::cout << "INFO: AdWorkSchedulerThread: finished next round." << std::endl; } );
                        
                        
                    } // end 'if ( !round_is_finished() )'
                    
                } // 'while ( !all_finished_ )'
                
                AD_VERBOSE( 5, { std::cout << "INFO: AvWorkSchedulerThread: all finished." << std::endl; } );
                
                assert( round_is_finished()==true );
                
            }  // end try - block
            catch ( AdRuntimeException& ex)
            {
                std::cout << "ERROR: AdWorkSchedulerThread()::run_(): Caught exception: " << std::endl << ex.what() << std::endl;
                exit(0);
            }
         
            
            total_timer.add_measured_time();
            
            busy_time_ += busy_timer.get_total_time();
            
             AD_VERBOSE( 0, { std::cout << "INFO: AdWorkSchedulerThread: all finished. total time = " <<  total_timer.get_total_time() << ". busy time = " << busy_timer.get_total_time() << std::endl; } );
            
            
        }
        
        
    private:
        
        const unsigned int number_of_cells_per_patch_dimension_;
        
        volatile TetDepositInfo work_item_;
        
        
        std::vector<AdWorkerThread*> worker_threads_;
        std::vector< std::vector<deposit_grid_type> > worker_deposit_grids_;
        
        volatile bool round_finished_;
        volatile bool all_finished_;
        
        volatile float& busy_time_;
        
        pthread_mutex_t& lock_;
        
        
    };
    
    
    class AdDepositThreadManager
    {
        
    public:
        
        AdDepositThreadManager(const unsigned int num_threads,
                             const unsigned int thread_ids_offset,
                             const unsigned int number_of_cells_per_patch_dimension,
                             float& busy_time )
        {
            if ( pthread_mutex_init(&lock_, 0)!=0 )
            {
                throw AdRuntimeException("ERROR: ADDepositThreadManager(): Failed to initialize mutex. Exiting now ... \n");
            }
            
            work_thread_ = new AdWorkSchedulerThread( 0, num_threads, thread_ids_offset, number_of_cells_per_patch_dimension, busy_time, lock_ );
            work_thread_->create();
            
            AD_VERBOSE( 5, { std::cout << "INFO: ADDepositThreadManager(), in constructor - main thread: started AvWorkSchedulerThread ..." << std::endl; } );
        }
        
        
        ~AdDepositThreadManager()
        {
            try
            {
                delete_work_thread_();
                pthread_mutex_destroy(&lock_);
            }
            catch ( std::exception& ex )
            {
                std::cerr << "ERROR: ~ADDepositThreadManager(): Caught exception: " << std::endl << ex.what() << std::endl;
                exit(0);
            }
        }
        
        
        inline bool ready_for_new_work() const
        {
            return ( work_thread_!=0 && work_thread_->round_is_finished() );
        }
        
        inline void set_next_work_item(const particle_array* particles,
                                       const std::vector<size_t>* connectivity,
                                       const pos_t* bbox,
                                       const int dims,
                                       std::vector<deposit_grid_type>* data )
        {
            
            AD_VERBOSE( 5, { std::cout << "INFO: AdDepositThreadManager():set_next_work_item() called." << std::endl; } );
            
            if ( !ready_for_new_work() )
            {
                throw AdRuntimeException( "ERROR: AdDepositThreadManager(): not ready for new work yet ...");
            }
            
            
            if ( particles==0 )
            {
                throw AdRuntimeException("ERROR: AdDepositThreadManager()::set_next_work_item(): work_item_.particles==0 ");
            }
            
            if ( connectivity==0 )
            {
                throw AdRuntimeException("ERROR: AdDepositThreadManager()::set_next_work_item(): work_item_.connectivity==0 ");
            }
            
            if ( connectivity->size()%4!=0 )
            {
                throw AdRuntimeException("ERROR: AdDepositThreadManager()::set_next_work_item(): work_item_.connectivity->size()%4!=0 ");
            }
            
            assert( work_thread_!= 0 );
            
            work_thread_->set_next_work_item( particles, connectivity, bbox, dims, data );
            
        }
        
        
    private:
        
        inline void delete_work_thread_()
        {
            // signal all threads that we are done
            if ( work_thread_ )
            {
                work_thread_->set_all_finished(true);
                work_thread_->finish();
                delete work_thread_;
                work_thread_ = 0;
            }
            
        }
        
        pthread_mutex_t lock_;
        AdWorkSchedulerThread* work_thread_;
        
    };
    
    
    class PendingNodesInfo
    {
    public:
        typedef enum types { LOCAL=0, REMOTE=1 } TetType;
        node_id_t node_id;
        size_t num_iterations;
        size_t current_iteration;
        TetType tet_type;
    };
    
    
    static inline size_t get_list_of_leaf_nodes( const TetOctree& octree, std::list<node_id_t>& leaf_list )
    {
        std::vector<node_id_t> leaf_vec;
        octree.getLeafNodes( leaf_vec );
        std::copy( leaf_vec.begin(), leaf_vec.end(), std::back_inserter( leaf_list ) );
        assert( leaf_vec.size()==leaf_list.size());
        return leaf_vec.size();
    }
    
    
    static inline node_id_t find_best_node_and_remove_it_from_list( const int process_rank, const TetOctree& octree, std::list<node_id_t>& leaf_list )
    {
        
        std::list<node_id_t>::iterator best_node = leaf_list.begin();
        // meta-cells: this should be ok
        size_t best_tet_coverage = octree.getNode(*best_node).getNodeData().num_tets_on_proc_estimate[process_rank];
        
        for ( std::list<node_id_t>::iterator it = ++(leaf_list.begin()); it!=leaf_list.end(); ++it  )
        {
            // meta-cells: this should be ok
            const size_t tet_coverage = octree.getNode(*it).getNodeData().num_tets_on_proc_estimate[process_rank];
            if ( tet_coverage>best_tet_coverage )
            {
                best_tet_coverage = tet_coverage;
                best_node = it;
            }
        }
        
        const node_id_t best_node_id = (*best_node);
        // now remove best node from list
        leaf_list.erase(best_node);
        
        return best_node_id;
        
    }
    
    
    template <class T>
    static void gather_vectors_via_mpi( const int my_rank, const int num_ranks, const int root_rank, const std::vector<T>& send_data, std::vector<T>& gathered_data  )
    {
        
        AD_VERBOSE( 10, { std::cout << "INFO: gather_vectors_via_mpi(): rank( " << my_rank << " )" << " entered routine " << std::endl; } );
        
        assert ( my_rank < num_ranks && root_rank<num_ranks );
        
        try
        {
            
            gathered_data.clear();
            
            std::vector<int> counts;
            std::vector<int> displacements;
            
            // counts and displacement arrays only required on the (receiving)root process
            counts.resize( num_ranks, 0 );
            displacements.resize( num_ranks, 0 );
            
            // first we have to gather the counts to the root process
            const unsigned long int local_size = sizeof(T)*send_data.size();
            
            CHECK_MPI_ERROR( MPI_Allgather( &local_size, 1, MPI_INT,  &(counts.front()), 1, MPI_INT, MPI_COMM_WORLD ) );
            
            // next calculate displacements and total size of receiving array on root process
            {
                size_t total_size=0;
                for( int i=0; i<num_ranks; ++i )
                {
                    total_size += counts[i];
                 
                    if ( counts[i]<0 || total_size>=size_t(std::numeric_limits<int>::max()) )
                    {
                        std::cerr << "counts[" << i << "] == " << counts[i] << ", total_size == " << total_size << std::endl;
                        throw AdRuntimeException("detected integer overflow");
                    }
         
                }
                
                displacements[0]=0;
                for( int i=1; i<num_ranks; ++i )
                {
                    displacements[i] = counts[i-1] + displacements[i-1];
                    if ( displacements[i]<0 )
                    {
                        std::cerr << "displacements["   << i    << "] == " << displacements[i];
                        std::cerr << "counts["          << i-1  << "] == " << counts[i-1];
                        std::cerr << "displacements["   << i-1  << "] == " << displacements[i-1] << std::endl;
                        throw AdRuntimeException("detected integer overflow in displacements");
                    }
                    
                }
                
                if ( total_size%sizeof(T)!=0 )
                {
                    std::cerr << "total_size = "    << total_size   << std::endl;
                    std::cerr << "sizeof(T) = "     << sizeof(T)    << std::endl;
                    throw AdRuntimeException("inconsistent total size");
                }
                
                gathered_data.resize( total_size/sizeof(T) );
            }
            
            CHECK_MPI_ERROR( MPI_Gatherv ((const void*)&(send_data.front()), sizeof(T)*send_data.size(), MPI_CHAR,
                                          (void*)&(gathered_data.front()),  &(counts.front()), &(displacements.front()),  MPI_CHAR,
                                          root_rank, MPI_COMM_WORLD ) );
            
            
        }
        catch ( AdException& ex )
        {
            std::cerr << "ERROR: gather_vectors_via_mpi(): rank( " << my_rank << " ): Caught exception: " << std::endl;
            std::cerr << ex.what() << std::endl;
            gathered_data.clear();
            exit(0);
        }
        
        
        AD_VERBOSE( 10, { std::cout << "INFO: gather_vectors_via_mpi(): rank( " << my_rank << " )" << " finished routine " << std::endl; } );
        
    }
    
    
    
    
    template <class id_functor>
    static void gather_remote_tets(const unsigned int my_rank, const unsigned int num_ranks, const unsigned int p,
                                   const int current_iteration, const int num_iterations,
                                   const MetaMesh<id_functor>& tets_mesh, const TetOctree::Node& current_node, std::vector<pos_t>& new_tets )
    {
  
        AD_VERBOSE( 10, { std::cout << "INFO: gather_remote_tets(): rank( " << my_rank << " )" << " entered routine " << std::endl; } );
        
        new_tets.clear();
        
        //particle_chunk send_chunk;
        std::vector<pos_t> send_chunk;
        
        
        if ( my_rank!=p )
        {
            /* 
             meta-cells:
             need to replace this with loop over all meta cells, collecting all (interleaved) tet coords into a vector
             meta_cells.get_tets( const AABBox& box, const uint64_t meta_cells_id, std::vector<pos_t>& interleaved_positions ) const
             
             then offset into that vector based on the iteration
             -- this includes some redundant work in the case of multiple iterations, but we have to live with it ...
             
             */
            
            
            std::vector<pos_t>  current_positions;
            const AABBox node_bbox(current_node.getLowerLeft(), current_node.getLowerLeft()+PosVec(current_node.getExtension()));
            
            for ( size_t m=0; m<current_node.getNodeData().local_meta_cell_ids.size(); ++m )
            {
                //const std::vector<uint64_t>& local_tet_ids = current_node.getNodeData().local_tet_ids;
                std::vector<pos_t> interleaved_positions;
                tets_mesh.get_tets( node_bbox, current_node.getNodeData().local_meta_cell_ids[m], interleaved_positions );
                std::copy( interleaved_positions.begin(), interleaved_positions.end(), back_inserter(current_positions) );
            }
            
            const size_t num_floats_per_meta_cell = 3*tets_mesh.get_number_of_vertices_per_meta_block();
            
            assert( current_positions.size()%(num_floats_per_meta_cell)==0 );
            
            
            const size_t num_meta_patches = current_positions.size()/num_floats_per_meta_cell;
            
            size_t num_patches_for_this_iteration = 0;
            size_t patch_offset = 0;
            
            get_offset_and_number( num_meta_patches, current_iteration, num_iterations, patch_offset, num_patches_for_this_iteration );
            
            std::copy(current_positions.begin()+num_floats_per_meta_cell*patch_offset,
                      current_positions.begin()+num_floats_per_meta_cell*(patch_offset+num_patches_for_this_iteration),
                      back_inserter(send_chunk) );
            
            assert( send_chunk.size()==num_floats_per_meta_cell*num_patches_for_this_iteration );

            
        }
        
        assert( MPI_Barrier(MPI_COMM_WORLD)==MPI_SUCCESS );
        
        gather_vectors_via_mpi( my_rank, num_ranks, p, send_chunk, new_tets );
        
        assert( MPI_Barrier(MPI_COMM_WORLD)==MPI_SUCCESS );
        
        assert ( new_tets.size()%4==0 );
        
        AD_VERBOSE( 10, { std::cout << "INFO: gather_remote_tets(): rank( " << my_rank << " )" << " exited routine " << std::endl; } );
        
    }
    

    
    
    // this is the first iteration, so we need to deposit out local tets
    template <class id_functor>
    static void gather_local_tets(const unsigned int my_rank,
                                  const int current_iteration,
                                  const int num_iterations,
                                  const MetaMesh<id_functor>& tets_mesh,
                                  const TetOctree::Node& current_node,
                                  particle_array& result_positions )
    {
        
        
        particle_array current_positions;
        
        /*
         meta-cells:
         need to replace this with loop over all meta cells, collecting all (interleaved) tet coords into a vector
         meta_cells.get_tets( const AABBox& box, const uint64_t meta_cells_id, std::vector<pos_t>& interleaved_positions ) const
         
         then offset into that vector based on the iteration
         -- this includes some redundant work in the case of multiple iterations, but we have to live with it ...
         
         */

        const AABBox node_bbox(current_node.getLowerLeft(), current_node.getLowerLeft()+PosVec(current_node.getExtension()));
        
        for ( size_t m=0; m<current_node.getNodeData().local_meta_cell_ids.size(); ++m )
        {
            //const std::vector<uint64_t>& local_tet_ids = current_node.getNodeData().local_tet_ids;
            std::vector<pos_t> interleaved_positions;
            tets_mesh.get_tets( node_bbox, current_node.getNodeData().local_meta_cell_ids[m], interleaved_positions );
            std::copy( interleaved_positions.begin(), interleaved_positions.end(), back_inserter(current_positions) );
        }
        
        
        const size_t num_floats_per_meta_cell = 3*tets_mesh.get_number_of_vertices_per_meta_block();
        
        assert( current_positions.size()%num_floats_per_meta_cell==0 );
        
        const size_t num_meta_patches = current_positions.size()/num_floats_per_meta_cell;
        
        
        size_t num_patches_for_this_iteration = 0;
        size_t patch_offset = 0;
        
        get_offset_and_number( num_meta_patches, current_iteration, num_iterations, patch_offset, num_patches_for_this_iteration );
        
        std::copy(current_positions.begin()+num_floats_per_meta_cell*patch_offset,
                  current_positions.begin()+num_floats_per_meta_cell*(patch_offset+num_patches_for_this_iteration),
                  back_inserter(result_positions) );
        
        if ( result_positions.size()!=(num_floats_per_meta_cell*num_patches_for_this_iteration) )
        {
            std::cerr << "ERROR: gather_local_tets(" << my_rank << "): iteration==" << current_iteration << "; num_floats_per_meta_cell==" << num_floats_per_meta_cell << "; num_patches ==" << num_patches_for_this_iteration << "; result_positions.size()==" << result_positions.size() << std::endl;
            throw AdRuntimeException("ERROR: fatal error in gather_local_tets.");
        }
        
        
    }
    
    
    static inline size_t get_number_of_local_tets( const unsigned int p, const TetOctree::Node& node )
    {
        assert( node.getNodeData().num_tets_on_proc_estimate.size()>p );
        return ( node.getNodeData().num_tets_on_proc_estimate[p] );
        
    }
    
    static inline size_t get_number_of_remote_tets( const unsigned int p, const TetOctree::Node& node )
    {
        assert( node.getNodeData().num_tets_on_proc_estimate.size()>p );
        assert( ( node.getNodeData().num_tets_covering_node_estimate -  node.getNodeData().num_tets_on_proc_estimate[p])>=0 );
        return ( node.getNodeData().num_tets_covering_node_estimate - node.getNodeData().num_tets_on_proc_estimate[p] );
    }
    
    // hack/to-do: optimize this
    template <typename T>
    static inline size_t get_sum( const std::vector<T>& vec )
    {
        size_t sum = 0;
        
        for ( size_t i=0; i<vec.size(); ++i )
        {
            sum += static_cast<size_t>(vec[i]);
        }
        
        return sum;
    }
    

    
    template<class id_functor>
    static void deposit_leaf_nodes(const unsigned int num_procs,
                                   const unsigned int my_rank,
                                   const unsigned int num_threads,
                                   const unsigned int num_mpi_ranks_per_node,
                                   const TetOctree& octree,
                                   const MetaMesh<id_functor>& tets,
                                   const unsigned int grid_dims,
                                   const std::string& output_path,
                                   std::shared_ptr<AdOctreeGridWriter> writer )
    {
        
        try
        {

            double total_mass = 0.;
            
       
            // to-do: this should be a user-defined parameter
            const int max_tets_per_iteration = 50000000;
            
            MeasureTime timer_total_time, timer_gather_remote_tets, timer_gather_local_tets, timer_mpi_allgather_calls, timer_fine_best_node_calls;
            
            timer_total_time.start();
            
            AD_VERBOSE( 5, { std::cout << "INFO: deposit_leaf_nodes(): rank( " << my_rank << " )" << ": entered function. " << std::endl; } );
            
            
            if ( num_procs==0 || my_rank>=num_procs )
            {
                throw AdRuntimeException( "ERROR: deposit_leaf_nodes(): Invalid MPI rank arguments." );
            }
            
            if ( num_threads==0 )
            {
                throw AdRuntimeException( "ERROR: deposit_leaf_nodes(): Invalid num_threads argument." );
            }
            
            // gather leaf nodes
            std::list<node_id_t> leaf_list;
            const size_t num_leaf_nodes = get_list_of_leaf_nodes( octree, leaf_list );
            
            
            // compute the thread id offset: requied to access the correct GPU device in the worker threads
            const unsigned int thread_ids_offset = num_threads*( my_rank%num_mpi_ranks_per_node );
            
            
            // variable to store time the threads actually did usefull work
            float local_busy_time = 0.f;

            // start the worker threads
            AdaptiveMassDeposit::AdDepositThreadManager* thread_manager =
                new AdaptiveMassDeposit::AdDepositThreadManager(num_threads,
                                                              thread_ids_offset,
                                                              tets.get_number_of_cells_per_dimension(),
                                                              local_busy_time );
            
            // allocate array to store information process - node - assignment
            std::vector< PendingNodesInfo > nodes_for_procs( num_procs );
            for ( size_t i=0; i<nodes_for_procs.size(); ++i )
            {
                nodes_for_procs[i].num_iterations = 0;
                nodes_for_procs[i].current_iteration = 0;
                nodes_for_procs[i].node_id = INVALID_NODE_ID;
                nodes_for_procs[i].tet_type = PendingNodesInfo::LOCAL;
            }
            
            // variable to store the current local work item
            particle_array current_positions;
            std::vector<ids_t> current_connectivity;
            
            pos_t bbox[6] = {0.,0.,0.,0.,0.,0.};
            std::vector<deposit_grid_type> grid_data( size_t(grid_dims)*size_t(grid_dims)*size_t(grid_dims), 0. );
            
            
            size_t num_processed_leaf_nodes = 0;
            
            assert( (num_processed_leaf_nodes+leaf_list.size())== num_leaf_nodes );
            
            int counter = 0;
            
            while ( num_processed_leaf_nodes < num_leaf_nodes )
            {
                
                // first communicate which processes are ready for new work
                const unsigned char local_ready_for_new_work = (unsigned char)(thread_manager->ready_for_new_work());
                
                std::vector<unsigned char> procs_ready_for_new_work( num_procs );
                
                timer_mpi_allgather_calls.start();
                CHECK_MPI_ERROR( MPI_Allgather( &local_ready_for_new_work, 1, MPI_UNSIGNED_CHAR, &(procs_ready_for_new_work.front()), 1, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD ) );
                timer_mpi_allgather_calls.add_measured_time();
                
                // now loop over the list of processes ready for work and pick the next one that is ready
                for ( size_t p=0; p<num_procs; ++p )
                {
                    
                    PendingNodesInfo& node_info = nodes_for_procs[p];
                    
                    if ( counter++==100000 && p==my_rank && (node_info.node_id!=INVALID_NODE_ID) )
                    {
                        counter = 0;
                        AD_VERBOSE( 10, { std::cout << "INFO: deposit_leaf_nodes(" << my_rank << "): Processing tets. " << std::endl; } );
                    }
                    
#ifndef NDEBUG
                    if ( p==my_rank )
                    {
                        if  (node_info.node_id != INVALID_NODE_ID )
                        {
                            if ( node_info.tet_type==PendingNodesInfo::LOCAL )
                            { AD_VERBOSE( 10, { std::cout << "INFO: deposit_leaf_nodes(" << my_rank << "): Processing local tets, iteration " << node_info.current_iteration << " of " << node_info.num_iterations << std::endl;} ); }
                            else
                            { AD_VERBOSE( 10, { std::cout << "INFO: deposit_leaf_nodes(" << my_rank << "): Processing remote tets, iteration " << node_info.current_iteration << " of " << node_info.num_iterations << std::endl;} ); }
                        }
                        else
                        {
                            AD_VERBOSE( 10, { std::cout << "INFO: deposit_leaf_nodes(" << my_rank << "): Not processing tets at the moment. ready for new work: " << bool(procs_ready_for_new_work[p]) << std::endl;} );
                        }
                    }
#endif
                    
                    
                    if ( bool(procs_ready_for_new_work[p])==true )
                    {
                        
                        // check if rank p is processsing a pending node ( one that needs multiple passes)
                        if ( node_info.node_id != INVALID_NODE_ID )
                        {
                            
                            // clear the data deposited during the last round
                            if ( p==my_rank )
                            {
                                current_positions.clear();
                                current_connectivity.clear();
                            }
                            
                            // check if node is finally processed and can be stored on disk
                            // when we are done with all iterations for the remote tets, we are done with everything
                            // (since the local ones are processed first )
                            // and finally store the data
                            assert( node_info.current_iteration<node_info.num_iterations );
                            
                            if ( node_info.tet_type==PendingNodesInfo::REMOTE )
                            {
                                if ( (node_info.current_iteration+1)==node_info.num_iterations )
                                {
                                    if ( p==my_rank )
                                    {
                                        // store patch data to disk
                                        total_mass += writer->write_patch( output_path, octree.getNode(node_info.node_id), octree, grid_dims, grid_data );
                                        std::fill( grid_data.begin(), grid_data.end(), 0. );
                                    }
                                    ++num_processed_leaf_nodes;
                                    
                                    // and reset the node info for this process
                                    node_info.node_id = INVALID_NODE_ID;
                                    node_info.num_iterations = 0;
                                    node_info.current_iteration = 0;
                                    node_info.tet_type = PendingNodesInfo::LOCAL;
                                }
                                else
                                {
                                    ++node_info.current_iteration;
                                    assert( node_info.current_iteration<node_info.num_iterations );
                                }
                            }
                            else
                            {
                                assert( node_info.tet_type==PendingNodesInfo::LOCAL );
                                assert( node_info.current_iteration<node_info.num_iterations );
                                
                                if ( (node_info.current_iteration+1)==node_info.num_iterations )
                                {
                                    // this was the last round for the local tets
                                    node_info.tet_type=PendingNodesInfo::REMOTE;
                                    node_info.num_iterations = ( get_number_of_remote_tets(p,octree.getNode(node_info.node_id)) + max_tets_per_iteration ) /  max_tets_per_iteration;
                                    node_info.current_iteration = 0;
                                }
                                else
                                {
                                    ++node_info.current_iteration;
                                }
                                
                                assert( node_info.current_iteration<node_info.num_iterations );
                            }
                            
                        } // end of: if ( node_info.node_id != INVALID_NODE_ID )
                        
                        
                        // if process p still has a node assigned at this point, by construction it is a pending node that needs multiple data transfer rounds
                        assert( node_info.node_id == INVALID_NODE_ID || node_info.current_iteration<node_info.num_iterations);
                        
                        // make sure to check if leaf nodes are still available: could be empty if last leaf is processed using multiple rounds ...
                        if ( node_info.node_id == INVALID_NODE_ID && !leaf_list.empty() )
                        {
                            assert(leaf_list.empty()==false);
                            
                            // find best node N for process p and remove N from leaf_ids
                            timer_fine_best_node_calls.start();
                            node_info.node_id = find_best_node_and_remove_it_from_list( p, octree, leaf_list );
                            timer_fine_best_node_calls.add_measured_time();
                            
                            assert( node_info.node_id!=INVALID_NODE_ID );
                            
                            // compute number of iterations and tets per round for the local nodes first
                            node_info.tet_type = PendingNodesInfo::LOCAL;
                            node_info.num_iterations = (get_number_of_local_tets( p, octree.getNode(node_info.node_id) ) + max_tets_per_iteration ) / max_tets_per_iteration; // ...............
                            node_info.current_iteration = 0;
                        }
                        
                        
                        if ( node_info.node_id == INVALID_NODE_ID )
                        {
                            // this can happen if we are just waiting for other proces to finish the last of their multiple rounds
                            continue;
                        }
                        
                        
                        
                        const TetOctree::Node& current_node = octree.getNode(node_info.node_id);
                        
                        // be careful touching this else if statement - nice way to generate MPI deadlocks ...
                        if (  node_info.tet_type==PendingNodesInfo::LOCAL )
                        {
                            // this is the first iteration, so we need to deposit out local tets
                            if ( p==my_rank )
                            {
                                timer_gather_local_tets.start();
                                gather_local_tets( my_rank, node_info.current_iteration, node_info.num_iterations, tets, current_node, current_positions );
                                timer_gather_local_tets.add_measured_time();
                            }
                        }
                        // be careful touching this else if statement - nice way to generate MPI deadlocks ...
                        else
                        {
                            assert( node_info.tet_type==PendingNodesInfo::REMOTE );
                            
                            assert( MPI_Barrier(MPI_COMM_WORLD)==MPI_SUCCESS );
                            
                            std::vector<pos_t> dummy_tets;
                            
                            std::vector<pos_t>& new_tets = p==my_rank ? current_positions : dummy_tets;
                            
                            timer_gather_remote_tets.start();
                            gather_remote_tets(  my_rank, num_procs,  p, node_info.current_iteration, node_info.num_iterations, tets, current_node, new_tets );
                            timer_gather_remote_tets.add_measured_time();
                            
                            assert( MPI_Barrier(MPI_COMM_WORLD)==MPI_SUCCESS );
                            
                        }
                        
                        if ( p==my_rank && current_positions.empty()==false )
                        {
                            assert( current_connectivity.empty() );
                            
                            bbox[0] = current_node.getLowerLeft()[0]; bbox[1] = current_node.getLowerLeft()[0]+current_node.getExtension();
                            bbox[2] = current_node.getLowerLeft()[1]; bbox[3] = current_node.getLowerLeft()[1]+current_node.getExtension();
                            bbox[4] = current_node.getLowerLeft()[2]; bbox[5] = current_node.getLowerLeft()[2]+current_node.getExtension();
                            
                            thread_manager->set_next_work_item(&current_positions,
                                                               &current_connectivity,
                                                               bbox,
                                                               grid_dims,
                                                               &grid_data );
                            
                            // assign new work to worker threads (non-blocking)
                            AD_VERBOSE( 5, { std::cout << "INFO: deposit_leaf_nodes( rank = " << my_rank << "): Assigned new work item to deposit threads. " << std::endl; } );
                            
                        }
                        
                        AD_VERBOSE( 10, { std::cout << "INFO: rank( " << my_rank << " ): " << num_processed_leaf_nodes << " .... " << num_leaf_nodes << std::endl; } );
                        
                    } // end: if ( bool(procs_ready_for_new_work[p])==true )
                    
                    // make sure to stop distributing work once
                    assert( num_processed_leaf_nodes <= num_leaf_nodes );
                    if ( num_processed_leaf_nodes >= num_leaf_nodes )
                    {
                        assert( leaf_list.empty() );
                        break;
                    }
                    
                } // end for loop: for ( size_t p=0; p<num_procs; ++p )
                
            } // end while loop: while ( num_processed_leaf_nodes < num_leaf_nodes )
            
            
            assert( thread_manager->ready_for_new_work() );
            assert( nodes_for_procs[my_rank].node_id == INVALID_NODE_ID );
            
            
            delete thread_manager;
            thread_manager = 0;
            
            AD_VERBOSE( 3, { std::cout << "INFO: deposit_leaf_nodes( rank = " << my_rank << "): all worker threads finished. " << std::endl; } );
            
            std::cout << "INFO: deposit_leaf_nodes( rank = " << my_rank << "): time for gathering remote tets =  " << timer_gather_remote_tets.get_total_time()     << std::endl;
            
            timer_total_time.add_measured_time();
            
            std::cout << "INFO: deposit_leaf_nodes( rank = " << my_rank << "): total time spend in routine    =  " << timer_total_time.get_total_time()             << std::endl;
            std::cout << "INFO: deposit_leaf_nodes( rank = " << my_rank << "): time for gathering local tets  =  " << timer_gather_local_tets.get_total_time()      << std::endl;
            std::cout << "INFO: deposit_leaf_nodes( rank = " << my_rank << "): time for MPI_Allgather calls   =  " << timer_mpi_allgather_calls.get_total_time()    << std::endl;
    
            
            CHECK_MPI_ERROR( MPI_Barrier(MPI_COMM_WORLD) );
            
            
            double local_elapsed = timer_total_time.get_total_time();
            double max_elapsed = 0., min_elapsed = 0.;
            
            CHECK_MPI_ERROR( MPI_Reduce( &local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD ));
            CHECK_MPI_ERROR( MPI_Reduce( &local_elapsed, &min_elapsed, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD ));
          
            float max_busy_time = 0., min_busy_time = 0.;
            CHECK_MPI_ERROR( MPI_Reduce( &local_busy_time, &min_busy_time, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD ));
            CHECK_MPI_ERROR( MPI_Reduce( &local_busy_time, &max_busy_time, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD ));
            
            
            double global_mass = 0.;
            
            CHECK_MPI_ERROR( MPI_Reduce( &total_mass, &global_mass, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD ));
            
            if ( my_rank == 0 )
            {
                std::cout << "INFO: RESULT: deposit_leaf_nodes: [min,max] time spend in routine        =  [ " << min_elapsed   << ", " << max_elapsed   << "] " << std::endl;
                std::cout << "INFO: RESULT: deposit_leaf_nodes: [min,max] busy time spend in threads   =  [ " << min_busy_time << ", " << max_busy_time << "] " << std::endl;
                std::cout << std::fixed;
                std::cout << "INFO: RESULT: deposit_leaf_nodes: total mass                  =  " << global_mass  << std::endl;
            }
            
            
        }
        catch ( std::exception& ex )
        {
            std::cerr << "ERROR: deposit_leaf_nodes( rank = " << my_rank << "): Caught exception: " << ex.what() << std::endl;
            exit(0);
        }

    }

    
    
};

#endif

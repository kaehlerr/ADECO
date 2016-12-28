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


#ifndef _ADM_AdaptiveMassDeposit_
#define _ADM_AdaptiveMassDeposit_


#include <mpi.h>
#include <limits>

#include "AdTypeDefinitions.h"


#define CHECK_MPI_ERROR( error_code )                                                   \
{                                                                                       \
    if ( (error_code) != MPI_SUCCESS  )                                                 \
    {                                                                                   \
        throw AdaptiveMassDeposit::AdRuntimeException(                                  \
            std::string("ERROR: Failed MPI call in file: \' ") +                        \
            std::string(__FILE__) + " \' , line " +                                     \
            AdaptiveMassDeposit::number2str(__LINE__) + " " +                           \
            AdaptiveMassDeposit::get_mpi_error_string( error_code ) + "\n", false       \
        );                                                                              \
    }                                                                                   \
}                                                                                       \


namespace AdaptiveMassDeposit
{
    
    
    static inline std::string get_mpi_error_string( const int error_code )
    {
        
        int result_len = 0;
        char error_string[MPI_MAX_ERROR_STRING] = "no error string";
        
        if ( MPI_Error_string( error_code, error_string, &result_len ) != MPI_SUCCESS )
        {
            return std::string("Failed to optain error string for error code: " + number2str<int>(error_code) );
        }
        
        return std::string("ERROR: get_mpi_error_string(): " + std::string(error_string));
        
    }
    
    
    /*
     received data will be appended to receive_vector
     */
    
    template <typename T>
    static size_t send_receive_vector_mpi( const int dst, const int src, const std::vector<T>& send_chunk, std::vector<T>& receive_vector )
    {
        
        // first communicate number of data items of type T to be exchanged between ranks 'dst' and 'src'
        unsigned long int send_num = send_chunk.size();
        unsigned long int recv_num = 0;
        
        CHECK_MPI_ERROR ( MPI_Sendrecv(&send_num,       1,  MPI_UNSIGNED_LONG,  dst,   0,
                                       &recv_num,       1,  MPI_UNSIGNED_LONG,  src,    0,
                                       MPI_COMM_WORLD,      MPI_STATUS_IGNORE ) );
        
        // and exchange the actual data
        if ( send_num>0 || recv_num>0 )
        {
            
            //const size_t send_size = send_chunk.size()*sizeof(T);
            //const size_t recv_size = receive_chunk.size()*sizeof(T);
        
            const size_t max_items = std::max( size_t(1), size_t(1E09)/sizeof(T) );
            
            const size_t num_iterations = std::max(ceil_div(send_num,max_items),
                                                   ceil_div(recv_num,max_items) );
            
            assert( num_iterations>0 );
            
            size_t num_reveiced_items = 0;
            
            AD_VERBOSE( 5, {  std::cout << "INFO: send_receive_vector_mpi( dst==" << dst << "): exchanging data (recv_num == " << recv_num << ", send_num == " << send_num << ") in: " << num_iterations << " iterations." << std::endl; } );
            
            for ( size_t i=0; i<num_iterations; ++i )
            {
            
                size_t num_send_items = 0, send_items_offset = 0;
                get_offset_and_number( send_num, i, num_iterations, send_items_offset, num_send_items );
                
                size_t num_recv_items = 0, recv_items_offset = 0;
                get_offset_and_number( recv_num, i, num_iterations, recv_items_offset, num_recv_items );
               
                std::vector<T> recv_chunk(num_recv_items);
                
                assert( (send_items_offset+num_send_items)<=send_chunk.size() );
                assert( num_recv_items<=recv_chunk.size() );
                
                
                char* recv_buffer       = recv_chunk.empty() ? 0 : reinterpret_cast<      char*>(&recv_chunk[0]);
                const char* send_buffer = send_chunk.empty() ? 0 : reinterpret_cast<const char*>(&send_chunk[send_items_offset]);
                
                const size_t send_bytes = num_send_items*sizeof(T);
                const size_t recv_bytes = num_recv_items*sizeof(T);
                
                if ( send_bytes>=std::numeric_limits<int>::max() || recv_bytes>=std::numeric_limits<int>::max() )
                {
                    throw AdRuntimeException( "ERROR: send_receive_vector_mpi(): invalid amount of send / receive bytes." );
                }
                
                CHECK_MPI_ERROR( MPI_Sendrecv(send_buffer,      int(send_bytes),   MPI_CHAR, dst, 0,
                                              recv_buffer,      int(recv_bytes),   MPI_CHAR, src, 0,
                                              MPI_COMM_WORLD,   MPI_STATUS_IGNORE ));
            
                num_reveiced_items += recv_chunk.size();
                
                // and append new data to receive_vector
                std::copy (recv_chunk.begin(), recv_chunk.end(), std::back_inserter(receive_vector) );
            
            }
       
            assert( num_reveiced_items == recv_num );
            
            
        }
        
        return recv_num;
  
    }

    
    
       
    
    static inline void initialize_mpi( int argc, char **argv, int& my_rank, int& num_procs )
    {
        //CHECK_MPI_ERROR( MPI_Init(&argc, &argv) );
        //
        int provided;
        CHECK_MPI_ERROR( MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) );

        if ( provided<MPI_THREAD_FUNNELED )
        { 
            std::cout << "ERROR: initialize_mpi(): thread-level support < MPI_THREAD_FUNNELED, exiting. " << std::endl;
            exit(1);
        }
        else 
        {
            AD_VERBOSE( 5, { std::cout << "INFO: initialize_mpi(): requested thread-level support: " << MPI_THREAD_FUNNELED << std::endl; } );
            AD_VERBOSE( 5, { std::cout << "INFO: initialize_mpi(): provided thread-level support: " << provided << std::endl; } );
        }
 
        CHECK_MPI_ERROR( MPI_Comm_size(MPI_COMM_WORLD, &num_procs) );
        CHECK_MPI_ERROR( MPI_Comm_rank(MPI_COMM_WORLD, &my_rank) );
        
        
        int version = -1;
        int subversion = -1;
        CHECK_MPI_ERROR( MPI_Get_version( &version, &subversion ) );
        
        AD_VERBOSE( 5, { std::cout << "INFO: initialize_mpi(): version = " << version << ", subversion = " << subversion << std::endl; } );
        
        
        char version_string[MPI_MAX_LIBRARY_VERSION_STRING]="";
        int resultlen = 0;
       
        CHECK_MPI_ERROR( MPI_Get_library_version(version_string, &resultlen ) );
        
        AD_VERBOSE( 5, { std::cout << "INFO: initialize_mpi(): MPI_Get_library_version() version string = " << std::string(version_string) << std::endl; } );
        
    }
    

    
    
    
    
    class MeasureMPI_WC_Time
    {
        
    public:
        
        MeasureMPI_WC_Time( const bool start_now, const MPI_Comm communicator = MPI_COMM_WORLD ) :
        comm_(communicator),
        local_start_time_(0.),
        result_(-1.)
        {
            if ( start_now)
            {
                start();
            }
        }
        
        inline void start()
        {
            CHECK_MPI_ERROR( MPI_Barrier(comm_) );
            local_start_time_ = MPI_Wtime();
        }
        
        inline double measureLocalElapsed( )
        {
            const double local_stop_time = MPI_Wtime();
            result_ = (local_stop_time - local_start_time_);
            return result_;
        }
        
        inline double measureGlobalMaxElapsed( const int master_rank = 0 )
        {
            int my_rank = -1;
            CHECK_MPI_ERROR( MPI_Comm_rank( comm_, &my_rank ) );
            
            double local_elapsed = measureLocalElapsed( );
            double max_elapsed = 0.;
            CHECK_MPI_ERROR( MPI_Reduce( &local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, master_rank, comm_ ));
            result_ = (my_rank==master_rank) ? max_elapsed : -1.;
            return result_;
        }
        
        inline bool isValid() const
        {
            return (result_>-1.E-07);
        }
        
        inline double getResult() const
        {
            if ( !isValid() )
            {
                std::cerr << "WARNING: MeasureMPI_WC_Time(): measured time negative. Did you call measure*Elapsed(...) ? " << std::endl;
            }
            return result_;
        }
        
    private:
        const MPI_Comm comm_;
        double local_start_time_;
        double result_;
    };
    
};

#endif

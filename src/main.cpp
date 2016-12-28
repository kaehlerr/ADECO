
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

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <stack>
#include <map>
#include <stdint.h>
#include <iomanip>
#include <set>

#include <sys/time.h>
#include <stdio.h>

#include "AdException.h"
#include "AdVecND.h"

#include "AdPointDataReaders.h"

#define WITH_MPI
#define VERBOSE_MODE

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "AdTypeDefinitions.h"
#include "AdUtils.h"
#include "AdMeasureMPIWallClockTime.h"
#include "AdOctree.h"
#include "AdDepositWorkFlow.h"


using namespace AdaptiveMassDeposit;



void parse_parameter_file( const std::string& filename,  AMD_Parameters& parameters )
{
    
    
    std::filebuf fb;
    if ( !fb.open (filename.c_str(),std::ios::in) )
    {
        throw AdRuntimeException( "failed to open parameter file: " + filename );
    }
    
    std::istream infile(&fb);
    
    size_t security_c = 0;
    
    parameters.stride = 1;
    parameters.max_refinement_level = 5;
    parameters.region_of_interest = AABBox( PosVec(0.,0.,0.) , PosVec(1.,1.,1.) );
    parameters.use_regions_of_interest= false;
    parameters.num_mpi_ranks_per_node = 1;
    parameters.num_pthreads = 1;
    
    while ( infile.good() && security_c++<10000 )
    {
        std::string line;
        std::getline( infile, line );
        const std::string key = line.substr(0, line.find("=")-1 );
        const std::string value = line.substr(line.find("=")+1, line.size() );
   
        if ( key.find("input_file")!=std::string::npos )
        {
            parameters.input_file = value;
        }
        else if ( key.find("output_directory")!=std::string::npos )
        {
            parameters.output_directory = value;
        }
        else if ( key.find("id_type")!=std::string::npos || key.find("file_format")!=std::string::npos )
        {
            parameters.id_type = value;
        }
        else if ( key.find("use_region_of_interest")!=std::string::npos )
        {
            parameters.use_regions_of_interest = (value.find("true")!=std::string::npos) ? true : false;
        }
        else if ( key.find("region_of_interest")!=std::string::npos )
        {
            std::vector< std::string > words;
            
            splitIntoWords( value, words );
            
            if ( words.size()!=6 )
            {
                throw AdRuntimeException("ERROR: parse_parameter_file(): 'regions_of_interest' needs 6 parameters");
            }
            
            parameters.region_of_interest = AABBox(PosVec(str2number<double>(words[0]), str2number<double>(words[2]), str2number<double>(words[4]) ),
                                                   PosVec(str2number<double>(words[1]), str2number<double>(words[3]), str2number<double>(words[5]) ));
        }
        else if ( key.find("max_refinement_level")!=std::string::npos )
        {
            parameters.max_refinement_level = str2number<int>(value);
        }
        else if ( key.find("stride")!=std::string::npos )
        {
            parameters.stride = str2number<int>(value);
        }
        else if ( key.find("max_tets_per_octree_node")!=std::string::npos )
        {
            parameters.max_tets_per_octree_node = str2number<uint64_t>(value);
        }
        else if ( key.find("linear_patch_resolution")!=std::string::npos )
        {
            parameters.linear_patch_resolution = str2number<int>(value);
        }
        else if ( key.find("num_pthreads")!=std::string::npos )
        {
            parameters.num_pthreads = str2number<int>(value);
        }
        else if ( key.find("num_mpi_ranks_per_node")!=std::string::npos )
        {
            parameters.num_mpi_ranks_per_node = str2number<int>(value);
        }
        else if ( key.find("restart_path")!=std::string::npos )
        {
            parameters.restart_path = value;
        }
        else if ( key.find("periodic_boundaries")!=std::string::npos )
        {
            parameters.periodic_boundaries = (value.find("true")!=std::string::npos) ? true : false;
        }
    }
    
    fb.close();
    
    if ( security_c++>=10000 )
    {
        throw AdRuntimeException("ERROR: Failed to parse parameter file: " + filename );
    }
    
    
}



int main(int argc, char **argv)
{
    
    using namespace AdaptiveMassDeposit;
    
    int my_rank = 0;
    int num_procs = 1;
    
    try
    {
     
        AMD_Parameters parameters;
        
        
#if 1
        if ( argc<2 )
        {
            throw AdRuntimeException( "ERROR: main(): invalid number of arguments. " );
        }
        
        parse_parameter_file( std::string(argv[1]), parameters );
#else
#endif
       
        // initialize MPI
        initialize_mpi( argc, argv, my_rank, num_procs );
        
        
        
        if ( my_rank==0 )
        {
            AD_VERBOSE( 0, { parameters.print(); } );
        }
        
        
        AD_VERBOSE( 0, { std::cerr << "INFO: (MPI rank = " << my_rank << "): initialized." << std::endl; } );
        
        assert( my_rank<num_procs );
        
        /* 
            to-do: we are currently assuming column-major order throughout the code,
                   so data readers have to transpose/remap if necessary, but we could
                   also make this a user parameter
         */
        deposit_workflow<ColumnMajorOrder>( parameters, my_rank, num_procs );
    
        CHECK_MPI_ERROR( MPI_Finalize () );
        
    }
    catch ( std::exception& ex )
    {
        std::cerr << "ERROR: (MPI rank = " << my_rank << "): Caught exception: \n" <<  ex.what() << "." << std::endl;
        
        // try to clean up MPI
        CHECK_MPI_ERROR( MPI_Finalize () );
        return 1;
    }
    
    AD_VERBOSE( 0, { std::cout << "INFO: (MPI rank = " << my_rank << "): finished successfully. \n" << std::endl; } );
 
    return 0;
    
}










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

#ifndef _AD_DEPOSIT_2_GRID_
#define _AD_DEPOSIT_2_GRID_


#include <iostream>
#include <sstream> 


#define AMD_CUDA_BLOCK_DIM 64
#define AMD_CUDA_GRID_DIM 512


namespace AdaptiveMassDeposit
{

    
class DepositWrapper
{

    
public:
    
    typedef float h_grid_mass_type;
    
    typedef float vertex_type;
    
    typedef float mass_type;
    

    static inline int get_cuda_grid_size( )
    {
        return AMD_CUDA_GRID_DIM;
    }
    
    static inline int get_cuda_block_size( )
    {
        return AMD_CUDA_BLOCK_DIM;
    }
    
    static void deposit_tets(const bool use_specific_gpu,
                             const size_t device_id_to_use,
                             const size_t tet_offset,
                             const size_t num_tets_2_process,
                             const float* positions,
                             const size_t positions_size,
                             const int    meta_cells_per_patch,
                             const float bbox[6],
                             const int dims,
                             h_grid_mass_type* data );

};
  
    

};


#endif

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

#include <iomanip>

#include <stdint.h>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <stack>
#include <limits>


#include "AdException.h"


#include "AdUtils.h"
#include "AdDeposit2Grid.h"
#include "AdTetIntersection.h"

#include "AdDepositTets2Grid.h"



using namespace AdaptiveMassDeposit;


void AdaptiveMassDeposit::DepositWrapper::deposit_tets(const bool use_specific_gpu,
                                                       const size_t gpu_device_id,
                                                       const size_t patch_offset,
                                                       const size_t num_patches_2_process,
                                                       const float* positions,
                                                       const size_t positions_size,
                                                       const int    meta_cells_per_patch,
                                                       const float bbox[6],
                                                       const int dims,
                                                       DepositWrapper::h_grid_mass_type* h_grid_mass )
{
    
#ifdef __CUDACC__
    
    std::cerr << "WARNING: AdaptiveMassDeposit::DepositWrapper::deposit_tets(): called CPU deposit code while '__CUDACC__' is defined. " << std::endl;
    assert(0);
    
#else
    
    if ( positions_size==0 )
        return;
    
    const int floats_per_patch = 3*(meta_cells_per_patch+1)*(meta_cells_per_patch+1)*(meta_cells_per_patch+1);
    const int cells_per_patch = meta_cells_per_patch*meta_cells_per_patch*meta_cells_per_patch;
    
    const size_t num_cells_to_process = cells_per_patch*num_patches_2_process;
    
    
    if ( num_cells_to_process>=std::numeric_limits<int>::max() )
    {
        throw AdRuntimeException("ERROR: deposit_tets(): num_cells will cause integer overflow.");
    }
    
    assert( positions_size%floats_per_patch==0 );
    
    int i=0;
    while ( i<num_cells_to_process )
    {
        
        const int meta_patch_offset = patch_offset + i/cells_per_patch;
        const int sub_patch_offset = i%cells_per_patch;
        
        resample_meta_patch(sub_patch_offset,
                            positions + meta_patch_offset*floats_per_patch,
                            meta_cells_per_patch,
                            bbox,
                            dims,
                            h_grid_mass);
       
        ++i;
        
        
    }

    
    
#endif
    
}


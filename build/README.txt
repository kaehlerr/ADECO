————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
LICENSING:
————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

Ralf Kaehler, December 15, 1026

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


————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
SOFTWARE REQUIREMENTS:
————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

Operating System
- Linux/Unix ( tested on Red Hat Enterprise Linux 6.7 )

Third-Party Libraries 
- cmake    ( tested with cmake version 2.8.12.2 )
- MPI 	   ( tested with Open MPI 1.8.5 )
- C++11    ( tested with gcc 4.9.2 )
- OpenMP   ( tested with gcc 4.9.2 )
- CUDA 	   ( only required for the GPU version, tested with CUDA 7.5 )


————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
COMPILATION:
————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

CPU version:
	cd ./build
	cmake -DCMAKE_BUILD_TYPE=Release ..
	make

GPU version:
	cd ./build
	cmake -DUSE_GPU=TRUE -DCMAKE_BUILD_TYPE=Release ..
	make

————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
EXECUTION:
————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

After successful compilation, the executable (named either ’dm_resampling_cpu’ or ‘dm_resampling_gpu’ ) 
will be located in directory ‘./bin’. To run it, call for example:

	mpirun -n <NUM_RANKS> ./dm_resampling_xxx PARAMETER_FILE.txt


‘PARAMETER_FILE.txt’ is a simple text file containing the following parameters:

	input_file               : filename of N-body input dataset 
	
	file_format              : input file format: ’SORTED_BY_ID’ or ‘DARK_SKY’ (see comment below)
				
				   We currently support two options for the parameter ‘file_format’:

					DARK_SKY:
						the data format used by the ‘Dark Sky Simulations’ 
						collaboration (see http://darksky.slac.stanford.edu). 
	
					SORTED_BY_ID:
						A simple binary input format that stores three coordinates using 
						32 bit (Little Endian) per coordinate. The particles are sorted 
						by their unique IDs.

				   We plan to add support for other file formats in the near future.


	
	output_directory         : name of output directory that stores the multi-resolution density field:
				   
				   Currently each leaf node in the oct-tree is stored as a separate binary file
				   (one mass value per cell in the node: 32 bit, Little Endian ). 
				   
				   We use the following naming convention for each file: 
					node_<level>_<offsetX>_<offsetY>_<offsetZ>
					level 	= refinement level (starting at 0 for the root level/node)
					offset	= offset of this node in number of cells on this refinement level
	
	use_region_of_interest   : ’true’ if code should refine only inside a region of interest (ROI) 
	
	region_of_interest       : ROI coordinates in the order: x-min x-max y-min y-max z-min z-max 
	
	max_refinement_level     : maximum refinement level in oct-tree 
	
	stride                   : subsampling factor for input data along each dimension - use 1 for full resolution
	
	max_tets_per_octree_node : refinement threshold = maximum number of tetrahedra per oct-tree node 
	
	linear_patch_resolution  : number of cells per oct-tree node along each dimension 
	
	num_mpi_ranks_per_node   : number of MPI ranks running of each node in the cluster

	num_pthreads		 : number of threads the code should launch per MPI rank 
				 	* GPU case: (’num_mpi_ranks_per_node’ * ‘num_pthreads’) must be less or equal the number of GPUs per node 
					* CPU case: no specific restrictions 	

	periodic_boundaries	 : set ‘true’ if input data assumes periodic boundaries 


Example:

input_file               =/scratch/user/test_file.bin
output_directory         =/scratch/user/
file_format              = SORTED_BY_ID
use_region_of_interest   = true
region_of_interest       = 0. 0.5 0. 0.5 0. 0.5 
max_refinement_level     = 8
stride                   = 1 
max_tets_per_octree_node = 40000
linear_patch_resolution  = 64
num_mpi_ranks_per_node   = 1
num_pthreads		 = 4
periodic_boundaries	 = false




————————————————————————————————————————————————————————————————————



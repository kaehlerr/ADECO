
/*****************************************************************
 *
 *
 *  DESCRIPTION:
 *
 *  AUTHOR: Ralf Kaehler
 *
 *  DATE: 01/15/2015
 *
 *
 *******************************************************************/

#include <iomanip>

#include <vector>
#include <stdint.h>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <stack>
#include <limits>
#include <unistd.h>

#include "AdException.h"
#include "AdDebug.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "AdDeposit2Grid.h"
#include "AdTetIntersection.h"

#include "AdDepositTets2Grid.h"


#define VERBOSE_MODE

#ifdef NDEBUG
#define CHECK_CUDA_ERROR(arg) { checkCudaError((arg), __FILE__, __LINE__); }
#else
#define CHECK_CUDA_ERROR(arg) { printf( "INFO: executing cuda call %s.\n ", #arg ); checkCudaError((arg), __FILE__, __LINE__); }
#endif


using namespace AdaptiveMassDeposit;


template <typename T> static std::string number2str( const T value )
{
        std::ostringstream os;
        if (!(os << value)) {
            throw AdRuntimeException("ERROR: AvGeneralUtils::number2string() failed.");
        }
        return os.str();
}


void checkCudaError(const cudaError_t error_code,
                    const char *file,
                    const int line,
                    const bool stop=true)
{
    if ( error_code != cudaSuccess )
    {
        
        char hostname[HOST_NAME_MAX];
        gethostname(hostname, HOST_NAME_MAX);
        
        std::cerr << "ERROR: CUDA ERROR on host: " << std::string(hostname) << " at line(" << file << "," << line << "): " << cudaGetErrorString( error_code ) << std::endl;
        if ( stop )
        {
            exit( EXIT_FAILURE );
        }
    }
}



__device__ bool check_cell(const int i, const int meta_cells_per_patch, const int floats_per_patch, const int cells_per_patch,
                           const float* positions, const dbox<float>& grid_bbox, float* cell_pos )
{
    
    const int meta_patch_offset = i/cells_per_patch;
    const int sub_patch_offset = i%cells_per_patch;
    
    sample_cell_vertices(sub_patch_offset,
                         positions + meta_patch_offset*floats_per_patch,
                         meta_cells_per_patch,
                         cell_pos);
    
        
    dbox<float> cell_bbox(dvec3<float>(cell_pos[0],cell_pos[1], cell_pos[2]),
                          dvec3<float>(cell_pos[0],cell_pos[1], cell_pos[2]) );
        
    for ( int j=0; j<24; j+=3 )
    {
        // check if any of the vertices == 666.f - if yes-> return
        // hack/to-do: magic number that indicates that this is a cell outside the domain
        if ( fabsf(cell_pos[j]-666.f) + fabsf(cell_pos[j+1]-666.f) + fabsf(cell_pos[j+2]-666.f) < 1.E-03 )
        {
            // this is in invalid cell
            return false;
        }
            
        cell_bbox.min[0] = min( cell_bbox.min[0], cell_pos[j]);
        cell_bbox.max[0] = max( cell_bbox.max[0], cell_pos[j]);
        cell_bbox.min[1] = min( cell_bbox.min[1], cell_pos[j+1]);
        cell_bbox.max[1] = max( cell_bbox.max[1], cell_pos[j+1]);
        cell_bbox.min[2] = min( cell_bbox.min[2], cell_pos[j+2]);
        cell_bbox.max[2] = max( cell_bbox.max[2], cell_pos[j+2]);
        
    }
        
    return intersection(cell_bbox, grid_bbox);
    
}

// get next cell id and share it between all threads in this warp
__device__ int get_next_cell( int* tet_counter )
{
    
    int tmp = 0;
    
    // get next cell to process on lane 0
    if (threadIdx.x % 32 == 0)
    {
        tmp = atomicAdd( tet_counter, 1);
    }
    
    // and share result to all other threads in this warp:
    const int i = __shfl( tmp, 0 );
    
    return i;

}


__global__ void resample_without_connectivity_GPU(int* tet_counter,
                                                  const float* /*__restrict___*/ positions,
                                                  const size_t positions_size,
                                                  const float* bbox,
                                                  const int dims,
                                                  const int meta_cells_per_patch,
                                                  DepositWrapper::h_grid_mass_type* h_grid_mass )

{

    const int floats_per_patch = 3*(meta_cells_per_patch+1)*(meta_cells_per_patch+1)*(meta_cells_per_patch+1);
    const int cells_per_patch = meta_cells_per_patch*meta_cells_per_patch*meta_cells_per_patch;
    
    const size_t num_cells = cells_per_patch*(positions_size/floats_per_patch);
    
    assert( positions_size%floats_per_patch==0 );
   
    
    const int grid_dims[3] = { dims, dims, dims };
    const dbox<float> grid_bbox( dvec3<float>(bbox[0],bbox[2], bbox[4]), dvec3<float>(bbox[1],bbox[3], bbox[5]) );

    while ( true )
    {
        // get next cell id which is the same for the whole warp
        const int i = get_next_cell(tet_counter);
       
        if ( i<num_cells )
        {
            // buffer for cell coordinates
            float cell_pos[24];
        
            // check if AABBox of this cell has overlap with patch domain
            if ( check_cell(i, meta_cells_per_patch, floats_per_patch,cells_per_patch, positions, grid_bbox, cell_pos ) )
            {
                // construct each of the 6 tets
                for ( int t=0; t<6; ++t )
                {
                    dvec3<float> tet_vertices[4];
                    for ( int v=0; v<4; ++v )
                    {
                        const size_t idx =  3*(tet_conn[t][v]);
                        tet_vertices [v] = dvec3<float>( cell_pos[idx],cell_pos[idx+1],cell_pos[idx+2] );
                    }
            
                    
                    {
                        const float tet_vol = get_tet_volume_times6( tet_vertices[0].getPtr(), tet_vertices[1].getPtr(), tet_vertices[2].getPtr(), tet_vertices[3].getPtr() );
                       
                        // make sure all threads base this decision on the exact same value
                        if ( __shfl(tet_vol,0)!=0.f )
                        {
                            if ( tet_vol<0.f )
                            {
                                dvec3<float> tmp = tet_vertices[1];
                                tet_vertices[1] = tet_vertices[0];
                                tet_vertices[0] = tmp;
                            }
                            deposit_tet_2_grid_refactored<float,float>( tet_vertices, fabsf(tet_vol)/6.f, 1.f, grid_dims, grid_bbox,  h_grid_mass );
                        }
                    }
                    
                }
            }
        
        }
        else
        {
            break;
        }
    
    }
    
}



struct DeviceData
{
    
    float* d_positions;
    //size_t* d_connectivity;
    size_t num_patches;
    DepositWrapper::h_grid_mass_type* d_grid_mass;
    float* d_bbox;
    int* d_counter;
    
    cudaStream_t stream;
    
};

void  resample(const bool select_device,
               const size_t device_id,
               const int cuda_grid_size,
               const int cuda_block_size,
               const size_t global_patch_offset,
               const size_t num_patches_2_process,
               const float* h_positions,
               const size_t positions_size,
               const int    meta_cells_per_patch_dim,
               const float h_bbox[6],
               const int dims,
               DepositWrapper::h_grid_mass_type* h_grid_mass )
{
    
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    
    if ( num_patches_2_process==0 )
    {
        std::cerr << "WARNING: resample( " << std::string(hostname) << "): no tets to process ( num_tets<=0 ) --- returning." << std::endl;
        return;
    }
    
    const size_t num_floats_per_patch = 3*(meta_cells_per_patch_dim+1)*(meta_cells_per_patch_dim+1)*(meta_cells_per_patch_dim+1);
    const size_t num_cells_per_patch = meta_cells_per_patch_dim*meta_cells_per_patch_dim*meta_cells_per_patch_dim;
    
    const size_t total_num_patches = positions_size/num_floats_per_patch;
    
    const size_t num_tets_per_patch = 6*num_cells_per_patch;
    
    
    if ( num_patches_2_process*num_tets_per_patch>std::numeric_limits<int>::max() )
    {
        throw AdRuntimeException( "ERROR: resample( " + std::string(hostname) + "): num_tets>=std::numeric_limits<int>::max(). Would result in overflow of GPU tet counter." );
    }
    
    if ( (global_patch_offset+num_patches_2_process)>total_num_patches )
    {
        throw AdRuntimeException( "ERROR: resample( " + std::string(hostname) + "): inconsistent point array." );
    }
    
    
    const float MAX_GPU_USAGE = 0.75f;
    
    
    const size_t resample_grid_size = size_t(dims)*size_t(dims)*size_t(dims);
    
    
    int num_gpu_devices;
    CHECK_CUDA_ERROR( cudaGetDeviceCount(&num_gpu_devices) );
    
    if ( num_gpu_devices<=0 )
    {
        throw AdRuntimeException( "ERROR: resample( " + std::string(hostname) + "): invalid device count as reported by 'cudaGetDeviceCount()'. " );
    }
    
    AD_VERBOSE( 5, {  std::cout << "INFO: resample( " << std::string(hostname) << "): number of CUDA devices: " << num_gpu_devices << std::endl; } );
   
    if ( select_device && device_id>=num_gpu_devices )
    {
        AD_VERBOSE( 5, { std::cout << "ERROR: resample( " << std::string(hostname) << "): selected invalid device number: " << device_id << " of " << num_gpu_devices << " devices. " << std::endl; } );
        throw AdRuntimeException( "ERROR: resample( " + std::string(hostname) + "): selected invalid device number: " + number2str(device_id) + " " + number2str(num_gpu_devices) );
    }
    
    size_t gpu_global_memory_in_bytes = 0;
    for ( int i=0; i<num_gpu_devices; ++i )
    {
        // do not query device information about devices used by other threads
        if ( select_device && i!=device_id )
            continue;
        
        cudaDeviceProp prop;
        if (  cudaGetDeviceProperties(&prop,i) == cudaSuccess )
        {
            AD_VERBOSE( 5, { std::cout << "INFO: resample( " << std::string(hostname) << "): number of SMs on CUDA device[" << i << "] is:  " << prop.multiProcessorCount << std::endl; } );
            // let's assume for now that all devices have the same amount of global memory
            // hack 4 debugging:
            gpu_global_memory_in_bytes = prop.totalGlobalMem;
        }
        else
        {
            throw AdRuntimeException( "ERROR: resample( " + std::string(hostname) + "): Failed to access information " + std::string(" on CUDA device: ") + number2str(i) );
        }
    }
    
    if ( resample_grid_size*sizeof(DepositWrapper::h_grid_mass_type) >= MAX_GPU_USAGE*gpu_global_memory_in_bytes )
    {
        throw AdRuntimeException( "ERROR: resample( " + std::string(hostname) + "): grid to large to store on GPU. bricking not supported yet. sorry ... " );
    }
    
    
    // hack/to-do: for now let's choose to use all devices, if the user does not select one
    const int device_count = select_device ? 1 : num_gpu_devices;
   
    
    // compute number of bytes required for storing the grid and the WHOLE positions array
    //const size_t required_bytes_per_device = (resample_grid_size*sizeof(DepositWrapper::h_grid_mass_type) + positions_size*sizeof(float) + device_count-1 ) / device_count;
   
   
    // how many bytes do we need to store the position information per device ?
    const size_t required_bytes_for_positions_per_device = (num_floats_per_patch*num_patches_2_process*sizeof(float)+device_count-1)/device_count;
        
    // first compute how much global memory we have left for the connectivity information (we assume we do not want to max it out completely, only by 90%)
    const size_t available_bytes_for_positions_per_device = MAX_GPU_USAGE*( gpu_global_memory_in_bytes-resample_grid_size*sizeof(DepositWrapper::h_grid_mass_type) );
        
    const size_t number_of_passes = (required_bytes_for_positions_per_device + available_bytes_for_positions_per_device-1)/available_bytes_for_positions_per_device;
        
    const size_t patches_per_pass_per_device  = (num_patches_2_process+device_count*number_of_passes-1)/(device_count*number_of_passes);
        
    if ( (patches_per_pass_per_device*num_floats_per_patch*sizeof(float))>available_bytes_for_positions_per_device )
    {
        throw AdRuntimeException( "ERROR: resample( " + std::string(hostname) + "): inconsistent device positions buffer size." );
    }
    
    if ( number_of_passes==0 || patches_per_pass_per_device==0 )
    {
        throw AdRuntimeException( "ERROR: resample( " + std::string(hostname) + "): number_of_passes==0 || tets_per_pass_per_device==0 . " );
    }
    
    
    AD_VERBOSE( 5, { std::cout << "INFO: resample( " << std::string(hostname) << "): using " << number_of_passes << " GPU pass(es) on " << device_count << " device(s) with " << patches_per_pass_per_device; } );
    AD_VERBOSE( 5, { std::cout << " patch(es) per pass per device." << std::endl; } );
    
    // allocate struct for data on different devices
    std::vector<DeviceData> device_data(device_count);
    
    /// loop over all devices and allocate memory resources
    for ( int i=0; i<device_data.size(); ++i )
    {
        if ( select_device )
        {
            AD_VERBOSE( 5, { std::cout << "INFO: resample( " << std::string(hostname) << "): selecting GPU device: " << device_id << std::endl; });
            CHECK_CUDA_ERROR( cudaSetDevice(device_id) );
        }
        else
        {
            AD_VERBOSE( 5, { std::cout << "INFO: resample( " << std::string(hostname) << "): selecting GPU device: " << i << std::endl; } );
            CHECK_CUDA_ERROR( cudaSetDevice(i) );
        }
        
        CHECK_CUDA_ERROR( cudaStreamCreate( &(device_data[i].stream) ) );
        
        
        // our kernel benefits from a large L1 cache:
        // important: make sure to call this methods after the device was selected (and the stream has been created) !!!
        //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
        
        
        // clear the connectivity pointer in any case, so we can securely free it at the end
        //device_data[i].d_connectivity = 0;
      
        // allocate device array for positions required for tets per pass
        CHECK_CUDA_ERROR( cudaMalloc( (void**)&(device_data[i].d_positions), patches_per_pass_per_device*num_floats_per_patch*sizeof(float) ));
        
        // allocate data array on device ( and clear it)
        CHECK_CUDA_ERROR( cudaMalloc( (void**)&(device_data[i].d_grid_mass), resample_grid_size*sizeof(DepositWrapper::h_grid_mass_type) ));
        CHECK_CUDA_ERROR( cudaMemset( device_data[i].d_grid_mass, 0, resample_grid_size*sizeof(DepositWrapper::h_grid_mass_type) ) );
        
        // allocate device array for bbox and copy them to the GPU
        CHECK_CUDA_ERROR( cudaMalloc( (void**)&(device_data[i].d_bbox), 6*sizeof(float) ));
        CHECK_CUDA_ERROR( cudaMemcpy( device_data[i].d_bbox, h_bbox, 6*sizeof(float), cudaMemcpyHostToDevice ));
        
        // allocate global tet counter in device memory space
        CHECK_CUDA_ERROR( cudaMalloc( (void**)&(device_data[i].d_counter), sizeof(int) ));
        
    }
    
    size_t processed_patches = 0;
    
    for ( size_t p=0; p<number_of_passes; ++p )
    {
        
        AD_VERBOSE( 5, { std::cout << "INFO: resample( " << std::string(hostname) << "): pass " << p+1 << " of " << number_of_passes << std::endl; } );
        
        // first copy position and/or connectivity info for this pass
        for ( int i=0; i<device_data.size(); ++i )
        {
            
            if ( select_device )
            {
                CHECK_CUDA_ERROR( cudaSetDevice(device_id) );
            }
            else
            {
                CHECK_CUDA_ERROR( cudaSetDevice(i) );
            }
            
            // reset tet counter for this device
            CHECK_CUDA_ERROR( cudaMemset( device_data[i].d_counter, 0, sizeof(int) ) );
            
            
            const size_t patch_offset = processed_patches + global_patch_offset;
            device_data[i].num_patches = std::min( total_num_patches-patch_offset, patches_per_pass_per_device );
            
            if ( (patch_offset+device_data[i].num_patches)>total_num_patches )
            {
                std::cerr << "ERROR: resample( " << std::string(hostname) << "): inconsistent patch_offset information." << std::endl;
                std::cerr << "global_patch_offset= " << global_patch_offset << "; ";
                std::cerr << "patch_offset=" << patch_offset << "; total_num_patches =" << total_num_patches << "; device_data[i].num_tets=" << device_data[i].num_patches;
                std::cerr << "; num_patch_2_process = " << num_patches_2_process << std::endl;
                throw AdRuntimeException( "ERROR: resample( " + std::string(hostname) + "): inconsistent patch_offset information." );
            }
            
            
            // copy next slice of patch array to device memory
            {
               const size_t offset = patch_offset*num_floats_per_patch;
               
               // and finally copy the buffer to device 'i'
               CHECK_CUDA_ERROR( cudaMemcpy(device_data[i].d_positions, (&(h_positions[0])+offset), device_data[i].num_patches*num_floats_per_patch*sizeof(float), cudaMemcpyHostToDevice ));
               
            }
            
            processed_patches += device_data[i].num_patches;
            
            if ( processed_patches>num_patches_2_process )
            {
                throw AdRuntimeException( "ERROR: resample( " + std::string(hostname) + "): internal error: processed_patches>num_patches_2_process. ");
            }
            
        }
        
        
        for ( int i=0; i<device_data.size(); ++i )
        {
            
            AD_VERBOSE( 5, { std::cout << "INFO: resample( " << std::string(hostname) << "): selecting GPU device: " << (select_device ? device_id : i) << std::endl; } );
            
            
            if ( select_device )
            {
                CHECK_CUDA_ERROR( cudaSetDevice(device_id) );
            }
            else
            {
                CHECK_CUDA_ERROR( cudaSetDevice(i) );
            }
            
            
            AD_VERBOSE( 5, { std::cout << "INFO: resample( " << std::string(hostname) << "): Pass " << p << ": calling resample_without_connectivity_GPU for device " << (select_device ? device_id : i) << ". num_tets = " << device_data[i].num_patches << std::endl; } );
            
            resample_without_connectivity_GPU<<<cuda_grid_size,cuda_block_size,0,device_data[i].stream>>>(device_data[i].d_counter,
                                                                                                          device_data[i].d_positions,
                                                                                                          device_data[i].num_patches*num_floats_per_patch,
                                                                                                          device_data[i].d_bbox, dims,
                                                                                                          meta_cells_per_patch_dim, //get_meta_mesh_block_size(0),
                                                                                                          device_data[i].d_grid_mass );
                
            AD_VERBOSE( 5, {  std::cout << "INFO: resample( " << std::string(hostname) << "): Device: " << (select_device ? device_id : i) << " called for pass: " << p << ". Processing " << processed_patches << " patches" << std::endl; } );
            
        }
        
        
    }  ///// end loop over rounds
    
    // copy back data array from device to host
    for ( int i=0; i<device_data.size(); ++i )
    {
        
        
        if ( select_device )
        {
            CHECK_CUDA_ERROR( cudaSetDevice(device_id) );
        }
        else
        {
            CHECK_CUDA_ERROR( cudaSetDevice(i) );
        }

        
        AD_VERBOSE( 5, { std::cerr << "INFO: resample( " << std::string(hostname) << "): Waiting for CUDA kernel in stream " << i << " to finish. " << std::endl; } );
    
        // check if kernel on device is finished yet
        if ( cudaStreamSynchronize(device_data[i].stream)==cudaSuccess )
        {
            std::vector<DepositWrapper::h_grid_mass_type> h_tmp_grid_data(resample_grid_size);
        
            // this call blocks, so once it returns, we can add the temporary array to the final result
            CHECK_CUDA_ERROR(  cudaMemcpy( &(h_tmp_grid_data[0]), device_data[i].d_grid_mass, resample_grid_size*sizeof(DepositWrapper::h_grid_mass_type), cudaMemcpyDeviceToHost ) );
        
            // and add its contribution to the main grid
            for ( size_t c=0; c<resample_grid_size; ++c )
            {
                h_grid_mass[c] += h_tmp_grid_data[c];
            }
        
        }
        else
        {
            std::cerr << "ERROR: resample( " << std::string(hostname) << "): Failed to finish kernel call on device " << i;
            if ( select_device )
            {
                std::cerr << " device id =  " << device_id;
            }
            std::cerr << std::endl;
        }
    
    
        CHECK_CUDA_ERROR( cudaFree( device_data[i].d_positions ) );

        //CHECK_CUDA_ERROR( cudaFree( device_data[i].d_connectivity ) );
        CHECK_CUDA_ERROR( cudaFree( device_data[i].d_grid_mass ) );
        CHECK_CUDA_ERROR( cudaFree( device_data[i].d_bbox ) );
        CHECK_CUDA_ERROR( cudaFree( device_data[i].d_counter ) );
    
        CHECK_CUDA_ERROR( cudaStreamDestroy( device_data[i].stream ) );
   
    }

    
}



void DepositWrapper::deposit_tets(const bool use_specific_gpu,
                                  const size_t device_id_to_use,
                                  const size_t tet_offset,
                                  const size_t num_tets_2_process,
                                  const float* positions,
                                  const size_t positions_size,
                                  const int    meta_cells_per_patch,
                                  const float bbox[6],
                                  const int dims,
                                  DepositWrapper::h_grid_mass_type* data )
{

#ifdef __CUDACC__
    
    
    assert( tet_offset==0 );
    assert( num_tets_2_process==connectivity_size/4 );
    
    AD_VERBOSE( 5, { std::cout << "INFO: DepositWrapper::deposit_tets(): cuda grid/block size = [ " << AMD_CUDA_GRID_DIM << ", " << AMD_CUDA_BLOCK_DIM << "]" << std::endl; } );
    

    resample(use_specific_gpu,
             device_id_to_use,
             AMD_CUDA_GRID_DIM,
             AMD_CUDA_BLOCK_DIM,
             tet_offset,
             num_tets_2_process,
             positions,
             positions_size,
             meta_cells_per_patch,
             bbox,
             dims,
             data );


#else
    
    std::cerr << "WARNING: DepositWrapper::deposit_tets(): CPU version called from CUDA runtime. returning ... " << std::endl;

#endif
    
}


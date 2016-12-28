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




#ifndef _AD_OCTREE_GRID_WRITER_
#define _AD_OCTREE_GRID_WRITER_

#include "AdTypeDefinitions.h"

namespace AdaptiveMassDeposit
{

class AdOctreeGridWriter
{
    
public:
    virtual void write_meta_data(const std::string& output_directory,
                                 const std::string& filename,
                                 const unsigned int linear_patch_resolution,
                                 const TetOctree& octree )
    {
        std::cout << "INFO: AdOctreeGridWriter::store_meta_data(): called for base class. No meta data will be written." << std::endl;
    }
    
    
    virtual double write_patch(const std::string& output_path,
                               const TetOctree::Node& node,
                               const TetOctree& octree,
                               const int grid_dims,
                               const std::vector<float>& orig_data )
    {
        double mass = 0.;
        
        const double bbox[6] =
        {
            node.getLowerLeft()[0], node.getLowerLeft()[0]+node.getExtension(),
            node.getLowerLeft()[1], node.getLowerLeft()[1]+node.getExtension(),
            node.getLowerLeft()[2], node.getLowerLeft()[2]+node.getExtension()
        };
        
        // now analyze the result
        const double cell_size[3] =
        {
            (bbox[1]-bbox[0])/grid_dims, (bbox[3]-bbox[2])/grid_dims, (bbox[5]-bbox[4])/grid_dims,
        };
        
        const double inv_cell_vol = 1./(cell_size[0]*cell_size[1]*cell_size[2]);
        
        std::vector<float> data( orig_data.size() );
       
        // map data array from mass per cell to densities
        for ( size_t i=0; i<data.size(); ++i )
        {
            mass += orig_data[i];
            data[i] = orig_data[i]*inv_cell_vol;
        }
        
        const int level = node.getLevel();
        
        const double delta[3] = {
            node.getExtension()/grid_dims,
            node.getExtension()/grid_dims,
            node.getExtension()/grid_dims
        };
        
        const double origin[3] = {
            node.getLowerLeft()[0],
            node.getLowerLeft()[1],
            node.getLowerLeft()[2]
        };
        
        const long long int iorigin[3] = {
            static_cast<long long int>(rint( (origin[0]-octree.getNode(0).getLowerLeft()[0])/delta[0]) ),
            static_cast<long long int>(rint( (origin[1]-octree.getNode(0).getLowerLeft()[1])/delta[1]) ),
            static_cast<long long int>(rint( (origin[2]-octree.getNode(0).getLowerLeft()[2])/delta[2]) )
        };
        
        const std::string filename =
            output_path + std::string("/node_") +
            number2str( level ) + "_" +
            number2str( iorigin[0] ) + "_" +
            number2str( iorigin[1] ) + "_" +
            number2str( iorigin[2] );
        
        std::ofstream out(filename.c_str(),std::ios_base::binary);
       
        if( out.good() )
        {
            out.write((char *)&(data[0]),sizeof(float)*data.size());
            out.close();
            AD_VERBOSE( 3, { std::cout << "INFO: write_patch(): wrote node data: " << filename << std::endl; } );
        }
        else
        {
            std::cerr << "ERROR: write_patch(): failed to write node data: " << filename << std::endl; std::cerr.flush();
            throw AdRuntimeException( "ERROR: write_patch(): failed to write node data." );
        }
        
        std::cout << "INFO: AdOctreeGridWriter::write_patch(): wrote file: " << filename << std::endl;
        
        return mass;
        
    }

    
    
    
};

    
    
  
    
};

#endif

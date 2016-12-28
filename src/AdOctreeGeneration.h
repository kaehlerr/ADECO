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


#ifndef _ADM_OCTREE_GENERATION_
#define _ADM_OCTREE_GENERATION_

#include "AdTypeDefinitions.h"
#include "AdOctree.h"


namespace AdaptiveMassDeposit
{
    

template <class id_functor> class AdOctreeGeneration
{
    
public:
    
    struct LocalCoverageInfo
    {
        uint64_t    num_tets;
        //float       intersection_vol;
    };
    
    
    static inline AABBox getBBox( const TetOctree::Node& node )
    {
        AABBox res;
        res.min = node.getLowerLeft();
        res.max = node.getLowerLeft()+PosVec(node.getExtension());
        return res;
    }
    
    
    static void generate_tree(const int num_procs,
                              const int my_rank,
                              const AABBox& box,
                              const AABBox& ROI,
                              const uint64_t max_tets_per_node,
                              const int max_refinement_level,
                              const MetaMesh<id_functor>& tets_mesh,
                              TetOctree& octree )
    {
        
        
        AD_VERBOSE( 5, { std::cout << "INFO: generate_tree(): entered routine." << std::endl; } );
        
        
        AABBox effective_roi = ROI;
        // in case of an invalid ROI, set it to global domain
        if ( ROI.min[0]>ROI.max[0] || ROI.min[1]>ROI.max[1] || ROI.min[2]>ROI.max[2] )
        {
            effective_roi = box;
        }
	       
        // add root node to octree
        octree.setRootNode( box.min, std::max(std::max(box.max[0]-box.min[0],box.max[1]-box.min[1]),box.max[2]-box.min[2]) );
        
        tets_mesh.get_cell_ids( octree.getNode(0).getNodeData().local_meta_cell_ids );
        
        // allocate stack for octree nodes
        std::stack<node_id_t> nodes_2_visit;
        
        // and push the root node to stack
        nodes_2_visit.push( 0 );
        
        uint64_t num_tets_in_leaves=0;
        
        // while loop over stack
        size_t security_counter = 0;
        int leaf_counter = 0;
        
        while ( !nodes_2_visit.empty() && security_counter++< 100000000 )
        {
            
            // get next node from octree stack
            const node_id_t current_node_id = nodes_2_visit.top();
            nodes_2_visit.pop();
            
            TetOctree::Node& current_node = octree.getNode(current_node_id);
            
            // change this: need to loop over all meta_cells and request estimate for num_tets based on intersection with node's bbox
            unsigned long local_tets_in_node = 0; //...;// current_node.getNodeData().local_tet_ids.size();
            const AABBox current_bbox = getBBox(current_node);
            for ( size_t m=0; m<current_node.getNodeData().local_meta_cell_ids.size(); ++m )
            {
                local_tets_in_node += tets_mesh.get_estimate_num_tets( current_bbox, current_node.getNodeData().local_meta_cell_ids[m] );
            }
            
            
            // broadcast actual bbox and obtain total number of particles using reduce operation
            unsigned long global_tets_in_node = 0;
            
            // prepare data for this node
            {
                // first initialize structure members
                LocalCoverageInfo local_coverage_info;
                local_coverage_info.num_tets = local_tets_in_node;
                
                std::vector<LocalCoverageInfo> receive_buffer(num_procs);
                CHECK_MPI_ERROR ( MPI_Allgather (reinterpret_cast<char*>(&local_coverage_info),
                                                 sizeof(LocalCoverageInfo),
                                                 MPI_CHAR,
                                                 reinterpret_cast<char*>(&(receive_buffer[0])),
                                                 sizeof(LocalCoverageInfo),
                                                 MPI_CHAR,
                                                 MPI_COMM_WORLD ) );
                
                current_node.getNodeData().num_tets_on_proc_estimate.resize(num_procs);
                for ( size_t i=0; i<size_t(num_procs); ++i )
                {
                    current_node.getNodeData().num_tets_on_proc_estimate[i] = receive_buffer[i].num_tets;
                    global_tets_in_node += receive_buffer[i].num_tets;
                }
                
            }
            
            current_node.getNodeData().num_tets_covering_node_estimate =  global_tets_in_node;
            
            // we only refine this node if it intersects the region of interest as defined by ROI
            if ( intersection( effective_roi, current_bbox ) )
            {
                
                if ( global_tets_in_node>0 && global_tets_in_node>max_tets_per_node && current_node.getLevel()<max_refinement_level )
                {
                    // to-do: this function probably would benefit from parallelization via OpenMP
                    distribute_tets_to_children( num_procs, my_rank, current_node_id, tets_mesh, octree );
                    
                    AD_ASSERT( octree.getNode(current_node_id).isRefined(), "internal error - refinement failed" );
                    
                    for ( int i=0; i<8; ++i )
                    {
                        const node_id_t child_id = octree.getNode(current_node_id).getChildId(i);
                        nodes_2_visit.push( child_id );
                        AD_ASSERT( child_id>0, "internal error - invalid child ptr" );
                    }
                    
                    // did the refinement work as expected ?
                    AD_ASSERT( octree.getNode(current_node_id).getNodeData().local_meta_cell_ids.empty(), "internal error - refinement failed" );
                    
                }
                else
                {
                    AD_ASSERT( octree.getNode(current_node_id).isRefined()==false, "internal error" );
                    num_tets_in_leaves += octree.getNode(current_node_id).getNodeData().num_tets_covering_node_estimate;
                    if ( my_rank==0 )
                    {
                        AD_VERBOSE( 3, { std::cout << "INFO: contruct_tree(): added leaf node number " << leaf_counter << std::endl; } );
                        leaf_counter++;
                    }
                    
                }
            }
            
            
        } // end while-loop
        
        
        if ( !nodes_2_visit.empty() )
        {
            throw AdRuntimeException("ERROR: generate_tree(): loop overflow during octree generation. Security counter = " + number2str(security_counter) );
        }
        
        
        AD_VERBOSE( 0, { std::cout << "INFO: generate_tree(): number of levels " << octree.getNumLevels() << std::endl; } );
        
    }
    
   
    
    // compute number of particles loaded by this processor
    static void distribute_tets_to_children(const int num_procs, const int my_rank,
                                            const node_id_t& current_node_id,
                                            const MetaMesh<id_functor>& tets_mesh,
                                            TetOctree& octree )
    {
        
        AD_ASSERT( !octree.getNode(current_node_id).isRefined(), "internal error - node already refined"  );
        
        // node contains data, so we need to refine it. do not store reference to node !!! (see implementation of refineNode)
        octree.refineNode(current_node_id);
        
        AD_ASSERT( octree.getNode(current_node_id).isRefined(), "internal error - refinement failed"  );
        
        // cache ids of children for faster access in the particle loop
        const node_id_t first_child_ptr = octree.getNode(current_node_id).getChildId(0);
        
        const std::vector<uint64_t>& parent_meta_cell_ids = octree.getNode(current_node_id).getNodeData().local_meta_cell_ids;
        
        // initialize child node data
        for ( int i=first_child_ptr; i<first_child_ptr+8; ++i )
        {
            octree.getNode(i).getNodeData().num_tets_on_proc_estimate.resize(num_procs);
            AD_ASSERT( octree.getNode(i).getNodeData().local_meta_cell_ids.size()==0, "internal error" );
        }
        
        for ( size_t i=0; i<parent_meta_cell_ids.size(); ++i )
        {
            const AABBox& meta_cell_bbox = tets_mesh.get_bbox( parent_meta_cell_ids[i] );
            
            for (int child_idx = 0; child_idx<8; ++child_idx )
            {
                const node_id_t child_id = first_child_ptr+child_idx;
                const AABBox child_bbox = getBBox(octree.getNode(child_id));
                if ( intersection( meta_cell_bbox, child_bbox) )
                {
                    octree.getNode(child_id).getNodeData().local_meta_cell_ids.push_back(parent_meta_cell_ids[i]);
                    octree.getNode(child_id).getNodeData().num_tets_on_proc_estimate[my_rank] += tets_mesh.get_estimate_num_tets( child_bbox, parent_meta_cell_ids[i] );
                }
            }
        }
        
        // get really rid of current/parent node's ids
        std::vector<uint64_t> tmp;
        octree.getNode(current_node_id).getNodeData().local_meta_cell_ids.swap( tmp );
        
    }
    
    
    
}; // end class OctreeContruction

}; // end namespace AdaptiveMassDeposit


#endif


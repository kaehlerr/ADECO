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



#ifndef _AMD_OCTREE_
#define _AMD_OCTREE_


#include <stdio.h>
#include <stack>
#include <cstdlib>

#include "AdTypeDefinitions.h"


namespace AdaptiveMassDeposit
{

// we need signed this for node id, since -1 is used as a flag in some routines (e.g. exchange_tets)
typedef int    node_id_t;
const int INVALID_NODE_ID = ~(int(0));

template <typename pos_type, class T> class Octree
{
    
public:
    
    class Node
    {
    public:
        Node( const AdVec3D<pos_type>& lower_left, const pos_type extension, const int level, const node_id_t parent ) :
        min_pos_(lower_left),
        ext_(extension),
        first_child_id_(-1),
        level_(level),
        parent_id_(parent)
        {
            if ( ext_<=0.f )
                throw AdRuntimeException("ERROR: Node(): extension <= 0.");
            
            if ( level_<0 )
                throw AdRuntimeException("ERROR: Node(): level <= 0.");
            
            if ( parent_id_<-1 || (parent_id_==-1 && level_>0) )
                throw AdRuntimeException("ERROR: Node(): invalid parent id");

        }
        
        inline bool isRefined() const
        {
            return (first_child_id_>0);
        }
        
        inline bool isLeaf() const
        {
            return !isRefined();
        }
        
        inline AdVec3D<pos_type> getCenter() const
        {
            return (min_pos_ + pos_type(0.5)*AdVec3D<pos_type>( ext_,ext_,ext_));
        }
        
        
        inline bool containsPoint( const AdVec3D<pos_type>& pos ) const
        {
            return (pos[0]>=min_pos_[0] &&  pos[0]<=(min_pos_[0]+ext_) &&
                    pos[1]>=min_pos_[1] &&  pos[1]<=(min_pos_[1]+ext_) &&
                    pos[2]>=min_pos_[2] &&  pos[2]<=(min_pos_[2]+ext_));
        }
        
        inline node_id_t getChildId( const node_id_t num ) const
        {
            AD_ASSERT( isRefined(), "ERROR: getChildId(): requested child id for unrefined node");
            AD_ASSERT( num>=0 && num<8, "ERROR: getChildId(): requesting invalid node id");
            return (first_child_id_+num);
        }
        
        inline int getLevel() const
        {
            AD_ASSERT( level_>=0, "internal error: level<0" );
            return level_;
            
        }
        
        inline node_id_t getParentId() const
        {
            AD_ASSERT( (level_==0&&parent_id_==-1) || (level_>0&&parent_id_>=0), "invalid parent pointer" );
            return parent_id_;
        }
        
        inline T& getNodeData()
        {
            return node_data_;
        }
        
        inline const T& getNodeData() const
        {
            return node_data_;
        }
        
        inline pos_type getExtension() const
        {
            AD_ASSERT( ext_>0.f, "invalid node extension" );
            return ext_;
        }
        
        inline const AdVec3D<pos_type>& getLowerLeft() const
        {
            return min_pos_;
        }
        
        inline void setFirstChildId( const node_id_t child_id )
        {
            
            AD_ASSERT( first_child_id_==-1, "internal error - node already refined"  );
            if ( child_id<=0 )
                throw AdRuntimeException("ERROR: setFirstChildNode(): invalid node id");
            else
                first_child_id_ = child_id;
        }
       
        
    private:
        
        // lower left corner of node
        AdVec3D<pos_type> min_pos_;
        // length of box in one dimension - could alternatively store the depth of the node in the hierarchy
        pos_type ext_;
        // indices to children in Octree vector - '0' indicates that a particular child is not used
        // processor id resonsible for this node, if it is a leaf
        T node_data_;
        
        node_id_t first_child_id_;
        int level_;
        node_id_t parent_id_;
    
    };
    
    
    Octree() : num_levels_(0)
    {};
    
    // Octree vector: first entry stores root node
    typedef std::vector<Node> Nodes;
    
    inline node_id_t setRootNode( const AdVec3D<pos_type>& lower_left, const pos_type extension )
    {
        // first remove old data, if present
        clear();
        const node_id_t node_id = addNode_( Node(lower_left, extension, 0, -1) );
        num_levels_ = 1;
        AD_ASSERT( nodes_.size()==1 && node_id==0, "internal error" );
        return node_id;
    }
    
    
    // refine node and add children to Nodes array
    // we can't work work references to node here, since 'addNode'
    // resizes the node vector and might invalidate the reference ...
    inline void refineNode( const node_id_t node_id )
    {
        
        // address of node might change after "addNode" call, so do not use references to node or its attributes
        getNode(node_id).setFirstChildId(nodes_.size()); AD_ASSERT( !nodes_.empty(), "internal error - octree data inconsistent" );
        AD_ASSERT( size_t(getNode(node_id).getChildId(0)) == nodes_.size(), "internal error - refinement failed" );
       
        // no it should be safe to use the parent reference
        const Node& parent = getNode(node_id);
        
        const pos_type child_ext = parent.getExtension()/2.;
        const AdVec3D<pos_type> min_pos = parent.getLowerLeft();
        const int level = parent.getLevel();
        
        unsigned int c = 0;
        // if this turns out to burn a lot of CPU cycle, maybe unroll the loop ...
        for ( int k=0; k<2; ++k )
        {
            const pos_type child_pos_z = min_pos[2] + k*child_ext;
            for ( int j=0; j<2; ++j )
            {
                const pos_type child_pos_y = min_pos[1] + j*child_ext;
                for ( int i=0; i<2; ++i )
                {
                    const pos_type child_pos_x = min_pos[0] + i*child_ext;
                    // add new child to
                    const node_id_t new_id = addNode_( Node(AdVec3D<pos_type>( child_pos_x, child_pos_y, child_pos_z ), child_ext, level+1, node_id ) ) ;
                    AD_ASSERT( isValid(new_id) && size_t(getNode(node_id).getChildId(0)+c)==size_t(new_id) , "internal error: child with invalid node ptr" );
                    ++c;
                }
            }
        }
        
        AD_ASSERT( c==8, "internal error refining nodes" );
        
        
    }
    
    inline Node getRootNode( ) const
    {
        if ( nodes_.empty() )
        {
            throw AdRuntimeException("ERROR: Octree::getRootNode(): tree is empty.");
        }
        return nodes_[0];
    }
    
    inline Node& getNode( const unsigned int node_id )
    {
        AD_ASSERT( isValid(node_id), "invalid octree index" );
        return nodes_[node_id];
    }
    
    inline const Node& getNode( const unsigned int node_id ) const
    {
        AD_ASSERT( isValid(node_id), "invalid octree index" );
        return nodes_[node_id];
    }
    
    inline Nodes& getNodes()
    {
        return nodes_;
    }
    
    inline size_t getLeafNodes( std::vector<node_id_t>& leaf_ids ) const
    {
        
        leaf_ids.clear();
        
        // allocate stack for octree nodes
        std::stack<node_id_t> nodes_2_visit;
        // start at root node
        nodes_2_visit.push(0);
        
        size_t security_counter = 0;
        while ( !nodes_2_visit.empty() && security_counter++< 100000000 )
        {
            // get next entry from nodes stack
            const node_id_t& current_node_id = nodes_2_visit.top();
            nodes_2_visit.pop();
            
            const Node& current_node = getNode(current_node_id);
            if ( current_node.isRefined() )
            {
                for ( int i=0; i<8; ++i )
                {
                    nodes_2_visit.push( current_node.getChildId(i) );
                }
            }
            else
            {
                leaf_ids.push_back(current_node_id);
            }
            
        }
        
        if ( !nodes_2_visit.empty() )
        {
            std::cerr << "WARNING: AMD_Octree::getLeafNodes(): premature end of octree traversal. Security counter = " << security_counter << std::endl;
            exit(1);
        }
        
        return leaf_ids.size();
        
    }
    
    
    
    inline int getNumLevels() const
    {
        return num_levels_;
    }
    
    inline void print_info( const bool leaves_only)
    {
        
        // allocate stack for octree nodes
        std::stack<node_id_t> nodes_2_visit;
        nodes_2_visit.push(0);
        
        size_t leaves = 0;
        size_t security_counter = 0;
        
        while ( !nodes_2_visit.empty() && security_counter++< 100000000 )
        {
            
            // get next node from octree stack
            const Node& current_node = getNode( nodes_2_visit.top() );
            nodes_2_visit.pop();
            
            if ( current_node.isRefined() )
            {
                for ( int i=0; i<8; ++i )
                {
                    nodes_2_visit.push( current_node.getChildId(i) );
                }
            }
            else
            {
                
                std::cout << "INFO: AMD_Octree::print_info(): Detected leaf number: " << leaves << " on level " << current_node.getLevel() << "." << std::endl;
                std::cout << "  leaf data info: " << std::endl;
                current_node.getNodeData().print_info();
                std::cout.flush();
                
                ++leaves;
            }
        
            
        }
        
        if ( !nodes_2_visit.empty() )
        {
            std::cerr << "ERROR: AMD_Octree::print_info(): premature end of octree traversal. Security counter = " << security_counter << std::endl;
            exit(1);
        }
        
        
        
    }
    

    inline bool isValid( const node_id_t node_id ) const
    {
        return node_id>=0 && size_t(node_id)<nodes_.size();
    }
    
    inline void clear()
    {
        Nodes tmp_nodes;
        std::swap( nodes_, tmp_nodes );
        nodes_.clear();
    }
    
    
    
private:
    
    inline node_id_t addNode_( const Node& node )
    {
        nodes_.push_back( node );
        num_levels_ = std::max( node.getLevel()+1, num_levels_ );
        return nodes_.size()-1;
    }
    
    
    Nodes nodes_;
    
    int num_levels_;
    
    
};

};

#endif

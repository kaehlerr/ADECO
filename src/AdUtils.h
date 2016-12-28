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


#ifndef _ADM_UTILS_
#define _ADM_UTILS_

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <limits.h>
#include <inttypes.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <map>
#include <vector>

#if __APPLE__
#include <unordered_map>
#define unordered_map std::unordered_map
#else
#include <tr1/unordered_map>
#define unordered_map std::tr1::unordered_map
#endif

#include "AdException.h"
#include "AdTypeDefinitions.h"

#ifdef _OPENMP
#include <omp.h>
#endif





namespace AdaptiveMassDeposit
{
    
    

    template <class T> static inline void shrink_to_fit( std::vector<T>& vec )
    {
#if __cplusplus <= 199711L
        std::cout << "WARNING: current compiler does not fully support C++11. Implementing shrink_to_fit using swap." << std::endl << std::endl;
        std::vector<T>(vec).swap(vec);
#else
        vec.shrink_to_fit();
#endif
        if ( vec.size()<vec.capacity() )
        {
            std::cerr << "WARNING: clear_vec(): size() == " << vec.size() << "; capacity== " << vec.capacity() << std::endl;
        }
    }
    
    template <class T> inline void clear_vec( std::vector<T>& vec )
    {
        vec.clear();
        shrink_to_fit(vec);
    }
    
#ifdef _OPENMP
    inline int get_num_openmp_threads() { return omp_get_num_threads(); }
    inline int get_openmp_thread_id() { return omp_get_thread_num(); }
#else
    inline int get_num_openmp_threads() { return 1; }
    inline int get_openmp_thread_id() { return 0; }
#endif
    
    template <class key, class value>
    void move_map_to_vector( std::map<key,value>& map, std::vector<value>& result )
    {
        //result.resize( map.size() );
        result.clear();
        
        size_t c = 0;
        for( typename std::map<key,value>::iterator it = map.begin(); it != map.end(); ++it )
        {
            // delete map data up to this point to avoid using too much memory
            if ( c==1000000 )
            {
                map.erase( map.begin(),it);
                c=0;
            }
            result.push_back(it->second);
            ++c;
        }
        
        // and clear the rest of the map
        map.clear();
        
    }

    
    inline void get_offset_and_number(const size_t num_items, const size_t current_iteration, const size_t num_iterations,
                                      size_t& item_offset, size_t& num_items_for_this_iteration )
    {
        
        if ( current_iteration>=num_iterations )
        {
            throw AdRuntimeException( "ERROR: get_offset_and_number(): inconsistent iteration parameters." );
        }
        
        const size_t items_per_iteration = (num_items + num_iterations-1) / num_iterations; assert( (items_per_iteration*num_iterations)>=num_items );
        item_offset = std::min( num_items, current_iteration*items_per_iteration);
        
        num_items_for_this_iteration = std::min( items_per_iteration, num_items-item_offset );
        
        assert( (item_offset+num_items_for_this_iteration)<=num_items );
        assert( (current_iteration+1)<num_iterations || (item_offset+num_items_for_this_iteration)==num_items );
        
    }
    
    
    
    template <class key, class value>
    void move_map_to_vector( unordered_map<key,value>& map, std::vector<value>& result )
    {
        //result.resize( map.size() );
        result.clear();
        
        size_t c = 0;
        for( typename unordered_map<key,value>::iterator it = map.begin(); it != map.end(); ++it )
        {
            // delete map data up to this point to avoid using too much memory
            if ( c==1000000 )
            {
                map.erase( map.begin(),it);
                c=0;
            }
            result.push_back(it->second);
            ++c;
        }
        
        // and clear the rest of the map
        map.clear();
        
    }

    template <typename T>
    inline T ceil_div( const T x, const T y )
    {
        return x/y + T(x%y>0);
    }
    
    template <typename T>
    static std::string number2str( const T value )
    {
        std::ostringstream os;
        if (!(os << value)) {
            throw AdRuntimeException("ERROR: AvGeneralUtils::number2string() failed.");
        }
        return os.str();
    }
    
    
    template <typename T>
    static T str2number( const std::string& string )
    {
        std::istringstream i(string);
        T num;
        if (!(i >> num)) {
            throw AdRuntimeException("ERROR: AvGeneralUtils::str2number() failed for string: " + string + "\n" );
        }
        return num;
    }
    
    
    
    template <typename T>
    static void remove_duplicates( std::vector<T>& data, bool is_sorted = false )
    {
        if (!is_sorted)
            std::sort( data.begin(), data.end() );
        typename std::vector<T>::iterator last = std::unique(data.begin(), data.end());
        data.erase(last, data.end());
    }
    
    static inline std::string getPath( const std::string& full_name )
    {
        std::string result = full_name;
        std::string::size_type posSlash = result.rfind("/");
        if ( posSlash==std::string::npos ) {
            return std::string();
        }
        
        result.erase( posSlash+1, result.size() );
        return result;
    }
    
    static inline size_t splitIntoWords( const std::string& line, std::vector< std::string >& words )
    {
        words.clear();
        std::stringstream strstr(line);
        std::string word;
        while (strstr >> word) {
            words.push_back(word);
        }
        return words.size();
    }

    
    class MeasureTime
    {
        
    public:
        MeasureTime() : total_time_(0.) {}
        
        inline void start()
        {
            gettimeofday(&t1, NULL);
        }
        
        inline void stop()
        {
            gettimeofday(&t2, NULL);
        }
        
        inline void add_measured_time()
        {
            stop();
            const double measured_time = ( (double) (t2.tv_usec - t1.tv_usec) / 1000000 + (double) (t2.tv_sec - t1.tv_sec) );
            total_time_ += measured_time;
        }
        
        inline double get_total_time() const
        {
            return total_time_;
        }
        
        inline void reset_total_time()
        {
            total_time_ = 0.;
        }
        
        
    private:
        struct timeval t1, t2;
        double total_time_;
    
    };
    

    static inline size_t get_file_size( const std::string& filename )
    {
        struct stat filestatus;
        if ( stat( filename.c_str(), &filestatus )==0 )
            return filestatus.st_size;
        else
            throw AdRuntimeException("Failed to compute size of file: " + filename);
    }
    
    
    template<typename T>
    size_t get_vector_bytes_size ( const typename std::vector<T>& v )
    {
        return sizeof(T)*v.size();
    }
    
    
    class SortFlaggedArray
    {
        
    public:
        
        SortFlaggedArray()
        {
            assert(unit_test_());
        }
        
        
        /*
         cutoff is the 'id' of the first element for which 'flag_array==true' holds
         cutoff == vec.size() indicates that no element was flagged
         */
        template <class T>
        static inline size_t move_flagged_2_end( std::vector<bool>& flag_array, std::vector<T>& vec )
        {
            
            if ( flag_array.size()!=vec.size() )
            {
                throw AdRuntimeException( "ERROR: move_flagged_2_end(): vector and flag array have different sizes." );
            }
           
            if ( flag_array.empty() )
            {
                return 0;
            }
            
            size_t c1 = vec.size()-1;
            size_t c2 = c1-1;
            
            while ( c1>0 && c2<c1 )
            {
                assert( c1<vec.size()&&c2<vec.size() );
                
                if ( flag_array[c1]==true )
                {
                    --c1;
                    --c2;
                }
                else
                {
                    if ( flag_array[c2]==true )
                    {
                        std::swap( vec[c1], vec[c2] );
                        {
                            // std::vector<bool> is a specialization, so std::swap does not work for it .... std::swap( flag_array[c1],  flag_array[c2] );
                        	// swap it manually
                        	const bool tmp = flag_array[c2];
                            flag_array[c2] = flag_array[c1];
                            flag_array[c1] = tmp;
                        }
                        --c1;
                        --c2;
                    }
                    else
                    {
                        --c2;
                    }
                }
                
            }
            
            const size_t cutoff = (flag_array[c1]==true) ? c1 : c1+1;
            
            assert( sorted_( flag_array, cutoff ) );
            
            return cutoff;
        }
        
    private:
        
        static inline bool sorted_( const std::vector<bool>& flag_array, const size_t cutoff )
        {
            
            for ( size_t i=0; i<flag_array.size(); ++i )
            {
                if ( i>=cutoff && flag_array[i]==false )
                {
                    assert(0);
                    return false;
                }
                
                if ( i<cutoff && flag_array[i]==true )
                {
                    assert(0);
                    return false;
                }
                
            }
            
            return true;
            
        }
        
        
        bool unit_test_()
        {
            std::vector<bool> test_set(100000);
            std::vector<int> vec(test_set.size());
            
            srand(time(0));
            
            for ( size_t i=0; i<test_set.size(); ++i )
            {
                test_set[i] = bool(rand() % 2);
            }
            
            const size_t cutoff = move_flagged_2_end( test_set, vec );
            return sorted_( test_set, cutoff );
            
        }
        
    };
    
    
    namespace ProcessLayout
    {
        
        static void find_factors( const unsigned int num, std::vector<unsigned int>& res )
        {
            res.clear();
            res.push_back(1);
            
            for( unsigned int i=2; i<num; ++i )
            {
                if ( num%i==0 )
                {
                    res.push_back(i);
                }
            }
            res.push_back(num);
            
        }
        
        
        static void find_dims( const unsigned int num, std::vector<unsigned int> d[3])
        {
            
            std::vector<unsigned int> tmp[3];
            
            find_factors( num, tmp[0]);
            
            for ( size_t j=0; j<tmp[0].size(); ++j )
            {
                find_factors(tmp[0][j],tmp[1]);
                
                for ( size_t k=0; k<tmp[1].size(); ++k )
                {
                    d[0].push_back(num/tmp[0][j]);
                    d[1].push_back(tmp[0][j]/tmp[1][k]);
                    d[2].push_back(tmp[1][k]);
                    
                    assert( d[0].back()*d[1].back()*d[2].back() == num );
                    assert( d[0].size()==d[1].size() && d[1].size()==d[2].size() );
                    
                    //std::cout << d[0].size()-1 << ": " << d[0].back() << " " << d[1].back() << " " << d[2].back() << std::endl;
                }
                
            }
            
            
        }
        
        
        static void find_best_match( const unsigned int num, const unsigned int* src_dims, unsigned int* dst_dims )
        {
            
            dst_dims[2] = dst_dims[1] = dst_dims[0] = 0;
            
            std::vector<unsigned int> d[3];
            find_dims( num, d );
            
            const float ideal_ratio1 = float(src_dims[0])/float(src_dims[1]);
            
            size_t idx = 0;
            float best_ratio1 = float(d[0][0])/float(d[1][0]); assert( d[1][0]>0 );
            
            for ( size_t i=1; i<d[0].size(); ++i )
            {
                
                const float new_ratio = float(d[0][i])/float(d[1][i]); assert( d[0][i]>0 && d[1][i]>0 );
                
                if ( fabsf(new_ratio-ideal_ratio1)<fabsf(best_ratio1-ideal_ratio1) )
                {
                    idx = i;
                    best_ratio1 = new_ratio;
                }
            }
            
            float best_ratio2 = float(d[1][idx])/float(d[2][idx]); assert( d[1][0]>0 );
            const float ideal_ratio2 = float(src_dims[1])/float(src_dims[2]);
            
            
            for ( size_t i=1; i<d[0].size(); ++i )
            {
                const float new_ratio = float(d[0][i])/float(d[1][i]); assert( d[0][i]>0 && d[1][i]>0 );
                if ( fabsf(new_ratio/best_ratio1-1.f)<1.E-5 )
                {
                    const float new_ratio2 = float(d[1][i])/float(d[2][i]);
                    if ( fabs(new_ratio2-ideal_ratio2)<fabs(best_ratio2-ideal_ratio2) )
                    {
                        idx = i;
                        best_ratio2 = new_ratio2;
                    }
                }
            }
            
            dst_dims[0] = d[0][idx];
            dst_dims[1] = d[1][idx];
            dst_dims[2] = d[2][idx];
            
        }
        
#if 0
        static void unit_test()
        {
            const size_t num_procs = 8;
            //std::vector<unsigned int> d[3];
            
            //find_dims( num_procs, d );
            
            const unsigned int src_dims[3] = {258,258,258};
            unsigned  int dst_dims[3];
            
            find_best_match( num_procs, src_dims, dst_dims);
            std::cout << " " << dst_dims[0] << " " << dst_dims[1] << " " << dst_dims[2] << std::endl;
        }
#endif
        
    }; // end namespace ProcessLayout
    

    
    
    
    
    
};

#endif

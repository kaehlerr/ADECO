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


#ifndef Ad_THREAD_INTERFACE_
#define Ad_THREAD_INTERFACE_


#include <pthread.h>
#include <string>

#include "AdException.h"


namespace AdaptiveMassDeposit
{

class AdThreadInterface
{

 public:

  AdThreadInterface( const size_t id ) : id_(id)
  { 
  }
  

  virtual ~AdThreadInterface()
  { 
  }


  void create( )	
  {
    if ( pthread_create( &handle_, NULL, thread_func, static_cast<void*>(this) ) != 0 )
    {
        throw AdRuntimeException( std::string( "ERROR: AdThreadInterface(): failed to create thread with id==%lu.\n", id_) );
    }
  }


  void run( ) 
  {
   
      AD_VERBOSE( 4, { std::cout << "INFO: AdThreadInterface(): starting thread with id: " << id_ << std::endl; } );
      
      try
      {
          run_();
      }
      catch ( std::exception& ex )
      {
          std::cerr << "ERROR: AdThreadInterface::run(): Caught exception. " << std::endl;
          std::cerr << ex.what() << std::endl;
      }
      
      AD_VERBOSE( 4, { std::cout << "INFO: AdThreadInterface(): finished thread with id: " << id_ << std::endl; } );
  
  }
 

  void finish()
  {
      if ( pthread_join(handle_, NULL)!=0 )
      {
          throw AdRuntimeException( std::string( "ERROR: finish(): failed to join thread." ) );
      }
  }

    
  static void* thread_func( void* thread_class  )
  {

    AdThreadInterface* thread_ptr = static_cast<AdThreadInterface*>( thread_class );
 
   
    if ( thread_ptr==0 ) 
    {
      throw AdRuntimeException( std::string( "ERROR: thread_func(): failed to cast argument." ));
    }
    
    thread_ptr->run( );
    
    // we have to return something ...
    return 0;

  }


protected:

  virtual void run_() = 0;


 protected:

  size_t id_;
  pthread_t handle_;
		
	
    
};


    
};


#endif



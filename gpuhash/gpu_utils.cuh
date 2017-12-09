#pragma once

#ifndef __GPU_UTILS_H__
#define __GPU_UTILS_H__

#include <stdlib.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <fstream>


#if CUDART_VERSION >= 4000
#define CUT_DEVICE_SYNCHRONIZE( )   cudaDeviceSynchronize();
#else
#define CUT_DEVICE_SYNCHRONIZE( )   cudaThreadSynchronize();
#endif


#       define FPRINTF(a) fprintf a 

#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);


#ifdef _DEBUG
#  define CUT_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    err = CUT_DEVICE_SYNCHRONIZE();                                           \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }
#else
#  define CUT_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }
#endif


#define cutilCheckMsg(msg)           __cutilGetLastError (msg, __FILE__, __LINE__)
inline void __cutilGetLastError( const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        FPRINTF((stderr, "%s(%i) : cutilCheckMsg() CUTIL CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString( err ) ));
        exit(-1);
    }
}

//====================================================
//Portable data structure
//====================================================
typedef struct
{
	int *stream;
	int *event;
	int *start;
} gcStream_t;

//-----------------------------------------------------
//Stream Management
//-----------------------------------------------------
void gc_stream_start(gcStream_t* stream);
void gc_stream_stop(gcStream_t* stream);
void gc_stream_wait(gcStream_t* stream);

//-----------------------------------------------------
//Memory Management
//-----------------------------------------------------
void* gc_malloc(size_t bufsize);
void gc_free(void* gpu_buf);
void* gc_host2device(gcStream_t stream, void* cpu_buf, size_t bufsize);
void* gc_device2host(gcStream_t stream, void* gpu_buf, size_t bufsize);
int byte_num(int max_num);

//====================================================
//CUDA macros
//====================================================
#define THREAD_CONF(grid, block, gridBound, blockBound) do {\
	    block.x = blockBound;\
	    grid.x = gridBound; \
		if (grid.x > 65535) {\
		   grid.x = (int)sqrt((double)grid.x);\
		   grid.y = CEIL(gridBound, grid.x); \
				}\
	}while (0)
#define BLOCK_ID ((gridDim.y * blockIdx.x) + blockIdx.y)
#define THREAD_ID (threadIdx.x)
#define TID ((BLOCK_ID * blockDim.x) + THREAD_ID)
#define THREAD_COUNT (blockDim.x * gridDim.x * gridDim.y)
#define CEIL(n, d) ((n+d-1)/d)

#define ONE_BYTE 0xff
#define TWO_BYTE 0xffff
#define THREE_BYTE 0xffffff
#define FOUR_BYTE 0xffffffff

#endif // __CUTIL_H__


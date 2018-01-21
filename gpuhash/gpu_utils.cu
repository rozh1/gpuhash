/*
* Copyright 2017-2018 Roman Klassen
*
* Licensed under the Apache License, Version 2.0 (the "License"); you may not
* use this file except in compliance with the License. You may obtain a copy
* of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*
*/

#include "gpu_utils.cuh"
#include "cuda_runtime.h"

void* gc_malloc(size_t bufsize)
{
	void* gpu_buf = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_buf, bufsize));
	return gpu_buf;
}

void gc_free(void* gpu_buf)
{
	CUDA_SAFE_CALL(cudaFree(gpu_buf));
}

void* gc_host2device(gcStream_t Stream, void* cpu_buf, size_t bufsize)
{
	void* gpu_buf = NULL;

	unsigned int round_bufsize = CEIL(bufsize, 16) * 16 + 4;

	CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_buf, round_bufsize));
	CUDA_SAFE_CALL(cudaMemcpy(gpu_buf, cpu_buf, bufsize, cudaMemcpyHostToDevice));

	return gpu_buf;
}

void* gc_device2host(gcStream_t Stream, void* gpu_buf, size_t bufsize)
{
	void* pinned = NULL;
	CUDA_SAFE_CALL(cudaMallocHost((void**)&pinned, bufsize));
	CUDA_SAFE_CALL(cudaMemcpyAsync(pinned, gpu_buf, bufsize, cudaMemcpyDeviceToHost, (CUstream_st*)Stream.stream));

	return pinned;
}

int align_size(int size, int align_by)
{
	int rest = size%align_by;
	if (rest == 0) return size;
	return size + rest;
}


void gc_stream_start(gcStream_t* Stream)
{
	CUDA_SAFE_CALL(cudaStreamCreate((cudaStream_t*)&Stream->stream));
	CUDA_SAFE_CALL(cudaEventCreate((cudaEvent_t*)&Stream->event));
	CUDA_SAFE_CALL(cudaEventCreate((cudaEvent_t*)&Stream->start));
	CUDA_SAFE_CALL(cudaEventRecord((cudaEvent_t)Stream->start, (cudaStream_t)Stream->stream));
}

void gc_stream_stop(gcStream_t* Stream)
{
	CUDA_SAFE_CALL(cudaEventRecord((cudaEvent_t)Stream->event, (cudaStream_t)Stream->stream));
	CUDA_SAFE_CALL(cudaEventSynchronize((cudaEvent_t)Stream->event));

	float etime = 0.0f;
	cudaEventElapsedTime(&etime, (cudaEvent_t)Stream->start, (cudaEvent_t)Stream->event);
	printf("***%f ms\n", etime);
	CUDA_SAFE_CALL(cudaEventDestroy((cudaEvent_t)Stream->event));
	CUDA_SAFE_CALL(cudaEventDestroy((cudaEvent_t)Stream->start));
	CUDA_SAFE_CALL(cudaStreamDestroy((cudaStream_t)Stream->stream));
}

void gc_stream_wait(gcStream_t* Stream)
{
	CUDA_SAFE_CALL(cudaStreamSynchronize((cudaStream_t)Stream->stream));
}
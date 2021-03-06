/*
* Copyright 2021 Roman Klassen
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

#include "gpuhash.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "gpu_utils.cuh"
#include "hash_algs.cuh"

#define TIMETRACE

#ifdef TIMETRACE
#include "TimeInMs.h"
#endif


struct CharLine
{
	unsigned int start;
	unsigned int end;
};

__global__
void parseKernel(char *data, unsigned int size, CharLine *lines, unsigned int * minPositions, unsigned int * linesCountPerChunk)
{
	const unsigned int index = TID;
	const unsigned int numThreads = THREAD_COUNT;
	unsigned int lineCount = 0;
	unsigned int chunkStop = index == numThreads-1 ? size : minPositions[index + 1];
	unsigned int chunkStart = index == 0 ? 0 : minPositions[index];

	if (chunkStart >= size) return;

	unsigned int startIndex = 0;
	if (index != 0)
	{
		for (int i = 0; i < index; i++)
		{
			startIndex += linesCountPerChunk[i];
		}
		startIndex++;
	}
	unsigned int lastEnd = chunkStart;
	for (unsigned int i = chunkStart; i < chunkStop; i++)
	{
		if (data[i] == '\n' || i == chunkStop - 1)
		{
			int lineLen = i - lastEnd + (i > 0 && data[i - 1] == '\r'
				? -1
				: (i == chunkStop - 1 && !(data[i] == '\r' || data[i] == '\n') ? 1 : 0));
			if (lineLen <= 0)
			{
				lastEnd = i + 1;
				continue;
			}
			lines[lineCount + startIndex].end = lastEnd + lineLen;
			lines[lineCount + startIndex].start = lastEnd;
			lineCount++;
			lastEnd = i + 1;
		}
	}
}

__global__
void CountLinesKernel(char *data, unsigned int size, unsigned int * linesCount, unsigned int * minPositions)
{
	const unsigned int index = TID;
	const unsigned int numThreads = THREAD_COUNT;
	unsigned int count = 0;
	unsigned int minPosition = size;
	unsigned int chunk = (size + numThreads - 1) / numThreads;
	unsigned int start = index*chunk;
	unsigned int end = (index + 1)*chunk;

	for (unsigned int i = start; i < end  && i < size; i++)
	{
		if (data[i] == '\n') {
			count++;
			if (i < minPosition) minPosition = i;
		}
	}

	linesCount[index] = count;
	minPositions[index] = minPosition;
}

__global__
void parseAndHashKeysKernel(char *data, unsigned int size, CharLine *lines, unsigned int linesSize, unsigned int *keyCols, unsigned int keyColsSize, unsigned int nodeCount, unsigned int *hash, int algo)
{
	const unsigned int index = TID;
	const unsigned int numThreads = THREAD_COUNT;

	if (index >= linesSize) return;

	unsigned int maxEnter = 0;
	int hashSum = 0;
	for (int k = 0; k < keyColsSize; k++)
	{
		if (keyCols[k] > maxEnter)
		{
			maxEnter = keyCols[k];
		}
	}

	int* keys = new int[keyColsSize];

	for (unsigned int i = index; i < linesSize; i += numThreads)
	{
		unsigned int lineStart = lines[i].start;
		unsigned int lineEnd = lines[i].end;
		unsigned int enter = lineStart;
		unsigned int enterCount = 0;
		for (unsigned int j = lineStart; j < lineEnd; j++)
		{
			if (data[j] == '|' || j == lineEnd - 1)
			{
				unsigned int len = j - enter + (j == lineEnd - 1 ? 1 : 0);
				for (int k = 0; k < keyColsSize; k++)
				{
					if (keyCols[k] == enterCount)
					{
						int key = 0;
						int numberLen = enter + len;
						for (int l = enter; l < numberLen; l++)
						{
							if (data[l] == '\"') continue;
							if (data[l] == '-')
							{
								key *= -1; continue;
							}
							if (data[l] >= '0' && data[l] <= '9')
								key = key * 10 + (data[l] - 48);
						}
						keys[k] = key;
					}
				}
				enter = j + 1;
				enterCount++;
			}
			if (enterCount > maxEnter) break;
		}
		char buf[4*4];
		switch (algo)
		{
			case 2:
				for (int k = 0; k < keyColsSize && k < 8; k++)
				{
					for (int i = 0; i < 4; i++)
						buf[k*4 + 3 - i] = (keys[k] >> (i * 8));
				}
				hashSum = CRC32(buf, keyColsSize*4);
				break;
			case 4:
				for (int k = 0; k < keyColsSize && k < 8; k++)
				{
					for (int i = 0; i < 4; i++)
						buf[k*4 + 3 - i] = (keys[k] >> (i * 8));
				}
				hashSum = MurMurHash(buf, keyColsSize*4);
				break;
			case 0:
			default:
				for (int k = 0; k < keyColsSize; k++)
				{
					hashSum += keys[k] % nodeCount;
				}
				break;
		}
		hash[i] = hashSum % nodeCount;
		hashSum = 0;
	}
}

__global__
void parseBytesAndHashKeysKernel(char *data, unsigned int size, CharLine *lines, unsigned int linesSize, unsigned int *keyCols, unsigned int keyColsSize, unsigned int nodeCount, unsigned int *hash, int algo)
{
	const unsigned int index = TID;
	const unsigned int numThreads = THREAD_COUNT;

	if (index >= linesSize) return;

	unsigned int maxEnter = 0;
	for (int k = 0; k < keyColsSize; k++)
	{
		if (keyCols[k] > maxEnter)
		{
			maxEnter = keyCols[k];
		}
	}

	int* keys = new int[keyColsSize];
	
	char buf[256];
	for (unsigned int i = index; i < linesSize; i += numThreads)
	{
		unsigned int lineStart = lines[i].start;
		unsigned int lineEnd = lines[i].end;
		unsigned int enter = lineStart;
		unsigned int enterCount = 0;
		unsigned int bufIndex = 0;
		for (unsigned int j = lineStart; j < lineEnd; j++)
		{
			if (data[j] == '|' || j == lineEnd - 1)
			{
				unsigned int len = j - enter + (j == lineEnd - 1 ? 1 : 0);
				for (int k = 0; k < keyColsSize; k++)
				{
					if (keyCols[k] == enterCount)
					{
						int key = 0;
						int numberLen = enter + len;
						for (int l = enter; l < numberLen; l++)
						{
							if (data[l] == '\"') continue;
							buf[bufIndex++] = data[l];
							j++;
						}
						keys[k] = key;
					}
				}
				enter = j + 1;
				enterCount++;
			}
			if (enterCount > maxEnter) break;
		}
		switch (algo)
		{
			case 1:
				hash[i] = CRC32(buf, bufIndex) % nodeCount;
				break;
			case 3:
				hash[i] = MurMurHash(buf, bufIndex) % nodeCount;
				break;
		}
	}
}


void CGpuHash::hashDataWithCuda(char *data, unsigned int size, unsigned int *keyCols, unsigned int keyColsSize, int nodeCount, HashedBlock **hashedBlocks, unsigned int *lenght, int algo)
{
	unsigned int *dev_lineCounts;
	unsigned int *dev_minPositions;
	unsigned int gpuThreadCount;

#ifdef TIMETRACE
	uint64 start_time = GetTimeMs64();
#endif
	dim3 blockDim(THREADS_PER_BLOCK, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, BLOCK_NUMBER, THREADS_PER_BLOCK);
	gpuThreadCount = THREAD_COUNT;
	printf("gpuThreadCount:  %d \n", gpuThreadCount);

	CUDA_SAFE_CALL(cudaSetDevice(CGpuHash::GpuNumber));
	CUDA_SAFE_CALL(cudaDeviceReset());

	gcStream_t s;
	gc_stream_start(&s);
	
	dev_lineCounts = (unsigned int *)gc_malloc(gpuThreadCount * sizeof(unsigned int));
	dev_minPositions = (unsigned int *)gc_malloc(gpuThreadCount * sizeof(unsigned int));
	auto dev_data = (char *)gc_host2device(s, data, size * sizeof(char));
		
	CountLinesKernel << <gridDim, blockDim, 0, (cudaStream_t)s.stream >> >(dev_data, size, dev_lineCounts, dev_minPositions);
	CUT_CHECK_ERROR("CountLinesKernel");
	
	unsigned int * host_lineCounts = (unsigned int *)gc_device2host(s, dev_lineCounts, gpuThreadCount * sizeof(unsigned int));
	gc_stream_wait(&s);
	
	unsigned linesSize = 0;
	for (unsigned int i = 0; i < gpuThreadCount; i++)
	{
		linesSize += host_lineCounts[i];
	}
	if (linesSize < gpuThreadCount / 2)
	{
		unsigned int * host_minPositions = (unsigned int *)gc_device2host(s, dev_minPositions, gpuThreadCount * sizeof(unsigned int));
		gc_stream_wait(&s);
	
		for (unsigned int i = 0, j = 0; i < gpuThreadCount; i++)
		{
			if (host_lineCounts[i] != 0)
			{
				host_lineCounts[j] = host_lineCounts[i];
				host_minPositions[j] = host_minPositions[i];
				host_lineCounts[i] = 0;
				host_minPositions[i] = size;
				j++;
			}
		}
		CUDA_SAFE_CALL(cudaMemcpy(dev_minPositions, host_minPositions, gpuThreadCount * sizeof(unsigned int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(dev_lineCounts, host_lineCounts, gpuThreadCount * sizeof(unsigned int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaFreeHost(host_minPositions));
	}
	CUDA_SAFE_CALL(cudaFreeHost(host_lineCounts));

	int *dev_keys = (int *)gc_malloc(linesSize * sizeof(int));
	unsigned int *dev_hash = (unsigned int *)gc_malloc(linesSize * sizeof(unsigned int));
	CharLine *dev_CharLines = (CharLine *)gc_malloc(linesSize * sizeof(CharLine));
	unsigned int *dev_keyCols = (unsigned int *)gc_host2device(s, keyCols, keyColsSize * sizeof(unsigned int));
			
	parseKernel <<<gridDim, blockDim, 0, (cudaStream_t)s.stream>>>(dev_data, size, dev_CharLines, dev_minPositions, dev_lineCounts);
	CUT_CHECK_ERROR("parseKernel");
	
	printf("Algo:  %d \n", algo);
	switch (algo)
	{
		case 0:
			printf("start kernel: parseAndHashKeysKernel for MOD HASH \n");
			parseAndHashKeysKernel <<<gridDim, blockDim, 0, (cudaStream_t)s.stream>>>(dev_data, size, dev_CharLines, linesSize, dev_keyCols, keyColsSize, nodeCount, dev_hash, algo);
			CUT_CHECK_ERROR("parseAndHashKeysKernel");
		break;
		case 1:
			printf("start kernel: parseBytesAndHashKeysKernel for CRC HASH \n");
			parseBytesAndHashKeysKernel<<<gridDim, blockDim, 0, (cudaStream_t)s.stream>>>(dev_data, size, dev_CharLines, linesSize, dev_keyCols, keyColsSize, nodeCount, dev_hash, algo);
			CUT_CHECK_ERROR("parseBytesAndHashKeysKernel");
		break;
		case 2:
			printf("start kernel: parseAndHashKeysKernel for INT CRC HASH \n");
			parseAndHashKeysKernel<<<gridDim, blockDim, 0, (cudaStream_t)s.stream>>>(dev_data, size, dev_CharLines, linesSize, dev_keyCols, keyColsSize, nodeCount, dev_hash, algo);
			CUT_CHECK_ERROR("parseAndHashKeysKernel");
		break;
		case 3:
			printf("start kernel: parseBytesAndHashKeysKernel for MurMur HASH \n");
			parseBytesAndHashKeysKernel<<<gridDim, blockDim, 0, (cudaStream_t)s.stream>>>(dev_data, size, dev_CharLines, linesSize, dev_keyCols, keyColsSize, nodeCount, dev_hash, algo);
			CUT_CHECK_ERROR("parseBytesAndHashKeysKernel");
		break;
		case 4:
			printf("start kernel: parseAndHashKeysKernel for INT MurMur HASH \n");
			parseAndHashKeysKernel<<<gridDim, blockDim, 0, (cudaStream_t)s.stream>>>(dev_data, size, dev_CharLines, linesSize, dev_keyCols, keyColsSize, nodeCount, dev_hash, algo);
			CUT_CHECK_ERROR("parseAndHashKeysKernel");
		break;
	}
		
	unsigned int * host_hash = (unsigned int *)gc_device2host(s, dev_hash, linesSize * sizeof(unsigned int));
	CharLine * host_CharLines = (CharLine *)gc_device2host(s, dev_CharLines, linesSize * sizeof(CharLine));

	gc_stream_wait(&s);
	gc_stream_stop(&s);

	gc_free(dev_data);
	gc_free(dev_keyCols);
	gc_free(dev_CharLines);
	gc_free(dev_keys);
	gc_free(dev_lineCounts);
	gc_free(dev_hash);
	gc_free(dev_minPositions);
	
#ifdef TIMETRACE
	printf("GPU:  %I64d ms \n", (GetTimeMs64() - start_time));
	start_time = GetTimeMs64();
#endif
	HashedBlock* hashedBlock = new HashedBlock[nodeCount];
	int* startIndexes = new int[nodeCount];

	for (unsigned int i = 0; i < nodeCount; i++)
	{
		hashedBlock[i].lenght = 0;
		startIndexes[i] = 0;
	}

	for (unsigned int i = 0; i < linesSize; i++)
	{
		hashedBlock[host_hash[i]].hash = host_hash[i];
		hashedBlock[host_hash[i]].lenght += host_CharLines[i].end - host_CharLines[i].start + 1;
	}
	for (unsigned int i = 0; i < nodeCount; i++)
	{
		hashedBlock[i].line = new char[hashedBlock[i].lenght];
	}
	for (unsigned int i = 0; i < linesSize; i++)
	{
		auto nodeIndex = host_hash[i];
		unsigned int k = startIndexes[nodeIndex];
		for (unsigned int j = host_CharLines[i].start; j < host_CharLines[i].end; j++, k++)
		{
			hashedBlock[nodeIndex].line[k] = data[j];
		}
		hashedBlock[nodeIndex].line[k] = '\n';
		startIndexes[nodeIndex] += host_CharLines[i].end - host_CharLines[i].start + 1;
	}
	*hashedBlocks = hashedBlock;
	*lenght = nodeCount;
#ifdef TIMETRACE
	printf("SPLIT:  %I64d ms \n", (GetTimeMs64() - start_time));
#endif
	CUDA_SAFE_CALL(cudaFreeHost(host_CharLines));
	CUDA_SAFE_CALL(cudaFreeHost(host_hash));
}



void CGpuHash::hashDataCuda(char* data, unsigned size, unsigned* keyCols, unsigned keyColsSize, int nodeCount, HashedBlock** hashedBlock, unsigned* lenght)
{
	hashDataWithCuda(data, size, keyCols, keyColsSize, nodeCount, hashedBlock, lenght, 0);
}

void CGpuHash::hashCrcCuda(char* data, unsigned size, unsigned* keyCols, unsigned keyColsSize, int nodeCount, HashedBlock** hashedBlock, unsigned* lenght)
{
	hashDataWithCuda(data, size, keyCols, keyColsSize, nodeCount, hashedBlock, lenght, 1);
}

void CGpuHash::hashIntCrcCuda(char* data, unsigned size, unsigned* keyCols, unsigned keyColsSize, int nodeCount, HashedBlock** hashedBlock, unsigned* lenght)
{
	hashDataWithCuda(data, size, keyCols, keyColsSize, nodeCount, hashedBlock, lenght, 2);
}

void CGpuHash::hashMurMurCuda(char* data, unsigned size, unsigned* keyCols, unsigned keyColsSize, int nodeCount, HashedBlock** hashedBlock, unsigned* lenght)
{
	hashDataWithCuda(data, size, keyCols, keyColsSize, nodeCount, hashedBlock, lenght, 3);
}

void CGpuHash::hashIntMurMurCuda(char* data, unsigned size, unsigned* keyCols, unsigned keyColsSize, int nodeCount, HashedBlock** hashedBlock, unsigned* lenght)
{
	hashDataWithCuda(data, size, keyCols, keyColsSize, nodeCount, hashedBlock, lenght, 4);
}
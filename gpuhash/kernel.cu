
#include "gpuhash.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <istream>
#include <fstream>
#include <iostream>
#include "TimeInMs.h"
#include "gpu_utils.cuh"


__global__ 
void hashKernel(int *keys, unsigned int size, unsigned int nodeCount, unsigned int *hash)
{
	const unsigned int index = TID;
	const unsigned int numThreads = THREAD_COUNT;

	for (unsigned int i = index; i < size; i += numThreads)
	{
		hash[i] = keys[i] % nodeCount;
	}
}

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
void parseKeysKernel(char *data, unsigned int size, CharLine *lines, unsigned int linesSize, unsigned int *keyCols, unsigned int keyColsSize, int *keys)
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
						keys[i] = key;
					}
				}
				enter = j + 1;
				enterCount++;
			}
			if (enterCount > maxEnter) break;
		}
	}
}


void HashData(char *data, unsigned int size, unsigned int *keyCols, unsigned int keyColsSize, int nodeCount, HashedBlock ** hashed_blocks, unsigned int *lenght)
{
	hashDataCuda(data, size, keyCols, keyColsSize, nodeCount, hashed_blocks, lenght);
}

static void appendLineToFile(std::string filepath, std::string line)
{
	std::ofstream file;
	//can't enable exception now because of gcc bug that raises ios_base::failure with useless message
	//file.exceptions(file.exceptions() | std::ios::failbit);
	file.open(filepath, std::ios::out | std::ios::app);
	if (file.fail())
		throw std::ios_base::failure(std::strerror(errno));

	//make sure write fails with exception if something is wrong
	file.exceptions(file.exceptions() | std::ios::failbit | std::ifstream::badbit);

	file << line.c_str() << std::endl;
}

int main()
{
	unsigned int size = 0;
	char * memblock;

	std::ifstream file("orders.tbl", std::ios::in | std::ios::binary | std::ios::ate);
	if (file.is_open())
	{
		size = file.tellg();
		memblock = new char[size];
		file.seekg(0, std::ios::beg);
		file.read(memblock, size);
		file.close();

		std::cout << "the entire file content is in memory";

		unsigned int* keys = new unsigned[1];
		keys[0] = 0;

		uint64 start_time = GetTimeMs64();
		HashedBlock *data = nullptr;
		unsigned int dataLen = 0;

		HashData(memblock, size, keys, 1, 4, &data, &dataLen);
		delete memblock;
		std::cout << GetTimeMs64() - start_time << " ms" << std::endl;

		for (int i = 0; i < dataLen; i++)
		{
			auto filename = new char[20];
			sprintf_s(filename, 20, "orders_%d.tbl\0", data[i].hash);
			auto str = std::string(data[i].line, data[i].lenght);
			auto name = new std::string(filename);
			appendLineToFile(filename, str);
		}

	}
	else std::cout << "Unable to open file";
	

    return 0;
}

void hashDataCuda(char *data, unsigned int size, unsigned int *keyCols, unsigned int keyColsSize, int nodeCount, HashedBlock **hashedBlocks, unsigned int *lenght)
{
	unsigned int *dev_lineCounts;
	unsigned int *dev_minPositions;
	unsigned int gpuThreadCount;

	uint64 start_time = GetTimeMs64();

	dim3 blockDim(256, 1, 1);
	dim3 gridDim(1, 1, 1);
	THREAD_CONF(gridDim, blockDim, BLOCK_NUMBER, THREADS_PER_BLOCK);
	gpuThreadCount = THREAD_COUNT;

	CUDA_SAFE_CALL(cudaSetDevice(0));

	gcStream_t s;
	gc_stream_start(&s);
	
	dev_lineCounts = (unsigned int *)gc_malloc(gpuThreadCount * sizeof(unsigned int));
	dev_minPositions = (unsigned int *)gc_malloc(gpuThreadCount * sizeof(unsigned int));
	char *dev_data = (char *)gc_host2device(s, data, size * sizeof(char));
		
	CountLinesKernel << <gridDim, blockDim, 0, (cudaStream_t)s.stream >> >(dev_data, size, dev_lineCounts, dev_minPositions);
	CUT_CHECK_ERROR("CountLinesKernel");
	
	unsigned int * host_lineCounts = (unsigned int *)gc_device2host(s, dev_lineCounts, gpuThreadCount * sizeof(unsigned int));
	gc_stream_wait(&s);
	
	unsigned linesSize = 1;
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
		free(host_minPositions);
	}
	cudaFreeHost(host_lineCounts);

	int *dev_keys = (int *)gc_malloc(linesSize * sizeof(int));
	unsigned int *dev_hash = (unsigned int *)gc_malloc(linesSize * sizeof(unsigned int));
	CharLine *dev_CharLines = (CharLine *)gc_malloc(linesSize * sizeof(CharLine));
	unsigned int *dev_keyCols = (unsigned int *)gc_host2device(s, keyCols, keyColsSize * sizeof(unsigned int));
			
	parseKernel << <gridDim, blockDim, 0, (cudaStream_t)s.stream >> >(dev_data, size, dev_CharLines, dev_minPositions, dev_lineCounts);
	CUT_CHECK_ERROR("parseKernel");
		
	parseKeysKernel << <gridDim, blockDim, 0, (cudaStream_t)s.stream >> >(dev_data, size, dev_CharLines, linesSize, dev_keyCols, keyColsSize, dev_keys);
	CUT_CHECK_ERROR("parseKernel");
	
	hashKernel << <gridDim, blockDim, 0, (cudaStream_t)s.stream >> >(dev_keys, linesSize, nodeCount, dev_hash);
	CUT_CHECK_ERROR("hashKernel");
	
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

	std::cout << "GPU: " << GetTimeMs64() - start_time << " ms" << std::endl;

	start_time = GetTimeMs64();
	HashedBlock* hashedBlock = new HashedBlock[linesSize];
	for (unsigned int i = 0; i < linesSize; i++)
	{
		hashedBlock[i].hash = host_hash[i];
		hashedBlock[i].lenght = host_CharLines[i].end - host_CharLines[i].start;
		hashedBlock[i].line = new char[hashedBlock[i].lenght];
		for (unsigned int j = host_CharLines[i].start, k = 0; j < host_CharLines[i].end; j++, k++)
		{
			hashedBlock[i].line[k] = data[j];
		}
	}
	*hashedBlocks = hashedBlock;
	*lenght = linesSize;
	std::cout << "SPLIT: " << GetTimeMs64() - start_time << " ms" << std::endl;

	cudaFreeHost(host_CharLines);
	cudaFreeHost(host_hash);
}
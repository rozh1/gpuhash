
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <istream>
#include <fstream>
#include <iostream>
#include "TimeInMs.h"

#define THREADS_PER_BLOCK 24
#define BLOCK_NUMBER 12

struct CharLine
{
	unsigned int start;
	unsigned int end;
};

struct HashedBlock
{
	char *line;
	int lenght;
	int hash;
};

cudaError_t hashDataCuda(char *data, unsigned int size, unsigned int *keyCols, unsigned int keyColsSize, int nodeCount, HashedBlock **hashedBlock);

__global__ 
void hashKernel(int *keys, unsigned int size, unsigned int nodeCount, unsigned int *hash)
{
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int numThreads = blockDim.x * gridDim.x;

	for (unsigned int i = index; i < size; i += numThreads)
	{
		hash[i] = keys[i] % nodeCount;
	}
}

__global__
void parseKernel(char *data, unsigned int size, CharLine *lines, unsigned int * minPositions, unsigned int * linesCountPerChunk)
{
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int numThreads = blockDim.x * gridDim.x;
	unsigned int lineCount = 0;
	unsigned int chunkStop = index == numThreads ? size : minPositions[index + 1];
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
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int numThreads = blockDim.x * gridDim.x;
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
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int numThreads = blockDim.x * gridDim.x;

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


HashedBlock *HashData(char *data, unsigned int size, unsigned int *keyCols, unsigned int keyColsSize, int nodeCount)
{
	HashedBlock *hashedBlocks = nullptr;
	cudaError_t cudaStatus = hashDataCuda(data, size, keyCols, keyColsSize, nodeCount, &hashedBlocks);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "hashDataCuda failed!");
	return  new HashedBlock();
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	return  new HashedBlock();
	}
	return hashedBlocks;
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

	std::ifstream file("region.tbl", std::ios::in | std::ios::binary | std::ios::ate);
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
		auto data = HashData(memblock, size, keys, 1, 4);
		delete memblock;
		std::cout << GetTimeMs64() - start_time << " ms" << std::endl;

		for (int i = 0; i < 10001; i++)
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

cudaError_t hashDataCuda(char *data, unsigned int size, unsigned int *keyCols, unsigned int keyColsSize, int nodeCount, HashedBlock **hashedBlocks)
{
	char *dev_data = 0;
	unsigned int *dev_keyCols = 0;
	CharLine *dev_CharLines = 0;
	int *dev_keys = 0;
	unsigned int *dev_lineCounts = 0;
	unsigned int *dev_hash = 0;
	unsigned int *dev_minPositions = 0;
	unsigned int *host_hash = 0;
	CharLine *host_CharLines = 0;
	unsigned int gpuThreadCount = BLOCK_NUMBER * THREADS_PER_BLOCK;
	cudaError_t cudaStatus;

	uint64 start_time = GetTimeMs64();

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_lineCounts, gpuThreadCount * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_minPositions, gpuThreadCount * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_data, size * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMemcpy(dev_data, data, size * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	
	CountLinesKernel << <BLOCK_NUMBER, THREADS_PER_BLOCK >> >(dev_data, size, dev_lineCounts, dev_minPositions);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "parseKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching parseKernel!\n", cudaStatus);
		goto Error;
	}

	unsigned int * host_lineCounts = new unsigned[gpuThreadCount];
	cudaStatus = cudaMemcpy(host_lineCounts, dev_lineCounts, gpuThreadCount * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	unsigned linesSize = 0;
	for (unsigned int i = 0; i < gpuThreadCount; i++)
	{
		linesSize += host_lineCounts[i];
	}
	if (linesSize < gpuThreadCount / 2)
	{
		unsigned int * host_minPositions = new unsigned[gpuThreadCount];
		cudaStatus = cudaMemcpy(host_minPositions, dev_minPositions, gpuThreadCount * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

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

		cudaStatus = cudaMemcpy(dev_minPositions, host_minPositions, gpuThreadCount * sizeof(unsigned int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		cudaStatus = cudaMemcpy(dev_lineCounts, host_lineCounts, gpuThreadCount * sizeof(unsigned int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	}


	cudaStatus = cudaMalloc((void**)&dev_keys, linesSize * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_hash, linesSize * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_keyCols, keyColsSize * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_CharLines, linesSize * sizeof(CharLine));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_keyCols, keyCols, keyColsSize * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
		
	parseKernel << <BLOCK_NUMBER, THREADS_PER_BLOCK >> >(dev_data, size, dev_CharLines, dev_minPositions, dev_lineCounts);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "parseKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching parseKernel!\n", cudaStatus);
		goto Error;
	}
	
	parseKeysKernel << <BLOCK_NUMBER, THREADS_PER_BLOCK >> >(dev_data, size, dev_CharLines, linesSize, dev_keyCols, keyColsSize, dev_keys);

	auto host_keys = new int[linesSize];

	cudaStatus = cudaMemcpy(host_keys, dev_keys, linesSize * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "parseKeysKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching parseKeysKernel!\n", cudaStatus);
		goto Error;
	}

	hashKernel << <BLOCK_NUMBER, THREADS_PER_BLOCK >> >(dev_keys, linesSize, nodeCount, dev_hash);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "hashKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching hashKernel!\n", cudaStatus);
		goto Error;
	}
	

	host_hash = new unsigned[linesSize];
	cudaStatus = cudaMemcpy(host_hash, dev_hash, linesSize * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	host_CharLines = new CharLine[linesSize];
	cudaStatus = cudaMemcpy(host_CharLines, dev_CharLines, linesSize * sizeof(CharLine), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
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
	std::cout << "SPLIT: " << GetTimeMs64() - start_time << " ms" << std::endl;

	free(host_CharLines);
	free(host_hash);


Error:
	cudaFree(dev_data);
	cudaFree(dev_keyCols);
	cudaFree(dev_CharLines);
	cudaFree(dev_keys);
	cudaFree(dev_hash);

	return cudaStatus;
}
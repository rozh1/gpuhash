#include "gpuhash.h"
#include <mutex>

CGpuHash::CGpuHash(int gpuNumber)
{
	GpuNumber = gpuNumber;
	return;
}

std::mutex gpuMutex;
void CGpuHash::HashData(char* data, unsigned size, unsigned* keyCols, unsigned keyColsSize, int nodeCount, HashedBlock** hashedBlock, unsigned* lenght)
{
	//gpuMutex.lock();
	hashDataCuda(data, size, keyCols, keyColsSize, nodeCount, hashedBlock, lenght);
	//gpuMutex.unlock();
}

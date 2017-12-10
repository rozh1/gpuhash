#include "gpuhash.h"

CGpuHash::CGpuHash(int gpuNumber)
{
	GpuNumber = gpuNumber;
	return;
}

void CGpuHash::HashData(char* data, unsigned size, unsigned* keyCols, unsigned keyColsSize, int nodeCount, HashedBlock** hashedBlock, unsigned* lenght)
{
	hashDataCuda(data, size, keyCols, keyColsSize, nodeCount, hashedBlock, lenght);
}

#ifndef __GPUHASH_H__
#define __GPUHASH_H__

#include <stdio.h>

#ifdef GPUHASHDLL_EXPORTS
#define GPUHASHDLL_API __declspec(dllexport) 
#else
#define GPUHASHDLL_API __declspec(dllimport) 
#endif

#undef BEGIN_C_DECLS
#undef END_C_DECLS
#ifdef __cplusplus
# define BEGIN_C_DECLS extern "C" {
# define END_C_DECLS }
#else
# define BEGIN_C_DECLS /* empty */
# define END_C_DECLS /* empty */
#endif

#define THREADS_PER_BLOCK 256
#define BLOCK_NUMBER 8

struct HashedBlock
{
	char *line;
	int lenght;
	int hash;
};


class GPUHASHDLL_API CGpuHash {
public:
	CGpuHash(int gpuNumber);
	void HashData(char *data, unsigned int size, unsigned int *keyCols, unsigned int keyColsSize, int nodeCount, HashedBlock **hashedBlock, unsigned int *lenght);
	void hashDataCuda(char *data, unsigned int size, unsigned int *keyCols, unsigned int keyColsSize, int nodeCount, HashedBlock **hashedBlock, unsigned int *lenght);
	int GpuNumber;
};

//====================================================
//APIs
//====================================================
BEGIN_C_DECLS
inline
GPUHASHDLL_API void HashDataWithGPU(void* handle, char* data, unsigned int size, unsigned int* keyCols, unsigned int keyColsSize, int nodeCount, HashedBlock** hashedBlock, unsigned int* lenght)
{
	auto *selHandle = (CGpuHash*)handle;
    selHandle->HashData(data, size, keyCols, keyColsSize, nodeCount, hashedBlock, lenght);
}

inline
GPUHASHDLL_API void* INIT(int gpuNumber)
{
	return new CGpuHash(gpuNumber);
}

inline
GPUHASHDLL_API void DESTROY(void* handle)
{
	auto *curHandle = (CGpuHash*)handle;
	delete curHandle;
}


END_C_DECLS

#endif

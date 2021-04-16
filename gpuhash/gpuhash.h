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
	void hashDataWithCuda(char *data, unsigned int size, unsigned int *keyCols, unsigned int keyColsSize, int nodeCount, HashedBlock **hashedBlocks, unsigned int *lenght, int algo);
	void hashDataCuda(char *data, unsigned int size, unsigned int *keyCols, unsigned int keyColsSize, int nodeCount, HashedBlock **hashedBlock, unsigned int *lenght);
	void hashIntCrcCuda(char *data, unsigned int size, unsigned int *keyCols, unsigned int keyColsSize, int nodeCount, HashedBlock **hashedBlock, unsigned int *lenght);
	void hashCrcCuda(char *data, unsigned int size, unsigned int *keyCols, unsigned int keyColsSize, int nodeCount, HashedBlock **hashedBlock, unsigned int *lenght);
	void hashIntMurMurCuda(char *data, unsigned int size, unsigned int *keyCols, unsigned int keyColsSize, int nodeCount, HashedBlock **hashedBlock, unsigned int *lenght);
	void hashMurMurCuda(char *data, unsigned int size, unsigned int *keyCols, unsigned int keyColsSize, int nodeCount, HashedBlock **hashedBlock, unsigned int *lenght);
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
GPUHASHDLL_API void HashIntCRC(void* handle, char* data, unsigned int size, unsigned int* keyCols, unsigned int keyColsSize, int nodeCount, HashedBlock** hashedBlock, unsigned int* lenght)
{
	auto *selHandle = (CGpuHash*)handle;
    selHandle->hashIntCrcCuda(data, size, keyCols, keyColsSize, nodeCount, hashedBlock, lenght);
}

inline
GPUHASHDLL_API void HashCRC(void* handle, char* data, unsigned int size, unsigned int* keyCols, unsigned int keyColsSize, int nodeCount, HashedBlock** hashedBlock, unsigned int* lenght)
{
	auto *selHandle = (CGpuHash*)handle;
    selHandle->hashCrcCuda(data, size, keyCols, keyColsSize, nodeCount, hashedBlock, lenght);
}

inline
GPUHASHDLL_API void HashIntMurMur(void* handle, char* data, unsigned int size, unsigned int* keyCols, unsigned int keyColsSize, int nodeCount, HashedBlock** hashedBlock, unsigned int* lenght)
{
	auto *selHandle = (CGpuHash*)handle;
    selHandle->hashIntMurMurCuda(data, size, keyCols, keyColsSize, nodeCount, hashedBlock, lenght);
}

inline
GPUHASHDLL_API void HashMurMur(void* handle, char* data, unsigned int size, unsigned int* keyCols, unsigned int keyColsSize, int nodeCount, HashedBlock** hashedBlock, unsigned int* lenght)
{
	auto *selHandle = (CGpuHash*)handle;
    selHandle->hashMurMurCuda(data, size, keyCols, keyColsSize, nodeCount, hashedBlock, lenght);
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

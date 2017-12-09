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

//====================================================
//APIs
//====================================================
BEGIN_C_DECLS

GPUHASHDLL_API
void hashDataCuda(char *data, unsigned int size, unsigned int *keyCols, unsigned int keyColsSize, int nodeCount, HashedBlock **hashedBlock, unsigned int *lenght);

END_C_DECLS

#endif

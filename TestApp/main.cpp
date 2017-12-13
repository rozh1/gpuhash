
#include <iostream>
#include <fstream>
#include "TimeInMs.h"
#include "../gpuhash/gpuhash.h"


void HashData(char *data, unsigned int size, unsigned int *keyCols, unsigned int keyColsSize, int nodeCount, HashedBlock ** hashed_blocks, unsigned int *lenght)
{
	auto handle = INIT(0);
	HashDataWithGPU(handle, data, size, keyCols, keyColsSize, nodeCount, hashed_blocks, lenght);
	DESTROY(handle);
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

		std::cout << "the entire file content is in memory" << std::endl;

		unsigned int* keys = new unsigned[1];
		keys[0] = 0;

		uint64 start_time = GetTimeMs64();

		HashedBlock* data = nullptr;
		unsigned int dataLen = 0;
		for (int i = 0; i < 10; i++)
		{
			HashData(memblock, size, keys, 1, 4, &data, &dataLen);

			for (int j = 0; j < dataLen; j++)
			{
				auto block = data[j];
				free(block.line);
			}

			free(data);
			data = nullptr;
		}
		std::cout << GetTimeMs64() - start_time << " ms" << std::endl;

		//for (int i = 0; i < dataLen; i++)
		//{
		//	auto filename = new char[20];
		//	sprintf_s(filename, 50, "c6997abd-f7e7-4e55-a301-67a901af23eb_3_%d\0", data[i].hash);
		//	auto str = std::string(data[i].line, data[i].lenght);
		//	auto name = new std::string(filename);
		//	appendLineToFile(filename, str);
		//}

	}
	else std::cout << "Unable to open file";


	return 0;
}

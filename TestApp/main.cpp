
#include <iostream>
#include <fstream>
#include "../gpuhash/gpuhash.h"
#include "TimeInMs.h"


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

		std::cout << "the entire file content is in memory" << std::endl;

		unsigned int* keys = new unsigned[1];
		keys[0] = 0;

		uint64 start_time = GetTimeMs64();
		HashedBlock *data = nullptr;
		unsigned int dataLen = 0;

		HashData(memblock, size, keys, 1, 4, &data, &dataLen);
		delete memblock;
		std::cout << GetTimeMs64() - start_time << " ms" << std::endl;

		//for (int i = 0; i < dataLen; i++)
		//{
		//	auto filename = new char[20];
		//	sprintf_s(filename, 20, "orders_%d.tbl\0", data[i].hash);
		//	auto str = std::string(data[i].line, data[i].lenght);
		//	auto name = new std::string(filename);
		//	appendLineToFile(filename, str);
		//}

	}
	else std::cout << "Unable to open file";


	return 0;
}

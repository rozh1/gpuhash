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

#include <iostream>
#include <fstream>
#include "TimeInMs.h"
#include "../gpuhash/gpuhash.h"


void HashData(char *data, unsigned int size, unsigned int *keyCols, unsigned int keyColsSize, int nodeCount, HashedBlock ** hashed_blocks, unsigned int *lenght)
{
	auto handle = INIT(0);
	HashIntCRC(handle, data, size, keyCols, keyColsSize, nodeCount, hashed_blocks, lenght);
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

	file << line.c_str();
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
		/*for (int i = 0; i < 10; i++)
		{
			HashData(memblock, size, keys, 1, 4, &data, &dataLen);

			for (int j = 0; j < dataLen; j++)
			{
				auto block = data[j];
				free(block.line);
			}

			free(data);
			data = nullptr;
		}*/
		HashData(memblock, size, keys, 1, 4, &data, &dataLen);
		std::cout << GetTimeMs64() - start_time << " ms" << std::endl;

		for (int i = 0; i < dataLen; i++)
		{
			auto filename = new char[20];
			sprintf_s(filename, 20, "orders_%d\0", data[i].hash);
			auto str = std::string(data[i].line, data[i].lenght);
			auto name = new std::string(filename);
			appendLineToFile(filename, str);
		}

	}
	else std::cout << "Unable to open file";


	return 0;
}

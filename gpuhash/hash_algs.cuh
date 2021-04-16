/*
* Copyright 2021 Roman Klassen
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

#pragma once

#ifndef __GPU_HASH_ALGS_H__
#define __GPU_HASH_ALGS_H__


#include <stdlib.h>
#include <cuda_runtime.h>

__device__
unsigned int CRC32(char *buf, unsigned int len);

__device__
unsigned int MurMurHash(char *buf, unsigned int len);

__device__
unsigned int MurmurHash2 (char * buf, int len, unsigned int seed );

#endif // __GPU_HASH_ALGS_H__
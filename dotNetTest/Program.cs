#region Copyright
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
 #endregion
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.IO;

namespace dotNetTest
{
    class Program
    {
        [DllImport("gpuhash.dll", EntryPoint = "HashDataWithGPU")]
        protected static extern void HashDataWithGPU(IntPtr handle, byte[] data, uint size, int[] keyCols,
            uint keyColsSize, int nodeCount, ref IntPtr hashedBlock, ref int lenght);

        [DllImport("gpuhash.dll", EntryPoint = "INIT")]
        protected static extern IntPtr Init(int gpuNumber);

        [DllImport("gpuhash.dll", EntryPoint = "DESTROY")]
        protected static extern void DESTROY(IntPtr ptr);

        [StructLayout(LayoutKind.Sequential)]
        struct HashedBlock
        {
            public IntPtr Data;
            public int Length;
            public int Hash;
        }

        public List<byte[]> ProcessData(byte[] data, int nodeCount, int[] keys)
        {
            var handle = Init(0);
            int lenght = 0;
            IntPtr hashedData = IntPtr.Zero;
            HashDataWithGPU(handle, data, (uint)data.Length, keys, (uint)keys.Length, nodeCount, ref hashedData, ref lenght);

            if (hashedData == IntPtr.Zero)
            {
                throw new Exception("Ошибка хеширования");
            }

            var blockPointers = new IntPtr[lenght];
            for (int i = 0; i < lenght; i++)
            {
                blockPointers[i] = hashedData + (IntPtr.Size + 4*2)* i;
            }

            var resultData = new List<byte[]>();
            for (int i = 0; i < lenght; i++)
            {
                var datablock = (HashedBlock)Marshal.PtrToStructure(blockPointers[i], typeof(HashedBlock));
                var buffer = new byte[datablock.Length];
                Marshal.Copy(datablock.Data, buffer, 0, buffer.Length);
                resultData.Add(buffer);
                Marshal.FreeHGlobal(datablock.Data);
                Marshal.DestroyStructure(blockPointers[i], typeof(HashedBlock));
            }

            DESTROY(handle);

            return resultData;
        }

        static void Main(string[] args)
        {
            var test = new Program();
            for (int index = 0; index < 100; index++)
            {
                test.ProcessData(File.ReadAllBytes("orders.tbl"), 4, new int[1] { 0 });
            }
            GC.Collect();
        }
    }
}

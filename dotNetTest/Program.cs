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
                Marshal.DestroyStructure(blockPointers[i], typeof(HashedBlock));
            }

            DESTROY(handle);

            return resultData;
        }

        static void Main(string[] args)
        {
            var test = new Program();
            var data = test.ProcessData(File.ReadAllBytes("orders.tbl"), 4, new int[1] { 0 });
            
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace VulkanTutorial.Tools
{
    public unsafe class StringArrayPtrWrapper : IDisposable
    {
        public int Count { get; private set; }
        public byte** StringArrayPtr { get; private set; }

        public StringArrayPtrWrapper(IReadOnlyList<string> inputStrings)
        {
            Count = inputStrings.Count;

            if (Count == 0)
            {
                StringArrayPtr = null;
                return;
            }

            //The allocated block of memory has the pointer array first followed by the strings
            var startOfStringOffset = sizeof(byte*) * Count;
            var totalSize = startOfStringOffset + inputStrings.Sum(x => Encoding.UTF8.GetByteCount(x) + 1);
            
            StringArrayPtr = (byte**) Marshal.AllocHGlobal(totalSize);

            var currentOffset = startOfStringOffset;
            
            for (var i = 0; i < Count; ++i)
            {
                StringArrayPtr[i] = (byte*)StringArrayPtr + currentOffset;//IntPtr.Add(basePtr, currentOffset);intPtrs[i] = IntPtr.Add(basePtr, currentOffset);
                
                var inputString = inputStrings[i];
                var byteCount = Encoding.UTF8.GetByteCount(inputString);
                fixed (char* inputPtr = inputString)
                {
                    Encoding.UTF8.GetBytes(inputPtr, inputString.Length, StringArrayPtr[i], byteCount);
                }
                
                //Zero terminate
                StringArrayPtr[i][byteCount] = 0;
                
                currentOffset += byteCount + 1;
            }
        }

        public void Dispose()
        {
            Marshal.FreeHGlobal((IntPtr)StringArrayPtr);
        }
    }
}
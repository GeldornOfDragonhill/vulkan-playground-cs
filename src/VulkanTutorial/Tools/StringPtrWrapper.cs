using System;
using System.Runtime.InteropServices;
using System.Text;

namespace VulkanTutorial.Tools
{
    public unsafe class StringPtrWrapper : IDisposable
    {
        public byte* StringPtr { get; private set; }
        public StringPtrWrapper(string input)
        {
            var byteCount = Encoding.UTF8.GetByteCount(input);

            StringPtr = (byte*)Marshal.AllocHGlobal(byteCount + 1);
            fixed (char* inputPtr = input)
            {
                Encoding.UTF8.GetBytes(inputPtr, input.Length, StringPtr, byteCount);
            }

            //Zero terminate
            StringPtr[byteCount] = 0;
        }

        public void Dispose()
        {
            Marshal.FreeHGlobal((IntPtr)StringPtr);
        }
    }

}
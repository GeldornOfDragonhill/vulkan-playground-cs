using System;
using System.Runtime.CompilerServices;
using Silk.NET.Vulkan;

namespace VulkanTutorial.Tools
{
    public static class VkCheck
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Success(Result result, string message = "Unexpected Vulkan error")
        {
            if (result != Result.Success)
            {
                throw new InvalidOperationException(message);
            }
        }
    }
}
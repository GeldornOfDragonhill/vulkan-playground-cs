using System;
using Silk.NET.Vulkan;

namespace VulkanTutorial.Tools
{
    public static class VkCheck
    {
        public static void Success(Result result, string message = "Unexpected Vulkan error")
        {
            if (result != Result.Success)
            {
                throw new InvalidOperationException(message);
            }
        }
    }
}
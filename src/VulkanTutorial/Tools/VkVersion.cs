namespace VulkanTutorial.Tools
{
    public static class VkVersion
    {
        public static uint Create(uint major, uint minor, uint patch)
        {
            return major << 22 | minor << 12 | patch;
        }
    }
}
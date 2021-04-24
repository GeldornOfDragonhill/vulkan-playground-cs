using System.Runtime.InteropServices;
using System.Text;

namespace VulkanTutorial.Tools
{
    public static unsafe class StringUtils
    {
        [DllImport("msvcrt.dll")]
        private static extern nuint strlen(byte* str);
        
        public static string GetString(byte* inputString)
        {
            var iter = inputString;
            while (*iter != 0)
            {
                ++iter;
            }

            return Encoding.UTF8.GetString(inputString, (int)(iter - inputString));
        }
    }
}
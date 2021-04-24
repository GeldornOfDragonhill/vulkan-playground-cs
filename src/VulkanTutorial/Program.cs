
using System;

namespace VulkanTutorial
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            try
            {
                var app = new HelloTriangle(true);
                app.Run();
            }
            catch (Exception exception)
            {
                Console.WriteLine(exception);
            }
        }
    }
}
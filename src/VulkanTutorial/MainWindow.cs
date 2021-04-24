using System;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.Windowing;

namespace VulkanTutorial
{
    public abstract class MainWindow
    {
        private readonly IWindow _window;
        
        public MainWindow()
        {
            var options = WindowOptions.DefaultVulkan;
            options.Size = new Vector2D<int>(1000, 800);
            options.Title = "Vulkan Tutorial";

            _window = Window.Create(options);
            
            _window.Load += OnLoad;
            _window.Closing += OnClose;
        }

        public void Run()
        {
            _window.Run();
        }

        protected virtual void OnLoad()
        {
            var inputContext = _window.CreateInput();
            
            foreach (var keyboard in inputContext.Keyboards)
            {
                keyboard.KeyDown += OnKeyDown;
            }
        }
        
        protected virtual void OnClose()
        {
            
        }

        protected void OnKeyDown(IKeyboard keyboard, Key key, int code)
        {
            if (key == Key.Escape)
            {
                _window.Close();
            }
        }

        protected unsafe byte** VkGetRequiredExtensions(out uint count)
        {
            var surface = _window.VkSurface;

            if (surface == null)
            {
                throw new InvalidOperationException("Window has no VkSurface");
            }

            return surface.GetRequiredExtensions(out count);
        }
    }
}
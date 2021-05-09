using System;
using Silk.NET.Core.Contexts;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.Vulkan;
using Silk.NET.Windowing;

namespace VulkanTutorial
{
    public abstract class MainWindow
    {
        private readonly IWindow _window;

        public int Width => _window.Size.X;
        public int Height => _window.Size.Y;

        public double TotalTime => _window.Time;
        
        public MainWindow()
        {
            var options = WindowOptions.DefaultVulkan;
            options.Size = new Vector2D<int>(1000, 800);
            options.Title = "Vulkan Tutorial";

            _window = Window.Create(options);

            _window.Load += OnLoad;
            _window.Resize += OnResize;
            _window.Closing += OnClose;
        }

        protected abstract void Render(double _);

        public void Run()
        {
            _window.Initialize();
            
            _window.Render += Render;
            
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
        
        protected virtual void OnResize(Vector2D<int> obj) { }
        
        protected virtual void OnClose() { }

        protected virtual void OnKeyDown(IKeyboard keyboard, Key key, int code)
        {
            if (key == Key.Escape)
            {
                _window.Close();
            }
        }

        private IVkSurface VkSurface
        {
            get
            {
                var surface = _window.VkSurface;

                if (surface == null)
                {
                    throw new InvalidOperationException("Window has no VkSurface");
                }

                return surface;
            }
        } 

        protected unsafe byte** VkGetRequiredExtensions(out uint count)
        {
            return VkSurface.GetRequiredExtensions(out count);
        }
        
        protected unsafe SurfaceKHR CreateSurface(Instance instance)
        {
            return VkSurface.Create<AllocationCallbacks>(instance.ToHandle(), null).ToSurface();
        }
    }
}
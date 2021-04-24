using System;
using System.Collections.Generic;
using System.Linq;
using Silk.NET.Core;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.EXT;
using Silk.NET.Vulkan.Extensions.KHR;
using VulkanTutorial.Tools;

namespace VulkanTutorial
{
    public unsafe class HelloTriangle : MainWindow
    {
        private const string EngineName = "N/A";

        private static readonly string[] ValidationLayers =
            {
                "VK_LAYER_KHRONOS_validation"
            };

        private static readonly string[] DeviceExtensions =
            {
                KhrSwapchain.ExtensionName
            };

        private readonly bool _enableValidationLayers;
        private readonly Vk _vk = Vk.GetApi();

        private ExtDebugUtils _extDebugUtils;
        private KhrSurface _extKhrSurface;
        private KhrSwapchain _extKhrSwapChain;

        private Instance _instance;
        private DebugUtilsMessengerEXT _debugMessenger;
        private SurfaceKHR _surface;
        private PhysicalDevice _physicalDevice;
        private Device _device;
        private Queue _graphicsQueue;
        private Queue _presentQueue;
        private SwapchainKHR _swapChain;
        private Image[] _swapChainImages;
        private ImageView[] _swapChainImageViews;

        public HelloTriangle(bool enableValidationLayers)
        {
            _enableValidationLayers = enableValidationLayers;
        }

        private void InitVulkan()
        {
            CreateVulkanInstance();
            SetupDebugMessenger();
            _surface = CreateSurface(_instance);
            PickPhysicalDevice(out var cachedDeviceInfo);
            CreateLogicalDevice(cachedDeviceInfo);
            CreateSwapChain(cachedDeviceInfo);
            CreateImageViews(cachedDeviceInfo);
        }

        private void CreateImageViews(CachedDeviceInfo cachedDeviceInfo)
        {
            _swapChainImageViews = new ImageView[_swapChainImages.Length];

            fixed (Image* swapChainImages = _swapChainImages)
            fixed (ImageView* swapChainImageViews = _swapChainImageViews)
            {

                for (var i = 0; i < _swapChainImages.Length; ++i)
                {
                    var imageViewCreateInfo = new ImageViewCreateInfo(
                        image: swapChainImages[i],
                        viewType: ImageViewType.ImageViewType2D,
                        format: cachedDeviceInfo.SurfaceFormat.Format,
                        components: new ComponentMapping(
                            r: ComponentSwizzle.Identity,
                            g: ComponentSwizzle.Identity,
                            b: ComponentSwizzle.Identity,
                            a: ComponentSwizzle.Identity
                        ),
                        subresourceRange: new ImageSubresourceRange(
                            aspectMask: ImageAspectFlags.ImageAspectColorBit,
                            baseMipLevel: 0,
                            levelCount: 1,
                            baseArrayLayer: 0,
                            layerCount: 1
                        )
                    );

                    VkCheck.Success(_vk.CreateImageView(_device, &imageViewCreateInfo, null, &swapChainImageViews[i]), "Failed to create image view");
                }
            }
        }

        private void CreateSwapChain(CachedDeviceInfo cachedDeviceInfo)
        {
            ref var surfaceCapabilities = ref cachedDeviceInfo.SurfaceCapabilities;
            
            var swapChainImageCount = surfaceCapabilities.MinImageCount + 1;

            if (surfaceCapabilities.MaxImageCount != 0 && surfaceCapabilities.MaxImageCount > swapChainImageCount)
            {
                swapChainImageCount = surfaceCapabilities.MaxImageCount;
            }

            var createInfo = new SwapchainCreateInfoKHR(
                surface: _surface,
                minImageCount: swapChainImageCount,
                imageFormat: cachedDeviceInfo.SurfaceFormat.Format,
                imageColorSpace: cachedDeviceInfo.SurfaceFormat.ColorSpace,
                imageExtent: cachedDeviceInfo.Extent2D,
                imageArrayLayers: 1,
                imageUsage: ImageUsageFlags.ImageUsageColorAttachmentBit,
                preTransform: surfaceCapabilities.CurrentTransform,
                compositeAlpha: CompositeAlphaFlagsKHR.CompositeAlphaOpaqueBitKhr,
                presentMode: cachedDeviceInfo.PresentMode,
                clipped: Vk.True
            );


            ref var indices = ref cachedDeviceInfo.QueueFamilyIndices;
            if (indices.GraphicsFamily.Value == indices.PresentFamily.Value)
            {
                createInfo.ImageSharingMode = SharingMode.Exclusive;
            }
            else
            {
                var queueFamilyIndices = stackalloc uint[2]; //Should be valid until the method returns
                queueFamilyIndices[0] = indices.GraphicsFamily.Value;
                queueFamilyIndices[1] = indices.PresentFamily.Value;
                
                createInfo.ImageSharingMode = SharingMode.Concurrent;
                createInfo.QueueFamilyIndexCount = 2;
                createInfo.PQueueFamilyIndices = queueFamilyIndices;
            }

            if (!_vk.TryGetDeviceExtension(_instance, _device, out _extKhrSwapChain))
            {
                throw new InvalidOperationException("KhrSwapChain extension not found");
            }
            
            VkCheck.Success(_extKhrSwapChain.CreateSwapchain(_device, &createInfo, null, out _swapChain), "Failed to create swap chain");
            
            VkCheck.Success(_extKhrSwapChain.GetSwapchainImages(_device, _swapChain, &swapChainImageCount, null));
            _swapChainImages = new Image[swapChainImageCount];
            fixed (Image* swapChainImagesPtr = _swapChainImages)
            {
                VkCheck.Success(_extKhrSwapChain.GetSwapchainImages(_device, _swapChain, &swapChainImageCount, swapChainImagesPtr));
            }
        }

        private void CreateLogicalDevice(CachedDeviceInfo cachedDeviceInfo)
        {
            var queuePriority = 1.0f;

            var uniqueIndices = cachedDeviceInfo.QueueFamilyIndices.GetUniqueIndices();

            var deviceQueueCreateInfos = stackalloc DeviceQueueCreateInfo[uniqueIndices.Count];

            var uniqueIndicesIndex = 0;
            foreach (var uniqueIndex in uniqueIndices)
            {
                deviceQueueCreateInfos[uniqueIndicesIndex] = new DeviceQueueCreateInfo(
                    queueFamilyIndex: uniqueIndex,
                    queueCount: 1,
                    pQueuePriorities: &queuePriority
                );

                ++uniqueIndicesIndex;
            }

            var deviceFeatures = new PhysicalDeviceFeatures();

            using var validationLayers = new StringArrayPtrWrapper(_enableValidationLayers ? ValidationLayers : Array.Empty<string>());
            using var deviceExtensions = new StringArrayPtrWrapper(DeviceExtensions);

            var deviceCreateInfo = new DeviceCreateInfo(
                pQueueCreateInfos: deviceQueueCreateInfos,
                queueCreateInfoCount: (uint)uniqueIndices.Count,
                pEnabledFeatures: &deviceFeatures,
                enabledLayerCount: (uint)validationLayers.Count,
                ppEnabledLayerNames: validationLayers.StringArrayPtr,
                enabledExtensionCount: (uint)deviceExtensions.Count,
                ppEnabledExtensionNames: deviceExtensions.StringArrayPtr
            );
            
            VkCheck.Success(_vk.CreateDevice(_physicalDevice, &deviceCreateInfo, null, out _device), "Failed to create logical device");
            
            _vk.GetDeviceQueue(_device, cachedDeviceInfo.QueueFamilyIndices.GraphicsFamily.Value, 0, out _graphicsQueue);
            _vk.GetDeviceQueue(_device, cachedDeviceInfo.QueueFamilyIndices.PresentFamily.Value, 0, out _presentQueue);
        }

        private void PickPhysicalDevice(out CachedDeviceInfo cachedDeviceInfo)
        {
            cachedDeviceInfo = null;
            
            uint deviceCount;
            VkCheck.Success(_vk.EnumeratePhysicalDevices(_instance, &deviceCount, null));

            if (deviceCount == 0)
            {
                throw new InvalidOperationException("Could not find a graphics card with vulkan support");
            }

            var physicalDevices = stackalloc PhysicalDevice[(int) deviceCount];
            
            VkCheck.Success(_vk.EnumeratePhysicalDevices(_instance, &deviceCount, physicalDevices));

            for (var i = 0; i < deviceCount; i++)
            {
                ref var current = ref physicalDevices[i];
                
                if (IsDeviceSuitable(current, out cachedDeviceInfo))
                {
                    _physicalDevice = current;
                    return;
                }
            }

            throw new InvalidOperationException("Could not find a suitable GPU");
        }

        private class CachedDeviceInfo
        {
            public QueueFamilyIndices QueueFamilyIndices;
            public SurfaceFormatKHR SurfaceFormat;
            public PresentModeKHR PresentMode;
            public SurfaceCapabilitiesKHR SurfaceCapabilities;
            public Extent2D Extent2D;
        }

        private bool IsDeviceSuitable(in PhysicalDevice physicalDevice, out CachedDeviceInfo cachedDeviceInfo)
        {
            cachedDeviceInfo = new CachedDeviceInfo
                {
                    QueueFamilyIndices = FindQueueFamilies(physicalDevice)
                };

            if (!cachedDeviceInfo.QueueFamilyIndices.IsComplete())
            {
                return false;
            }

            if (!CheckDeviceExtensionSupport(physicalDevice))
            {
                return false;
            }

            return QuerySwapChainSupport(physicalDevice, cachedDeviceInfo);
        }

        private bool CheckDeviceExtensionSupport(in PhysicalDevice physicalDevice)
        {
            uint extensionCount;
            VkCheck.Success(_vk.EnumerateDeviceExtensionProperties(physicalDevice, (byte*)null, &extensionCount, null));

            var availableExtensions = stackalloc ExtensionProperties[(int) extensionCount];
            VkCheck.Success(_vk.EnumerateDeviceExtensionProperties(physicalDevice, (byte*)null, &extensionCount, availableExtensions));

            var requiredExtensions = new HashSet<string>(DeviceExtensions);

            //TODO: possibly change for something comparing the utf-8 strings as the number of extensions might be large
            for (uint i = 0; i < extensionCount; ++i)
            {
                var extensionName = StringUtils.GetString(availableExtensions[i].ExtensionName);

                requiredExtensions.Remove(extensionName);
            }

            return requiredExtensions.Count == 0;
        }

        private bool QuerySwapChainSupport(PhysicalDevice physicalDevice, CachedDeviceInfo cachedDeviceInfo)
        {
            uint formatCount;
            VkCheck.Success(_extKhrSurface.GetPhysicalDeviceSurfaceFormats(physicalDevice, _surface, &formatCount, null));

            if (formatCount == 0)
            {
                return false;
            }

            var surfaceFormats = stackalloc SurfaceFormatKHR[(int) formatCount];
            VkCheck.Success(_extKhrSurface.GetPhysicalDeviceSurfaceFormats(physicalDevice, _surface, &formatCount, surfaceFormats));
            
            //Choose the format that will be used, fallback to the first format
            cachedDeviceInfo.SurfaceFormat = surfaceFormats[0];
            for (uint i = 0; i < formatCount; ++i)
            {
                ref var availableFormat = ref surfaceFormats[i];

                if (availableFormat.Format == Format.B8G8R8A8Srgb && availableFormat.ColorSpace == ColorSpaceKHR.ColorspaceSrgbNonlinearKhr)
                {
                    cachedDeviceInfo.SurfaceFormat = availableFormat;
                }
            }
            

            uint presentModeCount;
            VkCheck.Success(_extKhrSurface.GetPhysicalDeviceSurfacePresentModes(physicalDevice, _surface, &presentModeCount, null));

            if (presentModeCount == 0)
            {
                return false;
            }
            
            var presentModes = stackalloc PresentModeKHR[(int) formatCount];
            VkCheck.Success(_extKhrSurface.GetPhysicalDeviceSurfacePresentModes(physicalDevice, _surface, &presentModeCount, presentModes));
            
            //Choose the present mode, fallback to the always existing FIFO mode
            cachedDeviceInfo.PresentMode = PresentModeKHR.PresentModeFifoKhr;
            for (uint i = 0; i < presentModeCount; ++i)
            {
                ref var presentMode = ref presentModes[i];

                if (presentModes[i] == PresentModeKHR.PresentModeMailboxKhr)
                {
                    cachedDeviceInfo.PresentMode = PresentModeKHR.PresentModeMailboxKhr;
                }
            }
            
            
            
            VkCheck.Success(_extKhrSurface.GetPhysicalDeviceSurfaceCapabilities(physicalDevice, _surface, out var surfaceCapabilities));

            if (surfaceCapabilities.CurrentExtent.Width != uint.MaxValue)
            {
                cachedDeviceInfo.Extent2D = surfaceCapabilities.CurrentExtent;
            }
            else
            {
                var actualExtend = new Extent2D((uint) Width, (uint) Height);

                actualExtend.Width = Math.Max(surfaceCapabilities.MinImageExtent.Width, Math.Min(surfaceCapabilities.MaxImageExtent.Width, actualExtend.Width));
                actualExtend.Height = Math.Max(surfaceCapabilities.MinImageExtent.Height, Math.Min(surfaceCapabilities.MaxImageExtent.Height, actualExtend.Height));

                cachedDeviceInfo.Extent2D = actualExtend;
                cachedDeviceInfo.SurfaceCapabilities = surfaceCapabilities;
            }

            return true;
        }

        private struct QueueFamilyIndices
        {
            public uint? GraphicsFamily;
            public uint? PresentFamily;

            public bool IsComplete()
            {
                return GraphicsFamily.HasValue && PresentFamily.HasValue;
            }

            public IReadOnlySet<uint> GetUniqueIndices()
            {
                var set = new HashSet<uint>(2);

                if (GraphicsFamily.HasValue)
                {
                    set.Add(GraphicsFamily.Value);
                }

                if (PresentFamily.HasValue)
                {
                    set.Add(PresentFamily.Value);
                }

                return set;
            }
        }

        private  QueueFamilyIndices FindQueueFamilies(in PhysicalDevice physicalDevice)
        {
            uint queueFamilyCount = 0;
            _vk.GetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, null);

            var queueFamilies = stackalloc QueueFamilyProperties[(int) queueFamilyCount];
            _vk.GetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies);

            var queueFamilyIndices = new QueueFamilyIndices();

            for (uint i = 0; i < queueFamilyCount; ++i)
            {
                ref var queueFamily = ref queueFamilies[i];

                if ((queueFamily.QueueFlags & QueueFlags.QueueGraphicsBit) != 0)
                {
                    queueFamilyIndices.GraphicsFamily = i;
                }

                Bool32 presentSupport;
                VkCheck.Success(_extKhrSurface.GetPhysicalDeviceSurfaceSupport(physicalDevice, i, _surface, &presentSupport), "Error checking for surface support");

                if (presentSupport.Value != 0)
                {
                    queueFamilyIndices.PresentFamily = i;
                }

                if (queueFamilyIndices.IsComplete())
                {
                    break;
                }
            }

            return  queueFamilyIndices;
        }  

        private void CreateVulkanInstance()
        {
            using var applicationName = new StringPtrWrapper(GetType().Name);
            using var engineName = new StringPtrWrapper(EngineName);

            var applicationInfo = new ApplicationInfo(
                pApplicationName: applicationName.StringPtr,
                applicationVersion: VkVersion.Create(1, 0, 0),
                pEngineName: engineName.StringPtr,
                engineVersion: VkVersion.Create(0, 0, 1),
                apiVersion: Vk.Version12
            );

            IReadOnlyList<string> requiredLayers = Array.Empty<string>();
            IReadOnlyList<string> additionalRequiredExtensions = Array.Empty<string>();

            if (_enableValidationLayers)
            {
                var layerNames = GetInstanceLayersNames();

                if (!ValidationLayers.All(validationLayerName => layerNames.Contains(validationLayerName)))
                {
                    throw new InvalidOperationException("Validation Layers are not available");
                }

                requiredLayers = ValidationLayers;
                additionalRequiredExtensions = new[]
                    {
                        ExtDebugUtils.ExtensionName
                    };
            }
            
            using var requiredLayerArrayPtr = new StringArrayPtrWrapper(requiredLayers);
            using var additionalRequiredExtensionsArrayPtr = new StringArrayPtrWrapper(additionalRequiredExtensions);
            
            var vkRequiredExtensions = VkGetRequiredExtensions(out var numRequiredExtensions);

            var numEnabledExtensions = (int)numRequiredExtensions + additionalRequiredExtensions.Count;
            var enabledExtensions = stackalloc byte*[numEnabledExtensions];

            for (var i = 0; i < numRequiredExtensions; i++)
            {
                enabledExtensions[i] = vkRequiredExtensions[i];
            }

            for (var i = numRequiredExtensions; i < numEnabledExtensions; ++i)
            {
                enabledExtensions[i] = additionalRequiredExtensionsArrayPtr.StringArrayPtr[i - numRequiredExtensions];
            }

            var instanceCreateInfo = new InstanceCreateInfo(
                pApplicationInfo: &applicationInfo,
                enabledExtensionCount: (uint)numEnabledExtensions,
                ppEnabledExtensionNames: enabledExtensions,
                enabledLayerCount: (uint)requiredLayerArrayPtr.Count,
                ppEnabledLayerNames: requiredLayerArrayPtr.StringArrayPtr
            );
            
            VkCheck.Success(_vk.CreateInstance(instanceCreateInfo, null, out _instance), "Couldn't create instance");

            if (!_vk.TryGetInstanceExtension(_instance, out _extKhrSurface))
            {
                throw new InvalidOperationException("KHR_surface was not found");
            }
        }

        private HashSet<string> GetInstanceLayersNames()
        {
            uint layerCount;

            VkCheck.Success(_vk.EnumerateInstanceLayerProperties(&layerCount, null));

            var layerProperties = stackalloc LayerProperties[(int) layerCount];
            
            VkCheck.Success(_vk.EnumerateInstanceLayerProperties(&layerCount, layerProperties));

            var result = new HashSet<string>((int)layerCount); 

            for (var i = 0; i < layerCount; i++)
            {
                result.Add(StringUtils.GetString(layerProperties[i].LayerName));
            }

            return result;
        }

        protected override void OnLoad()
        {
            base.OnLoad();
            
            InitVulkan();
        }

        protected override void OnClose()
        {
            foreach (var swapChainImageView in _swapChainImageViews)
            {
                _vk.DestroyImageView(_device, swapChainImageView, null);
            }
            
            _extKhrSwapChain.DestroySwapchain(_device, _swapChain, null);
            
            _vk.DestroyDevice(_device, null);
            
            _extKhrSurface.DestroySurface(_instance, _surface, null);
            
            if (_debugMessenger.Handle != 0)
            {
                _extDebugUtils.DestroyDebugUtilsMessenger(_instance, _debugMessenger, null);
            }
            
            _vk.DestroyInstance(_instance, null);
            
            base.OnClose();
        }

        private void SetupDebugMessenger()
        {
            if (!_enableValidationLayers)
            {
                return;
            }

            if (!_vk.TryGetInstanceExtension(_instance, out _extDebugUtils))
            {
                throw new InvalidOperationException("Could not get the debug utils extension");
            }

            var createInfo = new DebugUtilsMessengerCreateInfoEXT(
                messageSeverity: DebugUtilsMessageSeverityFlagsEXT.DebugUtilsMessageSeverityWarningBitExt |
                                 DebugUtilsMessageSeverityFlagsEXT.DebugUtilsMessageSeverityErrorBitExt,
                messageType: DebugUtilsMessageTypeFlagsEXT.DebugUtilsMessageTypeGeneralBitExt |
                             DebugUtilsMessageTypeFlagsEXT.DebugUtilsMessageTypePerformanceBitExt |
                             DebugUtilsMessageTypeFlagsEXT.DebugUtilsMessageTypeValidationBitExt,
                pfnUserCallback: (DebugUtilsMessengerCallbackFunctionEXT)DebugCallback
            );
            
            VkCheck.Success(_extDebugUtils.CreateDebugUtilsMessenger(_instance, &createInfo, null, out _debugMessenger), "Could not create debug utils messenger");
        }
        
        private uint DebugCallback(DebugUtilsMessageSeverityFlagsEXT messageSeverity, DebugUtilsMessageTypeFlagsEXT messageTypes, DebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
        {
            var message = StringUtils.GetString(pCallbackData->PMessage);

            var fullMessage = $"[{messageSeverity}] ({messageTypes}) {message}";

            if ((messageSeverity & DebugUtilsMessageSeverityFlagsEXT.DebugUtilsMessageSeverityErrorBitExt) != 0)
            {
                throw new InvalidOperationException("A Vulkan validation error was encountered: " + fullMessage);
            }

            Console.WriteLine(fullMessage);
            
            return Vk.False;
        }

    }
}
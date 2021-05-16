using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Silk.NET.Assimp;
using Silk.NET.Core;
using Silk.NET.Maths;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.EXT;
using Silk.NET.Vulkan.Extensions.KHR;
using SixLabors.ImageSharp.PixelFormats;
using VulkanTutorial.Tools;
using Buffer = Silk.NET.Vulkan.Buffer;

namespace VulkanTutorial
{
    public unsafe class HelloTriangle : MainWindow
    {
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
        private class CachedDeviceInfo
        {
            public QueueFamilyIndices QueueFamilyIndices;
            public SurfaceFormatKHR SurfaceFormat;
            public PresentModeKHR PresentMode;
            public SurfaceCapabilitiesKHR SurfaceCapabilities;
            public Extent2D Extent2D;
            public float? MaxSamplerAnisotropy;
        }
        
        struct Vertex
        {
            public Vector3D<float> Pos;
            public Vector3D<float> Color;
            public Vector2D<float> TexCoord;

            public static VertexInputBindingDescription BindingDescription => new VertexInputBindingDescription(
                binding: 0,
                stride: (uint)sizeof(Vertex),
                inputRate: VertexInputRate.Vertex
            );

            public static VertexInputAttributeDescription[] AttributeDescriptions => new VertexInputAttributeDescription[]
                {
                    new(
                        binding: 0,
                        location: 0,
                        format: Format.R32G32B32Sfloat,
                        offset: (uint) Marshal.OffsetOf<Vertex>(nameof(Pos))
                    ),
                    new(
                        binding: 0,
                        location: 1,
                        format: Format.R32G32B32Sfloat,
                        offset: (uint) Marshal.OffsetOf<Vertex>(nameof(Color))
                    ),
                    new(
                        binding: 0,
                        location: 2,
                        format: Format.R32G32Sfloat,
                        offset: (uint) Marshal.OffsetOf<Vertex>(nameof(TexCoord))
                    )
                };
        }

        [StructLayout(LayoutKind.Sequential)]
        struct UniformBufferObject
        {
            public Matrix4X4<float> Model;
            public Matrix4X4<float> View;
            public Matrix4X4<float> Proj;
        }

        private const string EngineName = "N/A";

        private const int MaxFramesInFlight = 2; 

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

        private Vertex[] _vertices;
        private ushort[] _indices;

        private ExtDebugUtils _extDebugUtils;
        private KhrSurface _extKhrSurface;
        private KhrSwapchain _extKhrSwapChain;

        private CachedDeviceInfo _cachedDeviceInfo;
        private bool _framebufferResized = false;
        private bool _pauseRendering = false;

        private Instance _instance;
        private DebugUtilsMessengerEXT _debugMessenger;
        private SurfaceKHR _surface;
        private PhysicalDevice _physicalDevice;
        private Device _device;
        private Queue _graphicsQueue;
        private Queue _presentQueue;
        private SwapchainKHR _swapChain;
        private Image[] _swapChainImages;
        private Format _swapChainImageFormat;
        private Extent2D _swapChainExtent;
        private ImageView[] _swapChainImageViews;
        private RenderPass _renderPass;
        private DescriptorSetLayout _descriptorSetLayout;
        private PipelineLayout _pipelineLayout;
        private Pipeline _graphicsPipeline;
        private Framebuffer[] _swapChainFramebuffers;
        private CommandPool _commandPool;
        private Buffer _vertexBuffer;
        private DeviceMemory _vertexBufferMemory;
        private Buffer _indexBuffer;
        private DeviceMemory _indexBufferMemory;
        private Buffer[] _uniformBuffers;
        private DeviceMemory[] _uniformBuffersMemory;
        private CommandBuffer[] _commandBuffers;
        private DescriptorPool _descriptorPool;
        private DescriptorSet[] _descriptorSets;
        private uint _mipLevels;
        private Image _textureImage;
        private DeviceMemory _textureImageMemory;
        private ImageView _textureImageView;
        private Sampler _textureSampler;
        private Image _depthImage;
        private DeviceMemory _depthImageMemory;
        private ImageView _depthImageView;

        private readonly Semaphore[] _imageAvailableSemaphores = new Semaphore[MaxFramesInFlight];
        private readonly Semaphore[] _renderFinishedSemaphores = new Semaphore[MaxFramesInFlight];
        private readonly Fence[] _inFlightFences = new Fence[MaxFramesInFlight];
        private Fence[] _imagesInFlight;

        private int _currentFrame = 0;
        
        public HelloTriangle(bool enableValidationLayers)
        {
            _enableValidationLayers = enableValidationLayers;
        }
        
        protected override void OnLoad()
        {
            base.OnLoad();
            
            InitVulkan();
        }
        
        private void InitVulkan()
        {
            CreateVulkanInstance();
            SetupDebugMessenger();
            _surface = CreateSurface(_instance);
            PickPhysicalDevice(out _cachedDeviceInfo);
            CreateLogicalDevice();

            AssembleSwapChain();
            
            CreateSyncObjects();
        }

        private void AssembleSwapChain(bool recreate = false)
        {
            if (recreate)
            {
                CleanupSwapChain();
            
                UpdateSurfaceCapabilities(_physicalDevice, _cachedDeviceInfo);
            }

            if (_cachedDeviceInfo.Extent2D.Width == 0 || _cachedDeviceInfo.Extent2D.Height == 0)
            {
                //Wait until window is valid again
                _pauseRendering = true;
                return;
            }

            _pauseRendering = false;
            
            CreateSwapChain();
            CreateImageViews();
            CreateRenderPass();
            
            if (!recreate)
            {
                CreateDescriptorSetLayout();
            }
            
            CreateGraphicsPipeline();
           
            if (!recreate)
            {
                CreateCommandPool();
            }
            
            CreateDepthResources();
            CreateFramebuffers();

            if(!recreate)
            {
                CreateTextureImage();
                CreateTextureImageView();
                CreateTextureSampler();
                LoadModel();
                CreateVertexBuffer();
                CreateIndexBuffer();
            }
            CreateUniformBuffer();
            CreateDescriptorPool();
            CreateDescriptorSets();
            CreateCommandBuffers();
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
                messageSeverity: DebugUtilsMessageSeverityFlagsEXT.DebugUtilsMessageSeverityWarningBitExt
                                 | DebugUtilsMessageSeverityFlagsEXT.DebugUtilsMessageSeverityErrorBitExt,
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

            if (!QuerySwapChainSupport(physicalDevice, cachedDeviceInfo))
            {
                return false;
            }

            return CheckDeviceFeatures(physicalDevice, cachedDeviceInfo);
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
                if (presentModes[i] == PresentModeKHR.PresentModeMailboxKhr)
                {
                    cachedDeviceInfo.PresentMode = PresentModeKHR.PresentModeMailboxKhr;
                    break;
                }
            }

            UpdateSurfaceCapabilities(physicalDevice, cachedDeviceInfo);

            return true;
        }

        private void UpdateSurfaceCapabilities(PhysicalDevice physicalDevice, CachedDeviceInfo cachedDeviceInfo)
        {
            VkCheck.Success(_extKhrSurface.GetPhysicalDeviceSurfaceCapabilities(physicalDevice, _surface, out cachedDeviceInfo.SurfaceCapabilities));
            
            ref var surfaceCapabilities = ref cachedDeviceInfo.SurfaceCapabilities;
            
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
            }
        }

        private bool CheckDeviceFeatures(PhysicalDevice physicalDevice, CachedDeviceInfo cachedDeviceInfo)
        {
            PhysicalDeviceFeatures deviceFeatures;
            _vk.GetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);
            
            PhysicalDeviceProperties properties;
            _vk.GetPhysicalDeviceProperties(physicalDevice, &properties);

            if (deviceFeatures.SamplerAnisotropy)
            {
                cachedDeviceInfo.MaxSamplerAnisotropy = properties.Limits.MaxSamplerAnisotropy;
            }

            return true;
        }
        
        private void CreateLogicalDevice()
        {
            var queuePriority = 1.0f;

            var uniqueIndices = _cachedDeviceInfo.QueueFamilyIndices.GetUniqueIndices();

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

            var deviceFeatures = new PhysicalDeviceFeatures(
                samplerAnisotropy: _cachedDeviceInfo.MaxSamplerAnisotropy.HasValue
            );

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
            
            _vk.GetDeviceQueue(_device, _cachedDeviceInfo.QueueFamilyIndices.GraphicsFamily.Value, 0, out _graphicsQueue);
            _vk.GetDeviceQueue(_device, _cachedDeviceInfo.QueueFamilyIndices.PresentFamily.Value, 0, out _presentQueue);
        }
        
        private void CreateSwapChain()
        {
            ref var surfaceCapabilities = ref _cachedDeviceInfo.SurfaceCapabilities;
            
            var swapChainImageCount = surfaceCapabilities.MinImageCount + 1;

            if (surfaceCapabilities.MaxImageCount != 0 && surfaceCapabilities.MaxImageCount > swapChainImageCount)
            {
                swapChainImageCount = surfaceCapabilities.MaxImageCount;
            }

            var createInfo = new SwapchainCreateInfoKHR(
                surface: _surface,
                minImageCount: swapChainImageCount,
                imageFormat: _cachedDeviceInfo.SurfaceFormat.Format,
                imageColorSpace: _cachedDeviceInfo.SurfaceFormat.ColorSpace,
                imageExtent: _cachedDeviceInfo.Extent2D,
                imageArrayLayers: 1,
                imageUsage: ImageUsageFlags.ImageUsageColorAttachmentBit,
                preTransform: surfaceCapabilities.CurrentTransform,
                compositeAlpha: CompositeAlphaFlagsKHR.CompositeAlphaOpaqueBitKhr,
                presentMode: _cachedDeviceInfo.PresentMode,
                clipped: Vk.True
            );


            ref var indices = ref _cachedDeviceInfo.QueueFamilyIndices;
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

            _swapChainImageFormat = _cachedDeviceInfo.SurfaceFormat.Format;
            _swapChainExtent = _cachedDeviceInfo.Extent2D;
        }

        private void CreateImageViews()
        {
            _swapChainImageViews = new ImageView[_swapChainImages.Length];

            for (var i = 0; i < _swapChainImages.Length; ++i)
            {
                _swapChainImageViews[i] = CreateImageView(_swapChainImages[i], _swapChainImageFormat, ImageAspectFlags.ImageAspectColorBit, 1);
            }
        }

        private void CreateRenderPass()
        {
            const int numAttachments = 2;
            var attachments = stackalloc AttachmentDescription[2];
            
            attachments[0] = new AttachmentDescription(
                format: _swapChainImageFormat,
                samples: SampleCountFlags.SampleCount1Bit,
                loadOp: AttachmentLoadOp.Clear,
                storeOp: AttachmentStoreOp.Store,
                stencilLoadOp: AttachmentLoadOp.DontCare,
                stencilStoreOp: AttachmentStoreOp.DontCare,
                initialLayout: ImageLayout.Undefined,
                finalLayout: ImageLayout.PresentSrcKhr
            );

            var colorAttachmentRef = new AttachmentReference(0, ImageLayout.ColorAttachmentOptimal);
            
            attachments[1] = new AttachmentDescription(
                format: FindDepthFormat(),
                samples: SampleCountFlags.SampleCount1Bit,
                loadOp: AttachmentLoadOp.Clear,
                storeOp: AttachmentStoreOp.DontCare,
                stencilLoadOp: AttachmentLoadOp.DontCare,
                stencilStoreOp: AttachmentStoreOp.DontCare,
                initialLayout: ImageLayout.Undefined,
                finalLayout: ImageLayout.DepthStencilAttachmentOptimal
            );

            var depthAttachmentRef = new AttachmentReference(
                attachment: 1,
                layout: ImageLayout.DepthStencilAttachmentOptimal
            );
            
            var subpass = new SubpassDescription(
                pipelineBindPoint: PipelineBindPoint.Graphics,
                colorAttachmentCount: 1,
                pColorAttachments: &colorAttachmentRef,
                pDepthStencilAttachment: &depthAttachmentRef
            );

            var dependency = new SubpassDependency(
                srcSubpass: Vk.SubpassExternal,
                dstSubpass: 0,
                srcStageMask: PipelineStageFlags.PipelineStageColorAttachmentOutputBit | PipelineStageFlags.PipelineStageEarlyFragmentTestsBit,
                srcAccessMask: 0,
                dstStageMask: PipelineStageFlags.PipelineStageColorAttachmentOutputBit | PipelineStageFlags.PipelineStageEarlyFragmentTestsBit,
                dstAccessMask: AccessFlags.AccessColorAttachmentWriteBit | AccessFlags.AccessDepthStencilAttachmentWriteBit
            );

            var renderPassInfo = new RenderPassCreateInfo(
                attachmentCount: numAttachments,
                pAttachments: attachments,
                subpassCount: 1,
                pSubpasses: &subpass,
                dependencyCount: 1,
                pDependencies: &dependency
            );
            
            VkCheck.Success(_vk.CreateRenderPass(_device, &renderPassInfo, null, out _renderPass), "Failed to create render pass");
        }

        private void CreateDescriptorSetLayout()
        {
            const int numBindings = 2;
            var bindings = stackalloc DescriptorSetLayoutBinding[numBindings];
            
            bindings[0] = new DescriptorSetLayoutBinding(
                binding: 0,
                descriptorType: DescriptorType.UniformBuffer,
                descriptorCount: 1,
                stageFlags: ShaderStageFlags.ShaderStageVertexBit,
                pImmutableSamplers: null
            );

            bindings[1] = new DescriptorSetLayoutBinding(
                binding: 1,
                descriptorType: DescriptorType.CombinedImageSampler,
                descriptorCount: 1,
                stageFlags: ShaderStageFlags.ShaderStageFragmentBit,
                pImmutableSamplers: null
            );

            var layoutInfo = new DescriptorSetLayoutCreateInfo(
                bindingCount: numBindings,
                pBindings: bindings
            );
            
            VkCheck.Success(_vk.CreateDescriptorSetLayout(_device, &layoutInfo, null, out _descriptorSetLayout), "Failed to create descriptor set layout!");
        }
        
        private void CreateGraphicsPipeline()
        {
            var vertShaderModule = CreateShaderModule("shader:/Shaders/shader.vert.spv");
            var fragShaderModule = CreateShaderModule("shader:/Shaders/shader.frag.spv");
            
            using var mainName = new StringPtrWrapper("main");

            var shaderStages = stackalloc PipelineShaderStageCreateInfo[2];
            
            shaderStages[0] = new PipelineShaderStageCreateInfo(
                stage: ShaderStageFlags.ShaderStageVertexBit,
                module: vertShaderModule,
                pName: mainName.StringPtr
            );
            
            shaderStages[1] = new PipelineShaderStageCreateInfo(
                stage: ShaderStageFlags.ShaderStageFragmentBit,
                module: fragShaderModule,
                pName: mainName.StringPtr
            );

            var bindingDescription = Vertex.BindingDescription;
            var attributeDescriptions = Vertex.AttributeDescriptions;

            fixed (VertexInputAttributeDescription* attributeDescriptionsPtr = attributeDescriptions)
            {
                var vertexInputInfo = new PipelineVertexInputStateCreateInfo(
                    vertexBindingDescriptionCount: 1,
                    vertexAttributeDescriptionCount: (uint) attributeDescriptions.Length,
                    pVertexBindingDescriptions: &bindingDescription,
                    pVertexAttributeDescriptions: attributeDescriptionsPtr
                );


                var inputAssembly = new PipelineInputAssemblyStateCreateInfo(
                    topology: PrimitiveTopology.TriangleList,
                    primitiveRestartEnable: Vk.False
                );

                var viewport = new Viewport(
                    x: 0f,
                    y: 0f,
                    width: (float) _swapChainExtent.Width,
                    height: (float) _swapChainExtent.Height,
                    minDepth: 0f,
                    maxDepth: 1f
                );

                var scissor = new Rect2D(new Offset2D(0, 0), _cachedDeviceInfo.Extent2D);

                var viewportState = new PipelineViewportStateCreateInfo(
                    viewportCount: 1,
                    pViewports: &viewport,
                    scissorCount: 1,
                    pScissors: &scissor
                );

                var rasterizer = new PipelineRasterizationStateCreateInfo(
                    depthClampEnable: Vk.False,
                    rasterizerDiscardEnable: Vk.False,
                    polygonMode: PolygonMode.Fill,
                    lineWidth: 1f,
                    cullMode: CullModeFlags.CullModeBackBit,
                    frontFace: FrontFace.CounterClockwise,
                    depthBiasEnable: Vk.False,
                    depthBiasConstantFactor: 0f,
                    depthBiasClamp: 0f,
                    depthBiasSlopeFactor: 0f
                );

                var multisampling = new PipelineMultisampleStateCreateInfo(
                    sampleShadingEnable: Vk.False,
                    rasterizationSamples: SampleCountFlags.SampleCount1Bit,
                    minSampleShading: 1f,
                    alphaToCoverageEnable: Vk.False,
                    alphaToOneEnable: Vk.False
                );

                var colorBlendAttachment = new PipelineColorBlendAttachmentState(
                    colorWriteMask: ColorComponentFlags.ColorComponentABit | ColorComponentFlags.ColorComponentRBit | ColorComponentFlags.ColorComponentGBit | ColorComponentFlags.ColorComponentBBit,
                    blendEnable: Vk.False
                );

                var colorBlending = new PipelineColorBlendStateCreateInfo(
                    logicOpEnable: Vk.False,
                    logicOp: LogicOp.Copy,
                    attachmentCount: 1,
                    pAttachments: &colorBlendAttachment
                );

                var descriptorSetLayout = _descriptorSetLayout;
                
                var pipelineLayoutInfo = new PipelineLayoutCreateInfo(
                    flags: 0,
                    setLayoutCount: 1,
                    pSetLayouts: &descriptorSetLayout
                );

                VkCheck.Success(_vk.CreatePipelineLayout(_device, &pipelineLayoutInfo, null, out _pipelineLayout), "Failed to create pipeline layout");

                var depthStencil = new PipelineDepthStencilStateCreateInfo(
                    depthTestEnable: Vk.True,
                    depthWriteEnable: Vk.True,
                    depthCompareOp: CompareOp.Less,
                    depthBoundsTestEnable: Vk.False,
                    stencilTestEnable: Vk.False
                );

                var pipelineInfo = new GraphicsPipelineCreateInfo(
                    stageCount: 2,
                    pStages: shaderStages,
                    pVertexInputState: &vertexInputInfo,
                    pInputAssemblyState: &inputAssembly,
                    pViewportState: &viewportState,
                    pRasterizationState: &rasterizer,
                    pMultisampleState: &multisampling,
                    pDepthStencilState: &depthStencil,
                    pColorBlendState: &colorBlending,
                    pDynamicState: null,
                    layout: _pipelineLayout,
                    renderPass: _renderPass,
                    subpass: 0,
                    basePipelineHandle: null,
                    basePipelineIndex: -1
                );

                VkCheck.Success(_vk.CreateGraphicsPipelines(_device, default, 1, &pipelineInfo, null, out _graphicsPipeline), "Failed to create the graphics pipeline");
            }


            _vk.DestroyShaderModule(_device, fragShaderModule, null);
            _vk.DestroyShaderModule(_device, vertShaderModule, null);
        }

        private ShaderModule CreateShaderModule(string shaderResource)
        {
            var shaderStream = Assembly.GetExecutingAssembly().GetManifestResourceStream(shaderResource);

            if (shaderStream == null)
            {
                throw new InvalidOperationException($"Shader '{shaderResource}' not found");
            }

            var shaderSize = (int)shaderStream.Length;
            //TODO: check if 4byte alignment requirements are always satisfied
            Span<byte> shaderBytes = stackalloc byte[shaderSize];

            if (shaderStream.Read(shaderBytes) != shaderSize)
            {
                throw new InvalidOperationException($"Could not read shader '{shaderResource}'");
            }

            ShaderModule shaderModule;
            
            fixed (byte* shaderBytesPtr = shaderBytes)
            {
                var createInfo = new ShaderModuleCreateInfo(
                    codeSize: (uint)shaderSize,
                    pCode: (uint*)shaderBytesPtr
                );

                
                VkCheck.Success(_vk.CreateShaderModule(_device, &createInfo, null, &shaderModule), "Failed to create shader module");
            }

            return shaderModule;
        }
        
        private void CreateFramebuffers()
        {
            _swapChainFramebuffers = new Framebuffer[_swapChainImageViews.Length];

            const int numAttachments = 2;
            var attachments = stackalloc ImageView[numAttachments];

            for (var i = 0; i < _swapChainImageViews.Length; ++i)
            {
                attachments[0] = _swapChainImageViews[i];
                attachments[1] = _depthImageView;

                var framebufferInfo = new FramebufferCreateInfo(
                    renderPass: _renderPass,
                    attachmentCount: numAttachments,
                    pAttachments: attachments,
                    width: _swapChainExtent.Width,
                    height: _swapChainExtent.Height,
                    layers: 1
                );
                
                VkCheck.Success(_vk.CreateFramebuffer(_device, &framebufferInfo, null, out _swapChainFramebuffers[i]), "Failed to create framebuffer");
            }
        }
        
        private void CreateCommandPool()
        {
            var poolInfo = new CommandPoolCreateInfo(
                queueFamilyIndex: _cachedDeviceInfo.QueueFamilyIndices.GraphicsFamily,
                flags: 0
            );
            
            VkCheck.Success(_vk.CreateCommandPool(_device, &poolInfo, null, out _commandPool), "Failed to create command pool");
        }

        private void CreateBuffer(uint size, BufferUsageFlags usage, MemoryPropertyFlags properties, out Buffer buffer, out DeviceMemory bufferMemory)
        {
            var bufferInfo = new BufferCreateInfo(
                size: size,
                usage: usage,
                sharingMode: SharingMode.Exclusive
            );
            
            VkCheck.Success(_vk.CreateBuffer(_device, &bufferInfo, null, out buffer), "Failed to create vertex buffer");

            MemoryRequirements memoryRequirements;
            _vk.GetBufferMemoryRequirements(_device, buffer, &memoryRequirements);

            var allocInfo = new MemoryAllocateInfo(
                allocationSize: memoryRequirements.Size,
                memoryTypeIndex: FindMemoryType(memoryRequirements.MemoryTypeBits, properties)
            );
            
            VkCheck.Success(_vk.AllocateMemory(_device, &allocInfo, null, out bufferMemory), "Failed to allocate vertex buffer memory!");
            
            VkCheck.Success(_vk.BindBufferMemory(_device, buffer, bufferMemory, 0));
        }

        private void CreateDepthResources()
        {
            var depthFormat = FindDepthFormat();
            
            CreateImage(_swapChainExtent.Width, _swapChainExtent.Height, 1, depthFormat, ImageTiling.Optimal, ImageUsageFlags.ImageUsageDepthStencilAttachmentBit, MemoryPropertyFlags.MemoryPropertyDeviceLocalBit, out _depthImage, out _depthImageMemory);

            _depthImageView = CreateImageView(_depthImage, depthFormat, ImageAspectFlags.ImageAspectDepthBit, 1);
        }

        private Format FindDepthFormat()
        {
            return FindSupportedFormat(new[] {Format.D32Sfloat, Format.D32SfloatS8Uint, Format.D24UnormS8Uint}, ImageTiling.Optimal, FormatFeatureFlags.FormatFeatureDepthStencilAttachmentBit);
        }

        private Format FindSupportedFormat(IEnumerable<Format> candidates, ImageTiling tiling, FormatFeatureFlags featureFlags)
        {
            foreach (var candidate in candidates)
            {
                FormatProperties props;
                
                _vk.GetPhysicalDeviceFormatProperties(_physicalDevice, candidate, &props);

                switch (tiling)
                {
                    case ImageTiling.Linear when (props.LinearTilingFeatures & featureFlags) == featureFlags:
                        return candidate;
                    case ImageTiling.Optimal when (props.OptimalTilingFeatures & featureFlags) == featureFlags:
                        return candidate;
                }
            }

            throw new InvalidOperationException("Failed to find supported format");
        }

        private void CreateTextureImage()
        {
            var textureStream = Assembly.GetExecutingAssembly().GetManifestResourceStream("VulkanTutorial.Textures.viking_room.png");

            if (textureStream == null)
            {
                throw new InvalidOperationException($"Texture not found");
            }

            SixLabors.ImageSharp.Image<Rgba32> textureImage;
            
            try
            {
                textureImage = SixLabors.ImageSharp.Image.Load<Rgba32>(textureStream);
            }
            catch
            {
                throw new InvalidOperationException("Failed to load texture");
            }
            
            var textureWidth = (uint)textureImage.Width;
            var textureHeight = (uint) textureImage.Height;

            _mipLevels = (uint)Math.Floor(Math.Log2(Math.Max(textureWidth, textureHeight))) + 1;

            var imageSize = textureWidth * textureHeight * 4;

            CreateBuffer(imageSize, BufferUsageFlags.BufferUsageTransferSrcBit, MemoryPropertyFlags.MemoryPropertyHostVisibleBit | MemoryPropertyFlags.MemoryPropertyHostCoherentBit, out var stagingBuffer, out var stagingBufferMemory);

            if (!textureImage.TryGetSinglePixelSpan(out var pixelDataSpan))
            {
                throw new InvalidOperationException("Could not get pixel data");
            }

            fixed (Rgba32* pixelBuffer = pixelDataSpan)
            {

                void* data;
                VkCheck.Success(_vk.MapMemory(_device, stagingBufferMemory, 0, imageSize, 0, &data));
                Unsafe.CopyBlock(data, pixelBuffer, imageSize);
                _vk.UnmapMemory(_device, stagingBufferMemory);
            }

            textureImage.Dispose();
            
            CreateImage(textureWidth, textureHeight, _mipLevels, Format.R8G8B8A8Srgb, ImageTiling.Optimal, ImageUsageFlags.ImageUsageTransferSrcBit | ImageUsageFlags.ImageUsageTransferDstBit | ImageUsageFlags.ImageUsageSampledBit, MemoryPropertyFlags.MemoryPropertyDeviceLocalBit, out _textureImage, out _textureImageMemory);
            
            TransitionImageLayout(_textureImage, Format.R8G8B8A8Srgb, ImageLayout.Undefined, ImageLayout.TransferDstOptimal, _mipLevels);
            
            CopyBufferToImage(stagingBuffer, _textureImage, textureWidth, textureHeight);

            _vk.DestroyBuffer(_device, stagingBuffer, null);
            _vk.FreeMemory(_device, stagingBufferMemory, null);
            
            GenerateMipmaps(_textureImage, Format.R8G8B8A8Srgb, (int)textureWidth, (int)textureHeight, _mipLevels);
        }

        private void GenerateMipmaps(Image image, Format imageFormat, int texWidth, int texHeight, uint mipLevels)
        {
            FormatProperties formatProperties;
            _vk.GetPhysicalDeviceFormatProperties(_physicalDevice, imageFormat, &formatProperties);

            if ((formatProperties.OptimalTilingFeatures & FormatFeatureFlags.FormatFeatureSampledImageFilterLinearBit) == 0)
            {
                throw new InvalidOperationException("Texture image format does not support linear blitting");
            }
            
            ExecuteSingleTimeCommands(commandBuffer =>
            {
                var barrier = new ImageMemoryBarrier(
                    image: image,
                    srcQueueFamilyIndex: Vk.QueueFamilyIgnored,
                    dstQueueFamilyIndex: Vk.QueueFamilyIgnored,
                    subresourceRange: new ImageSubresourceRange(
                        aspectMask: ImageAspectFlags.ImageAspectColorBit,
                        baseArrayLayer: 0,
                        layerCount: 1,
                        levelCount: 1
                    )
                );

                var mipWidth = texWidth;
                var mipHeight = texHeight;

                for (uint i = 1; i < mipLevels; ++i)
                {
                    barrier.SubresourceRange.BaseMipLevel = i - 1;
                    barrier.OldLayout = ImageLayout.TransferDstOptimal;
                    barrier.NewLayout = ImageLayout.TransferSrcOptimal;
                    barrier.SrcAccessMask = AccessFlags.AccessTransferWriteBit;
                    barrier.DstAccessMask = AccessFlags.AccessTransferReadBit;
                        
                    _vk.CmdPipelineBarrier(commandBuffer, PipelineStageFlags.PipelineStageTransferBit, PipelineStageFlags.PipelineStageTransferBit, 0, 0, null, 0, null, 1, &barrier);

                    var blit = new ImageBlit
                        {
                            SrcOffsets =
                                {
                                    [0] = new Offset3D(0, 0, 0),
                                    [1] = new Offset3D(mipWidth, mipHeight, 1)
                                },
                            SrcSubresource = new ImageSubresourceLayers(ImageAspectFlags.ImageAspectColorBit, i - 1, 0, 1),
                            DstOffsets =
                                {
                                    [0] = new Offset3D(0, 0, 0),
                                    [1] = new Offset3D(mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1)
                                },
                            DstSubresource = new ImageSubresourceLayers(ImageAspectFlags.ImageAspectColorBit, i, 0, 1)
                        };


                    _vk.CmdBlitImage(commandBuffer, image, ImageLayout.TransferSrcOptimal, image, ImageLayout.TransferDstOptimal, 1, &blit, Filter.Linear);

                    barrier.OldLayout = ImageLayout.TransferSrcOptimal;
                    barrier.NewLayout = ImageLayout.ShaderReadOnlyOptimal;
                    barrier.SrcAccessMask = AccessFlags.AccessTransferReadBit;
                    barrier.DstAccessMask = AccessFlags.AccessShaderReadBit;
                    
                    _vk.CmdPipelineBarrier(commandBuffer, PipelineStageFlags.PipelineStageTransferBit, PipelineStageFlags.PipelineStageFragmentShaderBit, 0, 0, null, 0, null, 1, &barrier);

                    if (mipWidth > 1)
                    {
                        mipWidth /= 2;
                    }
                    
                    if (mipHeight > 1)
                    {
                        mipHeight /= 2;
                    }
                }

                barrier.SubresourceRange.BaseMipLevel = mipLevels - 1;
                barrier.OldLayout = ImageLayout.TransferDstOptimal;
                barrier.NewLayout = ImageLayout.ShaderReadOnlyOptimal;
                barrier.SrcAccessMask = AccessFlags.AccessTransferWriteBit;
                barrier.DstAccessMask = AccessFlags.AccessShaderReadBit;
                
                _vk.CmdPipelineBarrier(commandBuffer, PipelineStageFlags.PipelineStageTransferBit, PipelineStageFlags.PipelineStageFragmentShaderBit, 0, 0, null, 0, null, 1, &barrier);
            });
        }

        private void CopyBufferToImage(Buffer buffer, Image image, uint width, uint height)
        {
            ExecuteSingleTimeCommands(commandBuffer =>
            {
                var region = new BufferImageCopy(
                    bufferOffset: 0,
                    bufferRowLength: 0,
                    bufferImageHeight: 0,
                    imageSubresource: new ImageSubresourceLayers(
                        aspectMask: ImageAspectFlags.ImageAspectColorBit,
                        mipLevel: 0,
                        baseArrayLayer: 0,
                        layerCount: 1
                    ),
                    imageOffset: new Offset3D(0, 0, 0),
                    imageExtent: new Extent3D(width, height, 1)
                );
                
                _vk.CmdCopyBufferToImage(commandBuffer, buffer, image, ImageLayout.TransferDstOptimal, 1, &region);
            });
        }

        private void TransitionImageLayout(Image image, Format format, ImageLayout oldLayout, ImageLayout newLayout, uint mipLevels)
        {
            ExecuteSingleTimeCommands(commandBuffer =>
            {
                var imageMemoryBarrier = new ImageMemoryBarrier(
                    oldLayout: oldLayout,
                    newLayout: newLayout,
                    srcQueueFamilyIndex: Vk.QueueFamilyIgnored,
                    dstQueueFamilyIndex: Vk.QueueFamilyIgnored,
                    image: image,
                    subresourceRange: new ImageSubresourceRange(
                        aspectMask: ImageAspectFlags.ImageAspectColorBit,
                        baseMipLevel: 0,
                        levelCount: mipLevels,
                        baseArrayLayer: 0,
                        layerCount: 1
                    )
                );
                
                PipelineStageFlags sourceStage;
                PipelineStageFlags destinationStage;

                if (oldLayout == ImageLayout.Undefined && newLayout == ImageLayout.TransferDstOptimal)
                {
                    imageMemoryBarrier.SrcAccessMask = 0;
                    imageMemoryBarrier.DstAccessMask = AccessFlags.AccessTransferWriteBit;

                    sourceStage = PipelineStageFlags.PipelineStageTopOfPipeBit;
                    destinationStage = PipelineStageFlags.PipelineStageTransferBit;
                }
                else if (oldLayout == ImageLayout.TransferDstOptimal && newLayout == ImageLayout.ShaderReadOnlyOptimal)
                {
                    imageMemoryBarrier.SrcAccessMask = AccessFlags.AccessTransferWriteBit;
                    imageMemoryBarrier.DstAccessMask = AccessFlags.AccessShaderReadBit;

                    sourceStage = PipelineStageFlags.PipelineStageTransferBit;
                    destinationStage = PipelineStageFlags.PipelineStageFragmentShaderBit;
                }
                else
                {
                    throw new InvalidOperationException("Unsupported layout transition");
                }
                
                _vk.CmdPipelineBarrier(
                    commandBuffer: commandBuffer,
                    srcStageMask: sourceStage,
                    dstStageMask: destinationStage,
                    dependencyFlags: 0,
                    memoryBarrierCount: 0,
                    pMemoryBarriers: null,
                    bufferMemoryBarrierCount: 0,
                    pBufferMemoryBarriers: null,
                    imageMemoryBarrierCount: 1,
                    pImageMemoryBarriers: &imageMemoryBarrier
                );
            });
        }

        private void CreateImage(uint width, uint height, uint mipLevels, Format format, ImageTiling tiling, ImageUsageFlags usage, MemoryPropertyFlags properties, out Image image, out DeviceMemory imageMemory)
        {
            var imageInfo = new ImageCreateInfo(
                imageType: ImageType.ImageType2D,
                extent: new Extent3D(width, height, 1),
                mipLevels: mipLevels,
                arrayLayers: 1,
                format: format,
                tiling: tiling,
                initialLayout: ImageLayout.Undefined,
                usage: usage,
                sharingMode: SharingMode.Exclusive,
                samples: SampleCountFlags.SampleCount1Bit,
                flags: 0
            );
            
            VkCheck.Success(_vk.CreateImage(_device, &imageInfo, null, out image), "Failed to create image");

            MemoryRequirements memoryRequirements;
            _vk.GetImageMemoryRequirements(_device, image, &memoryRequirements);

            var allocInfo = new MemoryAllocateInfo(
                allocationSize: memoryRequirements.Size,
                memoryTypeIndex: FindMemoryType(memoryRequirements.MemoryTypeBits, properties)
            );
            
            VkCheck.Success(_vk.AllocateMemory(_device, &allocInfo, null, out imageMemory), "Failed to allocate image memory");

            _vk.BindImageMemory(_device, image, imageMemory, 0);
        }

        private void CreateTextureImageView()
        {
            _textureImageView = CreateImageView(_textureImage, Format.R8G8B8A8Srgb, ImageAspectFlags.ImageAspectColorBit, _mipLevels);
        }

        private ImageView CreateImageView(Image image, Format format, ImageAspectFlags aspectFlags, uint mipLevel)
        {
            var viewInfo = new ImageViewCreateInfo(
                image: image,
                viewType: ImageViewType.ImageViewType2D,
                format: format,
                subresourceRange: new ImageSubresourceRange(
                    aspectMask: aspectFlags,
                    baseMipLevel: 0,
                    levelCount: mipLevel,
                    baseArrayLayer: 0,
                    layerCount: 1
                )
            );

            ImageView imageView;
            VkCheck.Success(_vk.CreateImageView(_device, &viewInfo, null, &imageView), "Failed to create image view");
            return imageView;
        }

        private void CreateTextureSampler()
        {
            var samplerInfo = new SamplerCreateInfo(
                magFilter: Filter.Linear,
                minFilter: Filter.Linear,
                addressModeU: SamplerAddressMode.Repeat,
                addressModeV: SamplerAddressMode.Repeat,
                addressModeW: SamplerAddressMode.Repeat,
                anisotropyEnable: _cachedDeviceInfo.MaxSamplerAnisotropy.HasValue,
                maxAnisotropy: _cachedDeviceInfo.MaxSamplerAnisotropy,
                borderColor: BorderColor.IntOpaqueBlack,
                unnormalizedCoordinates: Vk.False,
                compareEnable: Vk.False,
                compareOp: CompareOp.Always,
                mipmapMode: SamplerMipmapMode.Linear,
                mipLodBias: 0f,
                minLod: 0f,
                maxLod: (float)_mipLevels
            );
            
            VkCheck.Success(_vk.CreateSampler(_device, &samplerInfo, null, out _textureSampler), "Failed to create texture sampler");
        }

        private void LoadModel()
        {
            var indices = new List<ushort>();
            var vertices = new List<Vertex>();
            var verticesMap = new Dictionary<Vector3, ushort>();

            var assimp = Assimp.GetApi();
            var scene = assimp.ImportFile(Path.Join(Directory.GetCurrentDirectory(), "Models", "viking_room.obj"), (uint) (PostProcessSteps.Triangulate | PostProcessSteps.FlipUVs));

            if (scene->MNumMeshes == 1)
            {
                ref var mesh = ref scene->MMeshes[0];

                for (uint iFace = 0; iFace < mesh->MNumFaces; ++iFace)
                {
                    ref var face = ref mesh->MFaces[iFace];

                    for (uint iIndex = 0; iIndex < face.MNumIndices; ++iIndex)
                    {
                        var index = (ushort)face.MIndices[iIndex];
                        ref var vertex = ref mesh->MVertices[index];

                        if (!verticesMap.TryGetValue(vertex, out var position))
                        {
                            position = (ushort) vertices.Count;
                            verticesMap.Add(vertex, position);
                            
                            ref var textureCoords = ref mesh->MTextureCoords[0][index];

                            vertices.Add(new Vertex
                                {
                                    Pos = new Vector3D<float>(vertex.X, vertex.Y, vertex.Z),
                                    TexCoord = new Vector2D<float>(textureCoords.X, textureCoords.Y),
                                    Color = new Vector3D<float>(1f, 1f, 1f)
                                });
                        }
                        
                        indices.Add(position);
                    }
                }

                _vertices = vertices.ToArray();
                _indices = indices.ToArray();
            }
            
            assimp.FreeScene(scene);
        }

        private void CreateVertexBuffer()
        {
            var bufferLength = (uint)(sizeof(Vertex) * _vertices.Length);
            
            CreateBuffer(bufferLength, BufferUsageFlags.BufferUsageTransferSrcBit, MemoryPropertyFlags.MemoryPropertyHostVisibleBit | MemoryPropertyFlags.MemoryPropertyHostCoherentBit, out var stagingBuffer, out var stagingBufferMemory);
            
            void* data;
            VkCheck.Success(_vk.MapMemory(_device, stagingBufferMemory, 0, bufferLength, 0, &data));

            fixed (Vertex* vertexData = _vertices)
            {
                //TODO: alignment requirement?
                Unsafe.CopyBlock(data, vertexData, bufferLength);
            }
            
            _vk.UnmapMemory(_device, stagingBufferMemory);
            
            CreateBuffer(bufferLength, BufferUsageFlags.BufferUsageTransferDstBit | BufferUsageFlags.BufferUsageVertexBufferBit, MemoryPropertyFlags.MemoryPropertyDeviceLocalBit, out _vertexBuffer, out _vertexBufferMemory);
            
            CopyBuffer(stagingBuffer, _vertexBuffer, bufferLength);
            
            _vk.DestroyBuffer(_device, stagingBuffer, null);
            _vk.FreeMemory(_device, stagingBufferMemory, null);
        }

        private void CreateIndexBuffer()
        {
            var bufferSize = (uint) (sizeof(ushort) * _indices.Length);
            
            CreateBuffer(bufferSize, BufferUsageFlags.BufferUsageTransferSrcBit, MemoryPropertyFlags.MemoryPropertyHostVisibleBit | MemoryPropertyFlags.MemoryPropertyHostCoherentBit, out var stagingBuffer, out var stagingBufferMemory);
            
            void* data;
            VkCheck.Success(_vk.MapMemory(_device, stagingBufferMemory, 0, bufferSize, 0, &data));

            fixed (ushort* indexData = _indices)
            {
                //TODO: alignment requirement?
                Unsafe.CopyBlock(data, indexData, bufferSize);
            }
            
            _vk.UnmapMemory(_device, stagingBufferMemory);
            
            CreateBuffer(bufferSize, BufferUsageFlags.BufferUsageTransferDstBit | BufferUsageFlags.BufferUsageIndexBufferBit, MemoryPropertyFlags.MemoryPropertyDeviceLocalBit, out _indexBuffer, out _indexBufferMemory);
            
            CopyBuffer(stagingBuffer, _indexBuffer, bufferSize);
            
            _vk.DestroyBuffer(_device, stagingBuffer, null);
            _vk.FreeMemory(_device, stagingBufferMemory, null);
        }

        private void ExecuteSingleTimeCommands(Action<CommandBuffer> action)
        {
            var allocateInfo = new CommandBufferAllocateInfo(
                level: CommandBufferLevel.Primary,
                commandPool: _commandPool,
                commandBufferCount: 1
            );

            CommandBuffer commandBuffer;
            
            VkCheck.Success(_vk.AllocateCommandBuffers(_device, &allocateInfo, &commandBuffer));

            var beginInfo = new CommandBufferBeginInfo(
                flags: CommandBufferUsageFlags.CommandBufferUsageOneTimeSubmitBit
            );

            _vk.BeginCommandBuffer(commandBuffer, &beginInfo);

            action(commandBuffer);

            VkCheck.Success(_vk.EndCommandBuffer(commandBuffer));

            var submitInfo = new SubmitInfo(
                commandBufferCount: 1,
                pCommandBuffers: &commandBuffer
            );

            _vk.QueueSubmit(_graphicsQueue, 1, &submitInfo, default);
            _vk.QueueWaitIdle(_graphicsQueue);
            
            _vk.FreeCommandBuffers(_device, _commandPool, 1, &commandBuffer);
        }

        private void CopyBuffer(Buffer srcBuffer, Buffer dstBuffer, uint size)
        {
            ExecuteSingleTimeCommands(commandBuffer =>
            {
                var copyRegion = new BufferCopy(0, 0, size);
                _vk.CmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
            });
        }

        private uint FindMemoryType(uint typeFilter, MemoryPropertyFlags properties)
        {
            PhysicalDeviceMemoryProperties memoryProperties;
            _vk.GetPhysicalDeviceMemoryProperties(_physicalDevice, &memoryProperties);

            for (uint i = 0; i < memoryProperties.MemoryTypeCount; ++i)
            {
                if ((typeFilter & (1 << (int) i)) != 0 && (memoryProperties.MemoryTypes[(int)i].PropertyFlags & properties) == properties)
                {
                    return i;
                }
            }

            throw new InvalidOperationException("Failed to allocate vertex buffer memory");
        }

        private void CreateUniformBuffer()
        {
            var bufferSize = (uint) sizeof(UniformBufferObject);

            _uniformBuffers = new Buffer[_swapChainImages.Length];
            _uniformBuffersMemory = new DeviceMemory[_swapChainImages.Length];

            for (var i = 0; i < _swapChainImages.Length; ++i)
            {
                CreateBuffer(bufferSize, BufferUsageFlags.BufferUsageUniformBufferBit, MemoryPropertyFlags.MemoryPropertyHostVisibleBit | MemoryPropertyFlags.MemoryPropertyHostCoherentBit, out _uniformBuffers[i], out _uniformBuffersMemory[i]);
            }
        }

        private void CreateDescriptorPool()
        {
            const int numPoolSizes = 2;

            var poolSizes = stackalloc DescriptorPoolSize[numPoolSizes];
            
            poolSizes[0] = new DescriptorPoolSize(
                type: DescriptorType.UniformBuffer,
                descriptorCount: (uint)_swapChainImages.Length
            );
            
            poolSizes[1] = new DescriptorPoolSize(
                type: DescriptorType.CombinedImageSampler,
                descriptorCount: (uint)_swapChainImages.Length
            );

            var poolInfo = new DescriptorPoolCreateInfo(
                poolSizeCount: numPoolSizes,
                pPoolSizes: poolSizes,
                maxSets: (uint)_swapChainImages.Length
            );
            
            VkCheck.Success(_vk.CreateDescriptorPool(_device, &poolInfo, null, out _descriptorPool), "Failed to create descriptor pool");
        }

        private void CreateDescriptorSets()
        {
            _descriptorSets = new DescriptorSet[_swapChainImages.Length];

            var layouts = stackalloc DescriptorSetLayout[_swapChainImages.Length];

            for (var i = 0; i < _swapChainImages.Length; ++i)
            {
                layouts[i] = _descriptorSetLayout;
            }

            var allocInfo = new DescriptorSetAllocateInfo(
                descriptorPool: _descriptorPool,
                descriptorSetCount: (uint)_swapChainImages.Length,
                pSetLayouts: layouts
            );

            fixed (DescriptorSet* descriptorSetsPtr = _descriptorSets)
            {
                VkCheck.Success(_vk.AllocateDescriptorSets(_device, &allocInfo, descriptorSetsPtr), "Failed to allocate descriptor sets");
            }
            
            //Will be reused in the loop
            const int numDescriptorWrites = 2;
            var descriptorWrites = stackalloc WriteDescriptorSet[numDescriptorWrites];

            for (var i = 0; i < _swapChainImages.Length; ++i)
            {
                var bufferInfo = new DescriptorBufferInfo(
                    buffer: _uniformBuffers[i],
                    offset: 0,
                    range: (uint)sizeof(UniformBufferObject)
                );

                var imageInfo = new DescriptorImageInfo(
                    imageLayout: ImageLayout.ShaderReadOnlyOptimal,
                    imageView: _textureImageView,
                    sampler: _textureSampler
                );

                descriptorWrites[0] = new WriteDescriptorSet(
                    dstSet: _descriptorSets[i],
                    dstBinding: 0,
                    dstArrayElement: 0,
                    descriptorType: DescriptorType.UniformBuffer,
                    descriptorCount: 1,
                    pBufferInfo: &bufferInfo
                );
                
                descriptorWrites[1] = new WriteDescriptorSet(
                    dstSet: _descriptorSets[i],
                    dstBinding: 1,
                    dstArrayElement: 0,
                    descriptorType: DescriptorType.CombinedImageSampler,
                    descriptorCount: 1,
                    pImageInfo: &imageInfo
                );

                _vk.UpdateDescriptorSets(_device, numDescriptorWrites, descriptorWrites, 0, null);
            }
        }
        
        private void CreateCommandBuffers()
        {
            _commandBuffers = new CommandBuffer[_swapChainFramebuffers.Length];

            var allocInfo = new CommandBufferAllocateInfo(
                commandPool: _commandPool,
                level: CommandBufferLevel.Primary,
                commandBufferCount: (uint)_commandBuffers.Length
            );

            fixed (CommandBuffer* commandBuffersPtr = _commandBuffers)
            {
                VkCheck.Success(_vk.AllocateCommandBuffers(_device, &allocInfo, commandBuffersPtr), "Failed to allocate command buffers");

                const int numClearValues = 2;
                var clearValues = stackalloc ClearValue[numClearValues];

                for (var i = 0; i < _commandBuffers.Length; ++i)
                {
                    var beginInfo = new CommandBufferBeginInfo(
                        flags: 0,
                        pInheritanceInfo: null
                    );
                    
                    VkCheck.Success(_vk.BeginCommandBuffer(commandBuffersPtr[i], &beginInfo), "Failed to begin recording command buffer");

                    clearValues[0].Color = new ClearColorValue {Float32_0 = 0, Float32_1 = 0, Float32_2 = 0, Float32_3 = 1};
                    clearValues[1].DepthStencil = new ClearDepthStencilValue(1.0f, 0);

                    var renderPassInfo = new RenderPassBeginInfo(
                        renderPass: _renderPass,
                        framebuffer: _swapChainFramebuffers[i],
                        renderArea: new Rect2D(new Offset2D(0, 0), _swapChainExtent),
                        clearValueCount: numClearValues,
                        pClearValues: clearValues
                    );
                    
                    _vk.CmdBeginRenderPass(commandBuffersPtr[i], &renderPassInfo, SubpassContents.Inline);
                    
                    _vk.CmdBindPipeline(commandBuffersPtr[i], PipelineBindPoint.Graphics, _graphicsPipeline);

                    ulong offset = 0;
                    _vk.CmdBindVertexBuffers(commandBuffersPtr[i], 0, 1, in _vertexBuffer, &offset);
                    _vk.CmdBindIndexBuffer(commandBuffersPtr[i], _indexBuffer, 0, IndexType.Uint16);
                    
                    _vk.CmdBindDescriptorSets(commandBuffersPtr[i], PipelineBindPoint.Graphics, _pipelineLayout, 0, 1, _descriptorSets[i], 0, null);
                    
                    _vk.CmdDrawIndexed(commandBuffersPtr[i], (uint)_indices.Length, 1, 0, 0, 0);
                    
                    _vk.CmdEndRenderPass(commandBuffersPtr[i]);
                    
                    VkCheck.Success(_vk.EndCommandBuffer(commandBuffersPtr[i]), "Failed to record command buffer");
                }
            }
        }
        
        private void CreateSyncObjects()
        {
            _imagesInFlight = new Fence[_swapChainImages.Length];
            
            var semaphoreInfo = new SemaphoreCreateInfo(
                flags: null
            );

            var fenceInfo = new FenceCreateInfo(
                flags: FenceCreateFlags.FenceCreateSignaledBit
            );
            
            for (var i = 0; i < MaxFramesInFlight; ++i)
            {
                VkCheck.Success(_vk.CreateSemaphore(_device, &semaphoreInfo, null, out _imageAvailableSemaphores[i]), "Failed to create image available semaphore");
                VkCheck.Success(_vk.CreateSemaphore(_device, &semaphoreInfo, null, out _renderFinishedSemaphores[i]), "Failed to create render finished semaphore");
                VkCheck.Success(_vk.CreateFence(_device, &fenceInfo, null, out _inFlightFences[i]), "Failed to create fence");
            }
        }
        
        protected override void Render(double _)
        {
            if (_pauseRendering)
            {
                if (_framebufferResized)
                {
                    _framebufferResized = true;
                    AssembleSwapChain(true);
                }
                return;
            }
            
            var inFlightFence = _inFlightFences[_currentFrame]; 
            
            VkCheck.Success(_vk.WaitForFences(_device, 1, &inFlightFence, Vk.True, uint.MaxValue));

            uint imageIndex;
            var result = _extKhrSwapChain.AcquireNextImage(_device, _swapChain, ulong.MaxValue, _imageAvailableSemaphores[_currentFrame], default, &imageIndex);

            if (result == Result.ErrorOutOfDateKhr)
            {
                AssembleSwapChain(true);
                return;
            }
            if (result != Result.Success && result != Result.SuboptimalKhr)
            {
                throw new InvalidOperationException("Failed to acquire swap chain image");
            }

            var imageInFlight = _imagesInFlight[imageIndex];
            if (imageInFlight.Handle != 0)
            {
                VkCheck.Success(_vk.WaitForFences(_device, 1, &imageInFlight, Vk.True, uint.MaxValue));
            }

            _imagesInFlight[imageIndex] = inFlightFence;

            var semaphore = _imageAvailableSemaphores[_currentFrame];
            var signalSemaphore = _renderFinishedSemaphores[_currentFrame];

            var waitStage = PipelineStageFlags.PipelineStageColorAttachmentOutputBit;

            var commandBuffer = _commandBuffers[imageIndex];
            
            UpdateUniformBuffer(imageIndex);
            
            var submitInfo = new SubmitInfo(
                waitSemaphoreCount: 1,
                pWaitSemaphores: &semaphore,
                pWaitDstStageMask: &waitStage,
                commandBufferCount: 1,
                pCommandBuffers: &commandBuffer,
                signalSemaphoreCount: 1,
                pSignalSemaphores: &signalSemaphore
            );
            
            VkCheck.Success(_vk.ResetFences(_device, 1, &inFlightFence));
            
            VkCheck.Success(_vk.QueueSubmit(_graphicsQueue, 1, &submitInfo, inFlightFence), "Failed to submit draw command buffer");

            var swapChain = _swapChain;
            
            var presentInfo = new PresentInfoKHR(
                waitSemaphoreCount: 1,
                pWaitSemaphores: &signalSemaphore,
                swapchainCount: 1,
                pSwapchains: &swapChain,
                pImageIndices: &imageIndex,
                pResults: null
            );
            
            result = _extKhrSwapChain.QueuePresent(_presentQueue, &presentInfo);

            if (result is Result.ErrorOutOfDateKhr or Result.SuboptimalKhr || _framebufferResized)
            {
                _framebufferResized = false;
                
                AssembleSwapChain(true);
            }
            else if (result is not Result.Success)
            {
                throw new InvalidOperationException("Failed to present swap chain image");
            }

            _currentFrame = (_currentFrame + 1) % MaxFramesInFlight;
        }

        private void UpdateUniformBuffer(uint imageIndex)
        {
            UniformBufferObject ubo;

            ubo.Model = Matrix4X4.CreateRotationZ((float)TotalTime * 1.5708f);
            ubo.View = Matrix4X4.CreateLookAt(new Vector3D<float>(2f, 2f, 2f), Vector3D<float>.Zero, new Vector3D<float>(0f, 0f, 1f));
            ubo.Proj = Matrix4X4.CreatePerspectiveFieldOfView(0.785398f, _swapChainExtent.Width / (float) _swapChainExtent.Height, 0.1f, 10f);

            ubo.Proj.M22 *= -1f;

            void* data;
            VkCheck.Success(_vk.MapMemory(_device, _uniformBuffersMemory[imageIndex], 0, (ulong)sizeof(UniformBufferObject), 0, &data));

            Unsafe.CopyBlock(data, &ubo, (uint)sizeof(UniformBufferObject));
            
            _vk.UnmapMemory(_device, _uniformBuffersMemory[imageIndex]);
        }

        private void CleanupSwapChain()
        {
            //Check if swap chain exists
            if (_swapChainFramebuffers == null)
            {
                return;
            }
            
            _vk.DeviceWaitIdle(_device);
            
            _vk.DestroyImageView(_device, _depthImageView, null);
            _vk.DestroyImage(_device, _depthImage, null);
            _vk.FreeMemory(_device, _depthImageMemory, null);
            
            foreach (var swapChainFramebuffer in _swapChainFramebuffers)
            {
                _vk.DestroyFramebuffer(_device, swapChainFramebuffer, null);
            }
            _swapChainFramebuffers = null;

            fixed (CommandBuffer* commandBuffersPtr = _commandBuffers)
            {
                _vk.FreeCommandBuffers(_device, _commandPool, (uint)_commandBuffers.Length, commandBuffersPtr);
            }
            _commandBuffers = null;

            for (var i = 0; i < _swapChainImages.Length; ++i)
            {
                _vk.DestroyBuffer(_device, _uniformBuffers[i], null);
                _vk.FreeMemory(_device, _uniformBuffersMemory[i], null);
            }
            _uniformBuffers = null;
            _uniformBuffersMemory = null;
            
            _vk.DestroyDescriptorPool(_device, _descriptorPool, null);
            _descriptorPool = default;

            _vk.DestroyPipeline(_device, _graphicsPipeline, null);
            _graphicsPipeline = default;
            
            _vk.DestroyPipelineLayout(_device, _pipelineLayout, null);
            _pipelineLayout = default;
            
            _vk.DestroyRenderPass(_device, _renderPass, null);
            _renderPass = default;
            
            foreach (var swapChainImageView in _swapChainImageViews)
            {
                _vk.DestroyImageView(_device, swapChainImageView, null);
            }
            _swapChainImageViews = null;
            
            _extKhrSwapChain.DestroySwapchain(_device, _swapChain, null);
            _swapChain = default;
        }
        
        protected override void OnClose()
        {
            CleanupSwapChain();
            
            _vk.DestroySampler(_device, _textureSampler, null);
            
            _vk.DestroyImageView(_device, _textureImageView, null);
            
            _vk.DestroyImage(_device, _textureImage, null);
            _vk.FreeMemory(_device, _textureImageMemory, null);
            
            _vk.DestroyDescriptorSetLayout(_device, _descriptorSetLayout, null);

            foreach (var renderFinishedSemaphore in _renderFinishedSemaphores)
            {
                _vk.DestroySemaphore(_device, renderFinishedSemaphore, null);
            }
            foreach (var imageAvailableSemaphore in _imageAvailableSemaphores)
            {
                _vk.DestroySemaphore(_device, imageAvailableSemaphore, null);
            }

            foreach (var inFlightFence in _inFlightFences)
            {
                _vk.DestroyFence(_device, inFlightFence, null);
            }
            
            _vk.DestroyBuffer(_device, _indexBuffer, null);
            _vk.FreeMemory(_device, _indexBufferMemory, null);
            
            _vk.DestroyBuffer(_device, _vertexBuffer, null);
            _vk.FreeMemory(_device, _vertexBufferMemory, null);

            _vk.DestroyCommandPool(_device, _commandPool, null);

            _vk.DestroyDevice(_device, null);
            
            _extKhrSurface.DestroySurface(_instance, _surface, null);
            
            if (_debugMessenger.Handle != 0)
            {
                _extDebugUtils.DestroyDebugUtilsMessenger(_instance, _debugMessenger, null);
            }
            
            _vk.DestroyInstance(_instance, null);
            
            base.OnClose();
        }

        protected override void OnResize(Vector2D<int> obj)
        {
            base.OnResize(obj);

            _framebufferResized = true;
        }
    }
}
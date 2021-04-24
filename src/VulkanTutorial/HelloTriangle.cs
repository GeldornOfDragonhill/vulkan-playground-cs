using System;
using System.Collections.Generic;
using System.Linq;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.EXT;
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

        private readonly bool _enableValidationLayers;
        private readonly Vk _vk = Vk.GetApi();

        private ExtDebugUtils _extDebugUtils;

        private Instance _instance;
        private DebugUtilsMessengerEXT _debugMessenger;
        private PhysicalDevice _physicalDevice;
        private Device _device;
        private Queue _graphicsQueue;

        public HelloTriangle(bool enableValidationLayers)
        {
            _enableValidationLayers = enableValidationLayers;
        }

        private void InitVulkan()
        {
            CreateVulkanInstance();
            SetupDebugMessenger();
            PickPhysicalDevice();
            CreateLogicalDevice();
        }

        private void CreateLogicalDevice()
        {
            var queueFamilyIndices = FindQueueFamilies(_physicalDevice);

            var queuePriority = 1.0f;
            
            var deviceQueueCreateInfo = new DeviceQueueCreateInfo(
                queueFamilyIndex: queueFamilyIndices.GraphicsFamily.Value,
                queueCount: 1,
                pQueuePriorities: &queuePriority
            );

            var deviceFeatures = new PhysicalDeviceFeatures();

            using var validationLayers = new StringArrayPtrWrapper(_enableValidationLayers ? ValidationLayers : Array.Empty<string>());

            var deviceCreateInfo = new DeviceCreateInfo(
                pQueueCreateInfos: &deviceQueueCreateInfo,
                queueCreateInfoCount: 1,
                pEnabledFeatures: &deviceFeatures,
                enabledLayerCount: (uint)validationLayers.Count,
                ppEnabledLayerNames: validationLayers.StringArrayPtr
            );
            
            VkCheck.Success(_vk.CreateDevice(_physicalDevice, &deviceCreateInfo, null, out _device), "Failed to create logical device");
            
            _vk.GetDeviceQueue(_device, queueFamilyIndices.GraphicsFamily.Value, 0, out _graphicsQueue);
        }

        private void PickPhysicalDevice()
        {
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
                var queueFamilyIndices = IsDeviceSuitable(current);
                if (queueFamilyIndices.HasValue)
                {
                    _physicalDevice = current;
                    return;
                }
            }

            throw new InvalidOperationException("Could not find a suitable GPU");
        }

        private QueueFamilyIndices? IsDeviceSuitable(in PhysicalDevice physicalDevice)
        {
            var indices = FindQueueFamilies(physicalDevice);

            return indices.IsComplete() ? indices : null;
        }

        private struct QueueFamilyIndices
        {
            public uint? GraphicsFamily;

            public bool IsComplete()
            {
                return GraphicsFamily.HasValue;
            }
        }

        private  QueueFamilyIndices FindQueueFamilies(in PhysicalDevice physicalDevice)
        {
            uint queueFamilyCount = 0;
            _vk.GetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, null);

            var queueFamilies = stackalloc QueueFamilyProperties[(int) queueFamilyCount];
            _vk.GetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies);

            QueueFamilyIndices queueFamilyIndices = new QueueFamilyIndices();

            for (uint i = 0; i < queueFamilyCount; ++i)
            {
                ref var queueFamily = ref queueFamilies[i];

                if ((queueFamily.QueueFlags & QueueFlags.QueueGraphicsBit) != 0)
                {
                    queueFamilyIndices.GraphicsFamily = i;
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
            _vk.DestroyDevice(_device, null);
            
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
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

        public HelloTriangle(bool enableValidationLayers)
        {
            _enableValidationLayers = enableValidationLayers;
        }

        private void InitVulkan()
        {
            CreateVulkanInstance();
            SetupDebugMessenger();
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
                messageSeverity: DebugUtilsMessageSeverityFlagsEXT.DebugUtilsMessageSeverityVerboseBitExt |
                                 DebugUtilsMessageSeverityFlagsEXT.DebugUtilsMessageSeverityInfoBitExt |
                                 DebugUtilsMessageSeverityFlagsEXT.DebugUtilsMessageSeverityWarningBitExt |
                                 DebugUtilsMessageSeverityFlagsEXT.DebugUtilsMessageSeverityErrorBitExt,
                messageType: DebugUtilsMessageTypeFlagsEXT.DebugUtilsMessageTypeGeneralBitExt |
                             DebugUtilsMessageTypeFlagsEXT.DebugUtilsMessageTypePerformanceBitExt |
                             DebugUtilsMessageTypeFlagsEXT.DebugUtilsMessageTypeValidationBitExt,
                pfnUserCallback: (DebugUtilsMessengerCallbackFunctionEXT)DebugCallback
            );
            
            VkCheck.Success(_extDebugUtils.CreateDebugUtilsMessenger(_instance, &createInfo, null, out _debugMessenger), "Could not create debug utils messenger");
        }
        
        private unsafe uint DebugCallback(DebugUtilsMessageSeverityFlagsEXT messageSeverity, DebugUtilsMessageTypeFlagsEXT messageTypes, DebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
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
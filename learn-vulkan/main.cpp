#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <cstdint>
#include <cstdlib>

#include <algorithm>
#include <array>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

// --- IO -----------------------------------------------------------------
// ------------------------------------------------------------------------
static std::vector<char> readFile(const std::string &fn) {
  // todo not sure if I fully understand this
  std::ifstream file(fn, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    static constexpr size_t s = 1024;
    char *buf = new char[s];
    sprintf_s(buf, s, "failed to open file! (%s)", fn.c_str());
    throw std::runtime_error(buf);
  }
  size_t fileSize = (size_t)file.tellg();
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();

  return buffer;
}

// --- Constants ----------------------------------------------------------
// ------------------------------------------------------------------------
const int WIDTH = 800;
const int HEIGHT = 600;
const int MAX_FRAMES_IN_FLIGHT = 20;

// --- ValidationLayers and DebugMessenger --------------------------------
// ------------------------------------------------------------------------

const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif // NDEBUG

bool checkValidationLayerSupport() {
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

  for (const char *layerName : validationLayers) {
    bool layerFound = false;

    for (const auto &layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }

    if (!layerFound) {
      return false;
    }
  }

  return true;
}

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks *pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}

VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              void *pUserData) {

  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

  return VK_FALSE;
}

void populateDebugMessengerCreateInfo(
    VkDebugUtilsMessengerCreateInfoEXT &createInfo) {
  createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  createInfo.pfnUserCallback = debugCallback;
}

// --- QueueFamily Helpers ------------------------------------------------
// ------------------------------------------------------------------------

struct QueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;
};

bool isComplete(QueueFamilyIndices *qfi) {
  return qfi->graphicsFamily.has_value() && qfi->presentFamily.has_value();
}

QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device,
                                     VkSurfaceKHR surface) {
  QueueFamilyIndices indices;

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                           queueFamilies.data());

  int i = 0;
  for (const auto &queueFamily : queueFamilies) {
    if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      indices.graphicsFamily = i;
    }

    VkBool32 presentSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
    if (presentSupport) {
      indices.presentFamily = i;
    }

    if (isComplete(&indices)) {
      break;
    }
    ++i;
  }

  return indices;
}

// --- Extensions ---------------------------------------------------------
// ------------------------------------------------------------------------

const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

std::vector<const char *> getRequiredExtensions() {
  uint32_t glfwExtensionCount = 0;
  const char **glfwExtensions;
  glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

  std::vector<const char *> extensions(glfwExtensions,
                                       glfwExtensions + glfwExtensionCount);

  if (enableValidationLayers) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  return extensions;
}

bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
  uint32_t extensionCount;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       nullptr);

  std::vector<VkExtensionProperties> availableExtensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       availableExtensions.data());

  std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                           deviceExtensions.end());
  for (const auto &ext : availableExtensions) {

    requiredExtensions.erase(ext.extensionName);
  }
  return requiredExtensions.empty();
}

// --- SwapChain Helpers --------------------------------------------------
// ------------------------------------------------------------------------

struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device,
                                              VkSurfaceKHR surface) {
  SwapChainSupportDetails det;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &det.capabilities);
  uint32_t formatCount;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
  if (formatCount != 0) {
    det.formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                         det.formats.data());
  }

  uint32_t modeCount;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &modeCount,
                                            nullptr);
  if (modeCount != 0) {
    det.presentModes.resize(modeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &modeCount,
                                              det.presentModes.data());
  }

  return det;
}

VkSurfaceFormatKHR
chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &available) {
  for (const auto &fmt : available) {
    if (fmt.format == VK_FORMAT_B8G8R8A8_SRGB &&
        fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return fmt;
    }
  }
  return available[0];
}

VkPresentModeKHR
chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &available) {
  for (const auto &pres : available) {
    if (pres == VK_PRESENT_MODE_MAILBOX_KHR) {
      return pres;
    }
  }
  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) {
  if (capabilities.currentExtent.width != UINT32_MAX) {
    return capabilities.currentExtent;
  } else {
    VkExtent2D actualExtent = {WIDTH, HEIGHT};
    actualExtent.width = std::max(
        capabilities.minImageExtent.width,
        std::min(capabilities.maxImageExtent.width, actualExtent.width));
    actualExtent.height = std::max(
        capabilities.minImageExtent.height,
        std::max(capabilities.maxImageExtent.height, actualExtent.height));

    return actualExtent;
  }
}

// --- DeviceSelection ----------------------------------------------------
// ------------------------------------------------------------------------

bool isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR surface) {
  // VkPhysicalDeviceProperties deviceProperties;
  // vkGetPhysicalDeviceProperties(device, &deviceProperties);
  // VkPhysicalDeviceFeatures deviceFeatures;
  // vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
  // return deviceProperties.deviceType ==
  // VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
  QueueFamilyIndices indices = findQueueFamilies(device, surface);

  // check to see if our extensions are supported
  bool allExtensionsSupported = checkDeviceExtensionSupport(device);

  // check swap chain details next
  bool swapChainOkay = false;
  if (allExtensionsSupported) {
    SwapChainSupportDetails det = querySwapChainSupport(device, surface);
    swapChainOkay = !(det.formats.empty() || det.presentModes.empty());
  }

  return isComplete(&indices) && allExtensionsSupported && swapChainOkay;
}

// --- Memory -------------------------------------------------------------
// ------------------------------------------------------------------------
uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                        VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkMemoryRequirements.html#_members
    // memoryTypeBits (typeFilter) is a bitmask and contains one bit set for
    // every supported memory type for the resource. Bit i is set if and only if
    // the memory type i in the VkPhysicalDeviceMemoryProperties structure for
    // the physical device is supported for the resource.
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("failed to find a suitable memory type!");
}

// --- Math ---------------------------------------------------------------
// ------------------------------------------------------------------------

struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;

  static VkVertexInputBindingDescription getBindingDescription() {
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
  }

  static std::array<VkVertexInputAttributeDescription, 2>
  getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, color);

    return attributeDescriptions;
  }
};

const std::vector<Vertex> vertices = {{{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
                                      {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
                                      {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}};

// --- HelloTriangleApplication -------------------------------------------
// ------------------------------------------------------------------------

class HelloTriangleApplication {
public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  void initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, resizeCallback);
  }

  static void resizeCallback(GLFWwindow *window, int width, int height) {
    ((HelloTriangleApplication *)glfwGetWindowUserPointer(window))
        ->framebufferResized = true;
  }

  void initVulkan() {
    createInstance();
    createDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createFramebuffers();
    createCommandPool();
    createVertexBuffer();
    createCommandBuffers();
    createSyncObjects();
  }

  void recreateSwapChain() {
    // idle while minimized
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
      glfwGetFramebufferSize(window, &width, &height);
      glfwWaitEvents();
    }
    // end idle

    vkDeviceWaitIdle(device);
    cleanupSwapChain();

    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createFramebuffers();
    createCommandBuffers();
  }

  void createSyncObjects() {
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightImages.resize(swapChainImages.size(), VK_NULL_HANDLE);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      if (vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                            &imageAvailableSemaphores[i]) != VK_SUCCESS ||
          vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                            &renderFinishedSemaphores[i]) != VK_SUCCESS ||
          vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) !=
              VK_SUCCESS) {
        throw std::runtime_error("failed to create sync objects!");
      }
    }
  }

  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usageFlags,
                    VkMemoryPropertyFlags propertyFlags, VkBuffer &buffer,
                    VkDeviceMemory &bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usageFlags;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create vertex buffer!");
    };

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, buffer, &memReq);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(physicalDevice, memReq.memoryTypeBits, propertyFlags);
    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to allocate vertex buffer memory!");
    }
    vkBindBufferMemory(device, buffer, bufferMemory, 0);
  }

  void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) const {
    // todo: create a separate command pool for temporary buffer creation
    // (like this one)
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmdBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &cmdBuffer);

    VkCommandBufferBeginInfo cmdBufferBegin{};
    cmdBufferBegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdBufferBegin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(cmdBuffer, &cmdBufferBegin);

    VkBufferCopy copyInfo{};
    copyInfo.size = size;
    copyInfo.srcOffset = 0;
    copyInfo.dstOffset = 0;
    vkCmdCopyBuffer(cmdBuffer, src, dst, 1, &copyInfo);

    vkEndCommandBuffer(cmdBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pCommandBuffers = &cmdBuffer;
    submitInfo.commandBufferCount = 1;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &cmdBuffer);
  }

  void createVertexBuffer() {
    VkDeviceSize bufSize = sizeof(vertices[0]) * vertices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMem;
    createBuffer(bufSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMem);
    void *data;
    vkMapMemory(device, stagingBufferMem, 0, bufSize, 0, &data);
    memcpy(data, vertices.data(), bufSize);
    vkUnmapMemory(device, stagingBufferMem);

    createBuffer(
        bufSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

    copyBuffer(stagingBuffer, vertexBuffer, bufSize);
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMem, nullptr);
  }

  void createCommandBuffers() {
    commandBuffers.resize(swapChainFramebuffers.size());
    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = commandPool;
    allocateInfo.commandBufferCount = (uint32_t)commandBuffers.size();
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    if (vkAllocateCommandBuffers(device, &allocateInfo,
                                 commandBuffers.data()) != VK_SUCCESS) {
      throw std::runtime_error("could not allocate command buffer storage!");
    }
    for (size_t i = 0; i < commandBuffers.size(); ++i) {
      VkCommandBufferBeginInfo beginInfo{};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginInfo.flags = 0;
      beginInfo.pInheritanceInfo = nullptr;
      if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to start recording command buffer!");
      }

      VkRenderPassBeginInfo renderPassInfo{};
      renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      renderPassInfo.renderPass = renderPass;
      renderPassInfo.framebuffer = swapChainFramebuffers[i];
      renderPassInfo.renderArea.offset = {0, 0};
      renderPassInfo.renderArea.extent = swapChainExtent;
      VkClearValue clearColor = {0.f, 0.f, 0.f, 1.f};
      renderPassInfo.clearValueCount = 1;
      renderPassInfo.pClearValues = &clearColor;

      vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo,
                           VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                        graphicsPipeline);
      VkBuffer vertexBuffers[] = {vertexBuffer};
      VkDeviceSize offsets[] = {0};
      vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);
      vkCmdDraw(commandBuffers[i], static_cast<uint32_t>(vertices.size()), 1, 0,
                0);
      vkCmdEndRenderPass(commandBuffers[i]);
      if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
      }
    }
  }

  void createCommandPool() {
    QueueFamilyIndices queueFamilyIndices =
        findQueueFamilies(physicalDevice, surface);
    VkCommandPoolCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    createInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    createInfo.flags = 0;
    if (vkCreateCommandPool(device, &createInfo, nullptr, &commandPool) !=
        VK_SUCCESS) {
      throw std::runtime_error("unable to create command pool!");
    }
  }

  void createFramebuffers() {
    swapChainFramebuffers.resize(swapChainImageViews.size());
    for (size_t i = 0; i < swapChainImageViews.size(); ++i) {
      VkImageView attachments[] = {swapChainImageViews[i]};
      VkFramebufferCreateInfo createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      createInfo.renderPass = renderPass;
      createInfo.attachmentCount = 1;
      createInfo.pAttachments = attachments;
      createInfo.height = swapChainExtent.height;
      createInfo.width = swapChainExtent.width;
      createInfo.layers = 1;
      if (vkCreateFramebuffer(device, &createInfo, nullptr,
                              &swapChainFramebuffers[i]) != VK_SUCCESS) {
        throw std::runtime_error("unable to create framebuffer!");
      }
    }
  }

  void createRenderPass() {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create the render pass");
    }
  }

  VkShaderModule createShaderModule(const std::vector<char> &code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create shader module!");
    }
    return shaderModule;
  }

  void createGraphicsPipeline() {
    auto vertShaderSPIRV = readFile("shaders_out/vert.spv");
    printf("read vert shader (%zdb)\n", vertShaderSPIRV.size());
    auto fragShaderSPIRV = readFile("shaders_out/frag.spv");
    printf("read frag shader (%zdb)\n", fragShaderSPIRV.size());

    VkShaderModule vertShaderModule = createShaderModule(vertShaderSPIRV);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderSPIRV);

    VkPipelineShaderStageCreateInfo vertShaderStageCreateInfo{};
    vertShaderStageCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageCreateInfo.module = vertShaderModule;
    vertShaderStageCreateInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageCreateInfo{};
    fragShaderStageCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageCreateInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageCreateInfo.module = fragShaderModule;
    fragShaderStageCreateInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {
        vertShaderStageCreateInfo, fragShaderStageCreateInfo};

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    VkVertexInputBindingDescription bindingDescription =
        Vertex::getBindingDescription();
    std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions =
        Vertex::getAttributeDescriptions();
    vertexInputInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyCreateInfo{};
    inputAssemblyCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyCreateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssemblyCreateInfo.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)swapChainExtent.width;
    viewport.height = (float)swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;

    VkPipelineViewportStateCreateInfo viewportCreateInfo{};
    viewportCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportCreateInfo.viewportCount = 1;
    viewportCreateInfo.pViewports = &viewport;
    viewportCreateInfo.scissorCount = 1;
    viewportCreateInfo.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasClamp = 0.0f;
    rasterizer.depthBiasSlopeFactor = 0.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

    // skipping depth and stencil buffers
    // VkPipelineDepthStencilStateCreateInfo

    // todo: the blending configuration is very flexible and
    // I'm not sure I fully grasp what's happening
    // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#framebuffer-logicop
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor =
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    // for changing things during runtime without recreating the pipeline
    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                      VK_DYNAMIC_STATE_LINE_WIDTH};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pSetLayouts = nullptr;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                               &pipelineLayout) != VK_SUCCESS) {
      throw std::runtime_error("failed to create the pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssemblyCreateInfo;
    pipelineInfo.pViewportState = &viewportCreateInfo;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = nullptr;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = nullptr;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                  nullptr, &graphicsPipeline) != VK_SUCCESS) {
      throw std::runtime_error("failed to create the graphics pipeline!");
    }

    vkDestroyShaderModule(device, vertShaderModule, nullptr);
    vkDestroyShaderModule(device, fragShaderModule, nullptr);
  }

  void createDebugMessenger() {
    if (!enableValidationLayers) {
      return;
    }

    VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
    populateDebugMessengerCreateInfo(createInfo);

    createInfo.pUserData = nullptr; // not using
    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr,
                                     &debugMessenger) != VK_SUCCESS) {
      throw std::runtime_error("failed to set up a debug messenger!");
    }
  }

  void createImageViews() {
    swapChainImageViews.resize(swapChainImages.size());
    for (size_t i = 0; i < swapChainImages.size(); ++i) {
      VkImageViewCreateInfo createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      createInfo.image = swapChainImages[i];
      createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
      createInfo.format = swapChainImageFormat;
      createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      createInfo.subresourceRange.baseMipLevel = 0;
      createInfo.subresourceRange.levelCount = 1;
      createInfo.subresourceRange.baseArrayLayer = 0;
      createInfo.subresourceRange.layerCount = 1;
      if (vkCreateImageView(device, &createInfo, nullptr,
                            &swapChainImageViews[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image view");
      }
    }
  }

  void createSwapChain() {
    SwapChainSupportDetails swapChainSupport =
        querySwapChainSupport(physicalDevice, surface);

    VkSurfaceFormatKHR surfaceFormat =
        chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode =
        chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
    // "simply sticking to this minimum means that we may sometimes have to wait
    // on the driver to complete internal operations before we can acquire
    // another image to render to. Therefore it is recommended to request at
    // least one more image than the minimum:"
    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilities.maxImageCount) {
      imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;

    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice, surface);
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(),
                                     indices.presentFamily.value()};

    if (indices.graphicsFamily != indices.presentFamily) {
      createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      createInfo.queueFamilyIndexCount = 2;
      createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
      createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
      createInfo.queueFamilyIndexCount = 0;
      createInfo.pQueueFamilyIndices = nullptr;
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount,
                            swapChainImages.data());

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
  }

  void createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice, surface);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(),
                                              indices.presentFamily.value()};
    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
      VkDeviceQueueCreateInfo queueCreateInfo{};
      queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueCreateInfo.queueFamilyIndex = queueFamily;
      queueCreateInfo.queueCount = 1;
      queueCreateInfo.pQueuePriorities = &queuePriority;
      queueCreateInfos.push_back(queueCreateInfo);
    }

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount =
        static_cast<uint32_t>(queueCreateInfos.size());
    VkPhysicalDeviceFeatures deviceFeatures{};
    createInfo.pEnabledFeatures = &deviceFeatures;

    createInfo.enabledExtensionCount =
        static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (enableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
      createInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
  }

  void pickPhysicalDevice() {
    uint32_t numDevices = 0;
    vkEnumeratePhysicalDevices(instance, &numDevices, nullptr);
    if (numDevices == 0) {
      throw std::runtime_error("failed to find any GPUs with Vulkan support!");
    }
    std::vector<VkPhysicalDevice> devices(numDevices);
    // do you always have to call these twice? so ugly...
    vkEnumeratePhysicalDevices(instance, &numDevices, devices.data());
    for (const auto &dev : devices) {
      if (isDeviceSuitable(dev, surface)) {
        physicalDevice = dev;
        break;
      }
    }
    if (physicalDevice == VK_NULL_HANDLE) {
      throw std::runtime_error("failed to find a suitable GPU!");
    }
  }

  void createInstance() {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
      throw std::runtime_error(
          "validation layers requested, but not available!");
    }

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0; // todo upgrade?

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = {};
    if (enableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();

      populateDebugMessengerCreateInfo(debugCreateInfo);
      createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&debugCreateInfo;
    } else {
      createInfo.enabledLayerCount = 0;
      createInfo.pNext = nullptr;
    }

    {
      // list extensions, this doesn't have anything to do with the instance
      // creation
      uint32_t extensionCount = 0;
      vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
      std::vector<VkExtensionProperties> extensionsProperties(extensionCount);
      vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
                                             extensionsProperties.data());
      puts("available extensions:");
      for (const auto &extension : extensionsProperties) {
        printf("\t%s\n", extension.extensionName);
      }
    }

    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
      throw std::runtime_error("failed to create instance!");
    }
  }

  void createSurface() {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create the window surface");
    }
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      drawFrame();
    }
    vkDeviceWaitIdle(device);
  }

  void drawFrame() {
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE,
                    UINT64_MAX);
    uint32_t imageIndex;
    VkResult nextImageResult = vkAcquireNextImageKHR(
        device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame],
        VK_NULL_HANDLE, &imageIndex);

    if (nextImageResult != VK_SUCCESS && nextImageResult != VK_SUBOPTIMAL_KHR) {
      throw new std::runtime_error("failed to acquire next swapchain image!");
    }

    if (inFlightImages[imageIndex] != VK_NULL_HANDLE) {
      vkWaitForFences(device, 1, &inFlightImages[imageIndex], VK_TRUE,
                      UINT64_MAX);
    }
    inFlightImages[imageIndex] = inFlightFences[currentFrame];

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
    VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    vkResetFences(device, 1, &inFlightFences[currentFrame]);
    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo,
                      inFlightFences[currentFrame]) != VK_SUCCESS) {
      throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    VkSwapchainKHR swapChains[] = {swapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;

    VkResult queuePresentationResult =
        vkQueuePresentKHR(presentQueue, &presentInfo);
    if (queuePresentationResult == VK_SUBOPTIMAL_KHR ||
        queuePresentationResult == VK_ERROR_OUT_OF_DATE_KHR ||
        framebufferResized) {
      framebufferResized = false;
      recreateSwapChain();
    } else if (queuePresentationResult != VK_SUCCESS) {
      throw new std::runtime_error("failed to present queue!");
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  void cleanupSwapChain() {
    for (auto framebuffer : swapChainFramebuffers) {
      vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    vkFreeCommandBuffers(device, commandPool,
                         static_cast<uint32_t>(commandBuffers.size()),
                         commandBuffers.data());
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);
    for (auto imageView : swapChainImageViews) {
      vkDestroyImageView(device, imageView, nullptr);
    }
    vkDestroySwapchainKHR(device, swapChain, nullptr);
  }

  void cleanup() {
    cleanupSwapChain();
    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexBufferMemory, nullptr);
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
      vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
      vkDestroyFence(device, inFlightFences[i], nullptr);
    }
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);

    if (enableValidationLayers) {
      DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }

    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
  }

  // member vars
  GLFWwindow *window = nullptr;

  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE; // (logical device)

  VkQueue graphicsQueue = VK_NULL_HANDLE;
  VkQueue presentQueue = VK_NULL_HANDLE;

  VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;

  VkSurfaceKHR surface = VK_NULL_HANDLE;

  VkSwapchainKHR swapChain = VK_NULL_HANDLE;
  VkFormat swapChainImageFormat = VK_FORMAT_UNDEFINED;
  VkExtent2D swapChainExtent = {0, 0};
  std::vector<VkImage> swapChainImages;
  std::vector<VkImageView> swapChainImageViews;
  std::vector<VkFramebuffer> swapChainFramebuffers;

  VkBuffer vertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;

  VkCommandPool commandPool = VK_NULL_HANDLE;
  std::vector<VkCommandBuffer> commandBuffers;

  VkRenderPass renderPass = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline graphicsPipeline = VK_NULL_HANDLE;

  std::vector<VkSemaphore> imageAvailableSemaphores;
  std::vector<VkSemaphore> renderFinishedSemaphores;
  std::vector<VkFence> inFlightFences;
  std::vector<VkFence> inFlightImages;
  size_t currentFrame = 0;

  bool framebufferResized = false;
};

int main() {
  HelloTriangleApplication app;

  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

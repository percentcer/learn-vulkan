if not exist ".\shaders_out" mkdir .\shaders_out
C:\VulkanSDK\1.2.131.2\Bin\glslc.exe .\shaders\shader.vert -o .\shaders_out\vert.spv
C:\VulkanSDK\1.2.131.2\Bin\glslc.exe .\shaders\shader.frag -o .\shaders_out\frag.spv
pause

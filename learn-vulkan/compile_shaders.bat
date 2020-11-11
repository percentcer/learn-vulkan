if not exist ".\shaders_out" mkdir .\shaders_out
%VULKAN_SDK%\Bin\glslc.exe .\shaders\shader.vert -o .\shaders_out\vert.spv
%VULKAN_SDK%\Bin\glslc.exe .\shaders\shader.frag -o .\shaders_out\frag.spv
pause

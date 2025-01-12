#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iterator>
#include <vector>
#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <jpeglib.h>
#include <jerror.h>
#include <chrono>
#include <cmath>
#include <numbers>

static const char vertexShaderSource[] = {
    #embed "shader/shader.vert"
};
static const char fragmentShaderSource[] = {
    #embed "shader/shader.frag"
};
static const char computeShaderSource[] = {
    #embed "shader/shader.comp"
};

static const char panoramaData[] = {
    #embed "../assets/milky_way_panorama.jpg"
};

int loadShaderSource(std::string& shaderSource, const char* shaderFile){
    std::ifstream shaderStream(shaderFile, std::ios::in);
    if(shaderStream.is_open()){
        std::stringstream stringStream;
        stringStream << shaderStream.rdbuf();
        shaderSource = stringStream.str();
        shaderStream.close();
    }else{
        std::cerr << "Failed to open shader source file '" << shaderFile << "'" << std::endl;
        return 1;
    }
    
    return 0;
}

GLuint loadShader(const char* shaderSource, GLuint shaderType){
    GLuint shaderId = glCreateShader(shaderType);
    if(!shaderId){
        std::cerr << "Failed to create shader" << std::endl;
        return 0;
    }
    
    glShaderSource(shaderId, 1, &shaderSource, NULL);
    glCompileShader(shaderId);
    
    GLint compileStatus = GL_FALSE;
    glGetShaderiv(shaderId, GL_COMPILE_STATUS, &compileStatus);
    if(!compileStatus){
        int infoLogLength;
        glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &infoLogLength);
        std::vector<char> errorMessage(infoLogLength+1);
        glGetShaderInfoLog(shaderId, infoLogLength, NULL, errorMessage.data());
        std::cerr << "Failed to compile shader: " << errorMessage.data() << std::endl;
        return 0;
    }
    
    return shaderId;
}

GLuint loadProgram(size_t count, const char** shaderSources, const GLuint* shaderTypes){
    std::vector<GLuint> shaderIds(count);
    
    for(size_t i = 0;i < count;++i){
        shaderIds[i] = loadShader(shaderSources[i], shaderTypes[i]);
        if(!shaderIds[i]){
            std::cerr << "Failed to load shader of type " << shaderTypes[i] << std::endl;
            return 0;
        }
    }
    
    GLuint programId = glCreateProgram();
    if(!programId){
        std::cerr << "Failed to create program" << std::endl;
        return 0;
    }
    for(size_t i = 0;i < count;++i){
        glAttachShader(programId, shaderIds[i]);
    }
    glLinkProgram(programId);
    GLint linkStatus = GL_FALSE;
    glGetProgramiv(programId, GL_LINK_STATUS, &linkStatus);
    if(!linkStatus){
        int infoLogLength;
        glGetProgramiv(programId, GL_INFO_LOG_LENGTH, &infoLogLength);
        std::vector<char> errorMessage(infoLogLength+1);
        glGetProgramInfoLog(programId, infoLogLength, NULL, errorMessage.data());
        std::cerr << "Failed to link program: " << errorMessage.data() << std::endl;
        return 0;
    }
    
    for(size_t i = 0;i < count;++i){
        glDetachShader(programId, shaderIds[i]);
        glDeleteShader(shaderIds[i]);
    }
    
    return programId;
}

int loadJpegData(std::vector<unsigned char>& data, const char* jpegFile){
    // https://stackoverflow.com/a/21802936/3874664
    std::ifstream jpegStream(jpegFile, std::ios::binary);
    if(jpegStream.is_open()){
        jpegStream.unsetf(std::ios::skipws);
        jpegStream.seekg(0, std::ios::end);
        std::streampos size = jpegStream.tellg();
        jpegStream.seekg(0, std::ios::beg);
        data.reserve(size);
        data.insert(data.begin(), std::istream_iterator<unsigned char>(jpegStream), std::istream_iterator<unsigned char>());
        jpegStream.close();
    }else{
        std::cerr << "Failed to open JPEG file '" << jpegFile << "'" << std::endl;
        return 1;
    }
    
    return 0;
}

// https://gist.github.com/PhirePhly/3080633
int loadJpeg(const unsigned char* data, const size_t length, std::vector<unsigned char>& dest, int& width, int& height, bool& hasAlpha){
    struct jpeg_decompress_struct info;
    struct jpeg_error_mgr error;
    info.err = jpeg_std_error(&error);
    jpeg_create_decompress(&info);
    jpeg_mem_src(&info, data, length);
    if(jpeg_read_header(&info, true) != 1){
        std::cerr << "JPEG data does not appear to be a JPEG" << std::endl;
        return 1;
    }
    jpeg_start_decompress(&info);
    width = info.output_width;
    height = info.output_height;
    int channels = info.output_components;
    if(channels == 4){
        hasAlpha = true;
    }else if(channels == 3){
        hasAlpha = false;
    }else{
        std::cerr << "Unsupported number of channels: " << channels << std::endl;
    }
    dest.resize(width * height * channels);
    int rowSize = width * channels;
    while(info.output_scanline < info.output_height){
        unsigned char* buffer[1];
        buffer[0] = dest.data() + info.output_scanline * rowSize;
        jpeg_read_scanlines(&info, buffer, 1);
    }
    jpeg_finish_decompress(&info);
    jpeg_destroy_decompress(&info);
    
    return 0;
}

int main(int argc, char** argv){
    int width = 1200;
    int height = 800;
    bool borderless = false;
    for(int i = 0;i < argc;++i){
        if(!strcmp(argv[i], "--borderless")){
            borderless = true;
        }else if(!strcmp(argv[i], "--width") && i + 1 < argc){
            int cliWidth = std::atoi(argv[i+1]);
            if(cliWidth > 0){
                width = cliWidth;
            }
        }else if(!strcmp(argv[i], "--height") && i + 1 < argc){
            int cliHeight = std::atoi(argv[i+1]);
            if(cliHeight > 0){
                height = cliHeight;
            }
        }else if(!strcmp(argv[i], "--x11")){
            if(glfwPlatformSupported(GLFW_PLATFORM_X11)){
                glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11); // GLFW Wayland has issues with monitor scaling
            }else{
                std::cerr << "X11 not available" << std::endl;
            }
        }
    }
    
    if(!glfwInit()){
        std::cerr << "GLFW initialisation failed" << std::endl;
        return 1;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* vidmode = glfwGetVideoMode(monitor);
    
    if(borderless){
        width = vidmode->width;
        height = vidmode->height;
        glfwWindowHint(GLFW_RED_BITS, vidmode->redBits);
        glfwWindowHint(GLFW_GREEN_BITS, vidmode->greenBits);
        glfwWindowHint(GLFW_BLUE_BITS, vidmode->blueBits);
    }else{
        width = width > vidmode->width ? vidmode->width : width;
        height = height > vidmode->height ? vidmode->height : height;
        monitor = NULL;
    }
    GLFWwindow* window = glfwCreateWindow(width, height, "GR Renderer", monitor, NULL);
    if(window == NULL){
        std::cerr << "Window creation failed" << std::endl;
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);
    
    gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);
    
    const char* computeShaderSources[1] = {computeShaderSource};
    const GLuint computeShaderTypes[1] = {GL_COMPUTE_SHADER};
    GLuint computeProgramId = loadProgram(1, computeShaderSources, computeShaderTypes);
    if(!computeProgramId){
        std::cout << "Compute shader program creation failed" << std::endl;
        glfwTerminate();
        return 1;
    }
    glUseProgram(computeProgramId);
    
    std::vector<unsigned char> panorama;
    int widthPanorama, heightPanorama;
    bool hasAlphaPanorama;
    if(loadJpeg(reinterpret_cast<const unsigned char*>(panoramaData), sizeof(panoramaData), panorama, widthPanorama, heightPanorama, hasAlphaPanorama)){
        glfwTerminate();
        return 1;
    }
    
    glUniform2f(glGetUniformLocation(computeProgramId, "destSize"), width, height);
    glUniform2f(glGetUniformLocation(computeProgramId, "backgroundSize"), widthPanorama, heightPanorama);
    
    const int workgroupSize = 8;
    int workgroupWidth = std::ceil((float) width / workgroupSize);
    int workgroupHeight = std::ceil((float) height / workgroupSize);
    
    const char* drawShaderSources[2] = {vertexShaderSource, fragmentShaderSource};
    const GLuint drawShaderTypes[2] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
    GLuint drawProgramId = loadProgram(2, drawShaderSources, drawShaderTypes);
    if(!drawProgramId){
        std::cerr << "Draw shader program creation failed" << std::endl;
        glfwTerminate();
        return 1;
    }
    glUseProgram(drawProgramId);
    
    GLuint vertexArray;
    glGenVertexArrays(1, &vertexArray);
    glBindVertexArray(vertexArray);
    
    float vertex[] = {-1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1};
    float texCoord[] = {0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0};
    
    GLuint vertexBuffer;
    glGenBuffers(1, &vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, 24 * sizeof(float), vertex, GL_STATIC_DRAW);
    
    GLuint texCoordBuffer;
    glGenBuffers(1, &texCoordBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, texCoordBuffer);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), texCoord, GL_STATIC_DRAW);
    
    glActiveTexture(GL_TEXTURE0);
    GLuint textureTarget;
    glGenTextures(1, &textureTarget);
    glBindTexture(GL_TEXTURE_2D, textureTarget);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
    glBindImageTexture(0, textureTarget, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
    
    glActiveTexture(GL_TEXTURE1);
    GLuint texturePanorama;
    glGenTextures(1, &texturePanorama);
    glBindTexture(GL_TEXTURE_2D, texturePanorama);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexImage2D(GL_TEXTURE_2D, 0, hasAlphaPanorama ? GL_RGBA32F : GL_RGB32F, widthPanorama, heightPanorama, 0, hasAlphaPanorama ? GL_RGBA : GL_RGB, GL_UNSIGNED_BYTE, panorama.data());
    glBindImageTexture(1, texturePanorama, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, texCoordBuffer);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    glBindTexture(GL_TEXTURE_2D, texturePanorama);
    
    const float startR = 20.0f;
    const float startTheta = std::numbers::pi_v<float> / 2 - std::numbers::pi_v<float> / 12;
    const float startPhi = 0.0f;
    float cameraPosition[] = {startR * std::sin(startTheta) * std::cos(startPhi), startR * std::sin(startTheta) * std::sin(startPhi), startR * std::cos(startTheta)};
    float cameraRotation[] = {startTheta, startPhi};
    const float cameraFov = 70.0f * std::numbers::pi_v<float> / 180.0f;
    bool stepsLock = false;
    
    const float mass = 1.0f;
    const float disk_min = 8.0f;
    const float disk_max = 16.0f;
    int steps = 500;
    const float dAffineFactor = 0.05;
    
    const size_t dtStride = 200;
    size_t dtStep = 0;
    auto time = std::chrono::high_resolution_clock::now();
    auto fpsTime = time;
    
    while(!glfwWindowShouldClose(window)){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        auto now = std::chrono::high_resolution_clock::now();
        float dt = (now - time).count() / 1e9;
        time = now;
        if(dtStep >= dtStride){
            float dtFps = (now - fpsTime).count() / 1e9 / dtStep;
            fpsTime = now;
            dtStep = 0;
            std::cout << "FPS: " << (1 / dtFps) << std::endl;
        }else{
            ++dtStep;
        }
        
        if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS){
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        
        float speed = 2.0f;
        if(glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) speed *= 5;
        if(glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS) speed /= 5;
        if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS){
            cameraPosition[0] -= speed * std::cos(cameraRotation[1]) * dt;
            cameraPosition[1] += speed * std::sin(cameraRotation[1]) * dt;
        }
        if(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS){
            cameraPosition[0] += speed * std::cos(cameraRotation[1]) * dt;
            cameraPosition[1] -= speed * std::sin(cameraRotation[1]) * dt;
        }
        if(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS){
            cameraPosition[0] += speed * std::sin(cameraRotation[1]) * dt;
            cameraPosition[1] += speed * std::cos(cameraRotation[1]) * dt;
        }
        if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS){
            cameraPosition[0] -= speed * std::sin(cameraRotation[1]) * dt;
            cameraPosition[1] -= speed * std::cos(cameraRotation[1]) * dt;
        }
        if(glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS){
            cameraPosition[2] += speed * dt;
        }
        if(glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS){
            cameraPosition[2] -= speed * dt;
        }
        if(glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS){
            cameraRotation[0] += speed * dt / 5;
            if(cameraRotation[0] > std::numbers::pi_v<float>) cameraRotation[0] = std::numbers::pi_v<float>;
        }
        if(glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS){
            cameraRotation[0] -= speed * dt / 5;
            if(cameraRotation[0] < 0) cameraRotation[0] = 0;
        }
        if(glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS){
            cameraRotation[1] += speed * dt / 5;
        }
        if(glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS){
            cameraRotation[1] -= speed * dt / 5;
        }
        bool decreaseSteps = glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS;
        bool increaseSteps = glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_KP_ADD) == GLFW_PRESS;
        if(!stepsLock && steps > (int) (50 * speed) && decreaseSteps){
            steps -= (int) (50 * speed);
            stepsLock = true;
            std::cout << "Steps: " << steps << std::endl;
        }else if(!stepsLock && increaseSteps){
            steps += (int) (50 * speed);
            stepsLock = true;
            std::cout << "Steps: " << steps << std::endl;
        }else if(!(decreaseSteps || increaseSteps)){
            stepsLock = false;
        }
        
        glUseProgram(computeProgramId);
        glUniform3fv(glGetUniformLocation(computeProgramId, "cameraPosition"), 1, cameraPosition);
        glUniform3fv(glGetUniformLocation(computeProgramId, "cameraRotation"), 1, cameraRotation);
        glUniform1f(glGetUniformLocation(computeProgramId, "cameraHFov"), cameraFov);
        glUniform1f(glGetUniformLocation(computeProgramId, "mass"), mass);
        glUniform1f(glGetUniformLocation(computeProgramId, "disk_min"), disk_min);
        glUniform1f(glGetUniformLocation(computeProgramId, "disk_max"), disk_max);
        glUniform1i(glGetUniformLocation(computeProgramId, "steps"), steps);
        glUniform1f(glGetUniformLocation(computeProgramId, "dAffineFactor"), dAffineFactor);
        glDispatchCompute(workgroupWidth, workgroupHeight, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        
        glUseProgram(drawProgramId);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    glDeleteTextures(1, &textureTarget);
    glDeleteTextures(1, &texturePanorama);
    glDeleteBuffers(1, &vertexBuffer);
    glDeleteBuffers(1, &texCoordBuffer);
    glDeleteProgram(drawProgramId);
    glfwTerminate();
    
    return 0;
}

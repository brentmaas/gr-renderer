#version 460 core

layout(location = 0) in vec4 vertexPosition;
layout(location = 1) in vec2 vertexTexCoord;

out vec2 texCoord;

void main(){
    gl_Position = vertexPosition;
    texCoord = vertexTexCoord;
}
#version 460 core

in vec2 texCoord;

uniform sampler2D texSampler;

void main(){
    gl_FragColor = texture(texSampler, texCoord);
}
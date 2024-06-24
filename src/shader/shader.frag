#version 460 core

in vec2 texCoord;

out vec4 fragColour;

uniform sampler2D texSampler;

void main(){
    fragColour = texture(texSampler, texCoord);
}

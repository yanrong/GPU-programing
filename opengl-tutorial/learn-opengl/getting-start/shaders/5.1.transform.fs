#version 330 core
out vec4 fragColor;

in vec2 texCoord;

//texture samplers
uniform sampler2D texture1;
uniform sampler2D texture2;

void main() {
    //linearly interpolate betweent both texture
    fragColor = mix(texture(texture1, texCoord), texture(texture2, texCoord), 0.2);
}
#version 330 core

out vec4 fColor;
in vec2 fTexCoord;

uniform sampler2D screenTexture;

void main(){
    vec3 color = texture(screenTexture, fTexCoord).rgb;
    float grayScale = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
    fColor = vec4(vec3(grayScale), 1.0);
}
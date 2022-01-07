#version 330 core

out vec4 fColor;
in vec2 fTexCoord;

uniform sampler2D screenTexture;

void main(){
    fColor = texture(screenTexture, fTexCoord);
    float average = (fColor.r + fColor.g + fColor.b) / 3.0;
    fColor = vec4(average, average, average, 1.0);
}
#version 330 core

out vec4 fColor;
in vec2 fTexCoord;

uniform sampler2D screenTexture;

void main(){
    fColor = texture(screenTexture, fTexCoord);
    float average =  0.2126 * fColor.r +  0.7152 * fColor.g +  0.0722 * fColor.b;
    fColor = vec4(average, average, average, 1.0);
}
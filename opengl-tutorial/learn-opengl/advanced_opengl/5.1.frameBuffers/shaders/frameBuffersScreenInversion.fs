#version 330 core

out vec4 fColor;
in vec2 fTexCoord;

uniform sampler2D screenTexture;

void main(){
    fColor = vec4(vec3(1.0 - texture(screenTexture, fTexCoord)), 1.0);
/*
    vec3 color = texture(screenTexture, fTexCoord).rgb;
    fColor = vec4(color, 1.0);
*/
}
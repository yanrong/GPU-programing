#version 330 core

layout (location = 0) in vec2 vPosition;
layout (location = 1) in vec3 vColor;
layout (location = 2) in vec2 vOffset;

out vec3 fColor;

void main(){
    fColor = vColor;
    gl_Position = vec4(vPosition + vOffset, 0.0, 1.0);
}
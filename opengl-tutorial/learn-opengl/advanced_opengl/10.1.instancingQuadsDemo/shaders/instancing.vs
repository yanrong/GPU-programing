#version 330 core

layout (location = 0) in vec2 vPosition;
layout (location = 1) in vec3 vColor;

out vec3 fColor;
uniform vec2 offsets[100];

void main(){
    fColor = vColor;
    vec2 offset = offsets[gl_InstanceID];
    gl_Position = vec4(vPosition + offset, 0.0, 1.0);
}
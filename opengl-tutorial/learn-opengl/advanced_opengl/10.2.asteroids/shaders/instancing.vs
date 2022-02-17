#version 330 core

layout (location = 0) in vec3 vPosition;
layout (location = 2) in vec2 vTexCoord;

out vec2 fTexCoord;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main(){
    fTexCoord = vTexCoord;
    gl_Position = projection * view * model * vec4(vPosition, 1.0f);
}
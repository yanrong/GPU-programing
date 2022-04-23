#version 430 core

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec2 vTexCoord;

uniform mat4 projection;
uniform mat4 model;

out vec2 texCoord;

void main(){
    gl_Position = projection * model * vec4(vPosition, 1.0f);
    texCoord = vTexCoord;
}
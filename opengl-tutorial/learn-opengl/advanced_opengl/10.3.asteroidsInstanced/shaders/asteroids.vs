#version 330 core

layout (location = 0) in vec3 vPosition;
layout (location = 2) in vec2 vTexCoord;
layout (location = 3) in mat4 vInstanceMatrix;

out vec2 fTexCoord;

uniform mat4 projection;
uniform mat4 view;

void main(){
    fTexCoord = vTexCoord;
    gl_Position = projection * view * vInstanceMatrix * vec4(vPosition, 1.0f);
}
#version 330 core

layout (location = 0) in vec3 position;

uniform mat4 gScaling;

void main(){
    gl_Position = gScaling * vec4(position, 1.0);
}
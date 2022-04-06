#version 330 core

layout (location = 0) in vec3 position;

uniform mat4 gRotation;

void main(){
    gl_Position = gRotation * vec4(position, 1.0);
}
#version 330 core

layout (location = 0) in vec3 vPosition;
layout (location = 2) in vec2 vTexCoord;

out VS_OUT{
    vec2 texCoord;
} vs_out;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main(){
    vs_out.texCoord = vTexCoord;
    gl_Position = projection * view *model * vec4(vPosition, 1.0);
}
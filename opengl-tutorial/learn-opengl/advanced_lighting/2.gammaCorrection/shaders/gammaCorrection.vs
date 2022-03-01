#version 330 core

layout (location = 0) in vec3 vPositon;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vTexCoord;

out VS_OUT {
    vec3 fragPosition;
    vec3 fNormal;
    vec2 fTexCoord;
} vsOut;

uniform mat4 projection;
uniform mat4 view;

void main(){
    vsOut.fragPosition = vPositon;
    vsOut.fNormal = vNormal;
    vsOut.fTexCoord = vTexCoord;

    gl_Position = projection * view * vec4(vPositon, 1.0);
}
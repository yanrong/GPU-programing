#version 330 core

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vTexCoord;

//declare an interface block
out VS_OUT {
    vec3 fragPosition;
    vec3 Normal;
    vec2 texCoord;
} vsOut;

uniform mat4 projection;
uniform mat4 view;

void main(){
    vsOut.fragPosition = vPosition;
    vsOut.Normal = vNormal;
    vsOut.texCoord = vTexCoord;

    gl_Position = projection * view * vec4(vPosition, 1.0);
}
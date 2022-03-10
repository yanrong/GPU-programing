#version 330 core

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vTexCoord;

//out vec2 fTexCoord;

out VS_OUT {
    vec3 fragPos;
    vec3 normal;
    vec2 texCoord;
    vec4 fragPosLightSpace;
} vsOut;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform mat4 lightSpaceMatrix;

void main(){
    vsOut.fragPos = vec3(model * vec4(vPosition, 1.0));
    vsOut.normal = transpose(inverse(mat3(model))) * vNormal;
    vsOut.texCoord = vTexCoord;
    vsOut.fragPosLightSpace = lightSpaceMatrix * vec4(vsOut.fragPos, 1.0);
    gl_Position = projection * view * model * vec4(vPosition, 1.0);
}
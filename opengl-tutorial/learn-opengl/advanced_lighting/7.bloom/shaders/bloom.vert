#version 330 core

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vTexCoord;

out VS_OUT {
    vec3 fragPosition;
    vec3 normal;
    vec2 texCoord;
} vsOut;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main(){
    vsOut.fragPosition = vec3(model * vec4(vPosition, 1.0));
    vsOut.texCoord = vTexCoord;

    mat3 normalMatrix = transpose(inverse(mat3(model)));
    vsOut.normal = normalize(normalMatrix * vNormal);

    gl_Position = projection * view * model * vec4(vPosition, 1.0);
}
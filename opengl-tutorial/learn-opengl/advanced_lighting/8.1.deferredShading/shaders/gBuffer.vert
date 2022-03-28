#version 330 core

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vTexCoord;

out vec3 fragPosition;
out vec2 texCoord;
out vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main(){
    vec4 worldPosition = model * vec4(vPosition, 1.0);
    fragPosition = worldPosition.xyz;
    texCoord = vTexCoord;

    mat3 normalMatrix = transpose(inverse(mat3(model)));
    normal = normalMatrix * vNormal;

    gl_Position = projection * view * worldPosition;
}
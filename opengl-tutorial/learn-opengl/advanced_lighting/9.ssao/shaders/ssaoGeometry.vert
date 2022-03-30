#version 330 core

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vTexCoord;

out vec3 fragPosition;
out vec2 texCoord;
out vec3 normal;

uniform bool invertedNormals;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main(){
    vec4 viewPosition = view * model * vec4(vPosition, 1.0);
    fragPosition = viewPosition.xyz;
    texCoord = vTexCoord;

    mat3 normalMatrix = transpose(inverse(mat3(view * model)));
    normal = normalMatrix * (invertedNormals ? -vNormal : vNormal);

    gl_Position = projection * viewPosition;
}
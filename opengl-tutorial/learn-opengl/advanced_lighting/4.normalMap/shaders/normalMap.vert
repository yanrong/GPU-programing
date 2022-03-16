#version 330 core

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vTexCoord;
layout (location = 3) in vec3 vTangent;
layout (location = 4) in vec3 vBitangent;

out VS_OUT{
    vec3 fragPosition;
    vec2 texCoord;
    vec3 tangentLightPosition;
    vec3 tangentViewPosition;
    vec3 tangentFragPosition;
} vsOut;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

uniform vec3 lightPosition;
uniform vec3 viewPosition;

void main(){
    vsOut.fragPosition = vec3(model * vec4(vPosition, 1.0));
    vsOut.texCoord = vTexCoord;

    mat3 normalMatrix = transpose(inverse(mat3(model)));
    vec3 T = normalize(normalMatrix * vTangent);
    vec3 N = normalize(normalMatrix * vNormal);
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);

    mat3 TBN = transpose(mat3(T, B, N));
    vsOut.tangentLightPosition = TBN * lightPosition;
    vsOut.tangentViewPosition = TBN * viewPosition;
    vsOut.tangentFragPosition = TBN * vsOut.fragPosition;

    gl_Position = projection * view * model * vec4(vPosition, 1.0);
}
#version 330 core

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vTexCoord;

out vec2 fTexCoord;

out VS_OUT {
    vec3 fragPosition;
    vec3 normal;
    vec2 texCoord;
} vsOut;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

uniform bool reverseNormal;

void main(){
    vsOut.fragPosition = vec3(model * vec4(vPosition, 1.0));
    if (reverseNormal) {
        // a slight hack to make sure the outer large cube displays lighting
        // from the 'inside' instead of the default 'outside'.
        vsOut.normal = transpose(inverse(mat3(model))) * (-1.0 * vNormal);
    } else {
        vsOut.normal = transpose(inverse(mat3(model))) * vNormal;
    }
    vsOut.texCoord = vTexCoord;
    gl_Position = projection * view * model * vec4(vPosition, 1.0);
}
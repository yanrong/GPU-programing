#version 330 core

out vec4 fColor;
in vec2 fTexCoord;

uniform sampler2D depthMap;
uniform float nearPlane;
uniform float farPlane;

//required when using a perspective projection matrix
float linearizeDepth(float depth)
{
    float z = depth * 2.0 - 1.0; //back to NDC
    return (2.0 * nearPlane * farPlane) / (farPlane + nearPlane - z * (farPlane - nearPlane));
}

void main(){
    float depthValue = texture(depthMap, fTexCoord).r;
    //fColor = vec4(vec3(linearizeDepth(depthValue) / farPlane), 1.0); //perspective
    fColor = vec4(vec3(depthValue), 1.0); //orthographic
}
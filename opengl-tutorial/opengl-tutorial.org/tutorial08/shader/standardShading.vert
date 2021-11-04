#version 330 core

//Input vertex data, different for all executation of this shader
layout(location = 0) in vec3 vertexPositionModelSpace;
layout(location = 1) in vec2 vertexUV;
layout(location = 2) in vec3 vertexNormalModelSpace;

//Output data; will be interpolated for each fragment
out vec2 UV;
out vec3 positionWorldSpace;
out vec3 normalCameraSpace;
out vec3 eyeDirectionCameraSpace;
out vec3 lightDirectionCameraSpce;

//Values that stay constant for the whole mesh
uniform mat4 MVP;
uniform mat4 V;
uniform mat4 M;
uniform vec3 lightPositionWorldSpace;

void main() {
    //Output position of the vertex, in clip space: MVP * position
    gl_Position = MVP * vec4(vertexPositionModelSpace, 1);
    //Position of the vertex, in worldspace : M * position
    positionWorldSpace = (M * vec4(vertexPositionModelSpace, 1)).xyz;

    //Vector that goes from the vertex to the camera, in camera space
    //The camera is at the origin(0, 0, 0)
    vec3 vertexPositionCameraSpace = (V * M * vec4(vertexPositionModelSpace, 1)).xyz;
    eyeDirectionCameraSpace = vec3(0, 0, 0) - vertexPositionCameraSpace;

    //Vector that gose from the vertex to the light, in camera space. M is ommit
    //because it's identity.
    vec3 lightPositionCameraSpace = (V * vec4(lightPositionWorldSpace, 1)).xyz;
    lightDirectionCameraSpce = lightPositionCameraSpace + eyeDirectionCameraSpace;

    //Normal of the vertex, in camera space
    //Only correct it modelMatrix does not scale the model ! use its inverse transpose if not
    normalCameraSpace = (V * M * vec4(vertexNormalModelSpace, 0)).xyz;

    UV = vertexUV;
}

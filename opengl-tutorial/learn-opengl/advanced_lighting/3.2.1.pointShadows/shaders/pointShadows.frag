#version 330 core

out vec4 fragColor;
in VS_OUT {
    vec3 fragPosition;
    vec3 normal;
    vec2 texCoord;
} fsIn;

uniform sampler2D diffuseTexture;
uniform samplerCube depthMap;

uniform vec3 lightPosition;
uniform vec3 viewPosition;
uniform float farPlane;
uniform bool shadows;


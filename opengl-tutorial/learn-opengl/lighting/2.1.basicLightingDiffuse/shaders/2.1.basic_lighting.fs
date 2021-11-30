#version 330 core

out vec4 fColor;

in vec3 fNormal;
in vec3 fPosition;

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

void main() {
    //ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    //diffuse
    vec3 normal = normalize(fNormal);
    vec3 lightDir = normalize(lightPos - fPosition);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    vec3 result = (ambient + diffuse) * objectColor;
    fColor = vec4(result, 1.0);
}
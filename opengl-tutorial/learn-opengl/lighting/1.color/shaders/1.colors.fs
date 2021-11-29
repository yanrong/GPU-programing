#version 330 core

out vec4 fColor;

uniform vec3 objectColor;
uniform vec3 lightColor;

void main() {
    fColor = vec4(lightColor * objectColor, 1.0);
}
#version 330 core
out vec4 fColor;

in vec3 lightingColor;
uniform vec3 objectColor;

void main() {

    fColor = vec4(lightingColor * objectColor, 1.0);
}
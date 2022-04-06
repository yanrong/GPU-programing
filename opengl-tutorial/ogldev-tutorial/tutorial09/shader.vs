#version 330 core

layout (location = 0) in vec3 vPosition;

uniform mat4 gWorld;

out vec4 fColor;

void main() {
    gl_Position = gWorld * vec4(vPosition, 1.0);
    fColor = vec4(clamp(vPosition, 0.0, 1.0), 1.0);
}
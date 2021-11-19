#version 330 core
// finally output color
out vec4 fragColor;
// input ourColor is defined in vertex shader and pass to fragColor
in vec3 ourColor;

void main() {
    fragColor = vec4(ourColor, 1.0f);
}
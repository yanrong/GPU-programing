#version 330 core
//location qualify for setup vectexs in CPU side
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
//this color will pass to fragment shader, pass through shader
out vec3 ourColor;

void main()
{
    gl_Position = vec4(aPos, 1.0);
    ourColor = aColor;
}
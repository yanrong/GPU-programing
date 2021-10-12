#version 330 core

//Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition
// Values that stay constant for the whole mesh
unifrom mat4 mvp

void main()
{
    //Output position of the vertext, in clip space: mvp * pistion
    gl_Position = mvp * vec4(vertexPosition, 1);
}
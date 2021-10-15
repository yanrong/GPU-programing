#version 330 core

//Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec3 vertexColor;

out vec3 fragmentColor;
// Values that stay constant for the whole mesh
uniform mat4 mvp;

void main()
{
    //Output position of the vertext, in clip space: mvp * pistion
    gl_Position = mvp * vec4(vertexPosition, 1);
    //The color of each vertex will be interrupted
    //to produce the color of the each fragment
    fragmentColor = vertexColor;
}
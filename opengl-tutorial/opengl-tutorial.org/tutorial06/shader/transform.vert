#version 330 core

// Input vertex data, different for all executions ot this shader.
layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec2 vertexUV;

// Output data, will be interpolated for each fragment.
out vec2 uv;
// Value that stay constant for the whole mesh
uniform mat4 mvp;

void main() {
    // Output postion of the vertex, in clip space: MVP * Position
    gl_Position = mvp * vec4(vertexPosition, 1);
    //UV of ther vertex, No special space for this one.
    uv = vertexUV;
}

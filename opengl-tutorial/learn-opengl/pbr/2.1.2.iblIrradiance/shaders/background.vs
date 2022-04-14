#version 330 core

layout (location = 0) in vec3 vPosition;

uniform mat4 projection;
uniform mat4 view;

out vec3 worldPosition;

void main()
{
    worldPosition = vPosition;
    mat4 rotView = mat4(mat3(view)); // remove translation from the view matrix
    vec4 clipPosition = projection * rotView * vec4(worldPosition, 1.0);

    gl_Position = clipPosition.xyww;
}
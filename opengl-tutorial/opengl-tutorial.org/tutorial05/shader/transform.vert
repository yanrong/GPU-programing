#version 330 core

//Input vertex data, different for all extension of this shader
layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec2 vertextUV;

//Output data; will be interpolate for each fragment
out vec2 uv;
//Values that stay constant for the whole mesh
uniform mat4 mvp;

void main(){
    // Ouput position of the vertex, inclip space: mvp * position
    gl_Position = mvp * vec4(vertexPosition, 1);
    //UV of the vertex, No special space for this one.
    uv = vertextUV;
}
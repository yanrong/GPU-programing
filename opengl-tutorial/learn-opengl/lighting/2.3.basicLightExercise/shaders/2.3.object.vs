#version 330 core
layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;

out vec3 fPosition;
out vec3 fNormal;
out vec3 fLightPos;

/*
    we now define the uniform in the vertex shader and pass the 'view space'
    lightPos to the fragment shader. lightPos is currently in world space.
*/
uniform vec3 lightPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(vPosition, 1.0);
    fPosition = vec3(view * model * vec4(vPosition, 1.0));
    fNormal = mat3(transpose(inverse(view * model))) * vNormal;
    // Transform world-space light position to view-space light position
    fLightPos = vec3(view * vec4(lightPos, 1.0));
}

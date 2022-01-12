#version 330 core

out vec4 fColor;
in vec3 fTexCoord;

uniform samplerCube skybox;

void main(){
    fColor = texture(skybox, fTexCoord);
}
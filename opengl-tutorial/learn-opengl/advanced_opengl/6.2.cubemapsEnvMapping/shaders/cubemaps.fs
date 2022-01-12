#version 330 core
out vec4 fColor;

in vec3 fNormal;
in vec3 fPosition;

uniform vec3 cameraPosition;
uniform samplerCube skybox;

void main(){
    //light direction
    vec3 I = normalize(fPosition - cameraPosition);
    //mirror reflect
    vec3 R = reflect(I, normalize(fNormal));
    fColor = vec4(texture(skybox, R).rgb, 1.0);
}
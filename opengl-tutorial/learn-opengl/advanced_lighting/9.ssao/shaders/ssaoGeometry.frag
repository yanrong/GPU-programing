#version 330 core

layout (location = 0) out vec3 gPosition;
layout (location = 1) out vec3 gNormal;
layout (location = 2) out vec3 gAlbedo;

in vec3 fragPosition;
in vec3 normal;
in vec2 texCoord;

void main(){
    //store the fragment position vector in the first gbuffer texture
    gPosition = fragPosition;
    //store the fragment normal in gbuffer
    gNormal = normalize(normal);
    //also store the diffuse
    gAlbedo.rgb = vec3(0.95);
}
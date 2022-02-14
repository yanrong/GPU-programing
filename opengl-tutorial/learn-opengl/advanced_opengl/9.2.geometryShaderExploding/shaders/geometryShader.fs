#version 330 core

out vec4 fColor;
in vec2 fTexCoord;

uniform sampler2D textureDiffuse1;

void main(){
    fColor = texture(textureDiffuse1, fTexCoord);
}
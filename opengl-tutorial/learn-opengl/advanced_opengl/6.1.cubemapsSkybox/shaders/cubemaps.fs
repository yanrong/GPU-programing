#version 330 core
out vec4 fColor;

in vec2 fTexCoord;

uniform sampler2D texture1;

void main(){
    fColor = texture(texture1, fTexCoord);
}
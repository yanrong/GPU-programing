#version 330 core

out vec4 fColor;
in vec2 fTexCoord;

uniform sampler2D texture1;

void main(){
    vec4 texColor = texture(texture1, fTexCoord);
    if (texColor.a < 0.1) {
        discard;
    }
    fColor = texColor;
}
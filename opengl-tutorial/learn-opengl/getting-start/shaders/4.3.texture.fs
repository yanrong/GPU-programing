#version 330 core
out vec4 fragColor;

in vec3 ourColor;
in vec2 texCoord;

uniform sampler2D ourTexture1;
uniform sampler2D ourTexture2;

void main() {
    fragColor = mix(texture(ourTexture1, texCoord),
    texture(ourTexture2, vec2(1.0 - texCoord.x, texCoord.y)), 0.2) * vec4(ourColor, 1.0) ;
}
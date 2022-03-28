#version 330 core

layout (location = 0) out vec3 gPosition;
layout (location = 1) out vec3 gNormal;
layout (location = 2) out vec4 gAlbedoSpec;

in vec3 fragPosition;
in vec2 texCoord;
in vec3 normal;

uniform sampler2D textureDiffuse;
uniform sampler2D textureSpecular;

void main(){
    //store the fragment position vector in first gbuffer texture
    gPosition = fragPosition;
    //also store pre-fragment normal in the gbuffer
    gNormal = normalize(normal);
    //and diffuse per-fragment color
    gAlbedoSpec.rgb = texture(textureDiffuse, texCoord).rgb;
    gAlbedoSpec.a = texture(textureSpecular, texCoord).r;
}
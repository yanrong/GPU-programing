#version 330 core
out vec4 fragColor;
in vec2 texCoord;

uniform sampler2D scene;
uniform sampler2D bloomBlur;
uniform bool bloom;
uniform float exposure;

void main(){
    const float gamma = 2.2;
    vec3 hdrColor = texture(scene, texCoord).rgb;
    vec3 bloomColor = texture(bloomBlur, texCoord).rgb;

    if (bloom) {
        hdrColor += bloomColor; //additive blending
    }

    //tone map
    vec3 result = vec3(1.0) - exp(-hdrColor * exposure);
    //also gamma correct
    result = pow(result, vec3(1.0 / gamma));
    fragColor = vec4(result, 1.0);
}
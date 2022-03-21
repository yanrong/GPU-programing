#version 330 core

out vec4 fragColor;
in vec2 texCoord;

uniform sampler2D hdrBuffer;
uniform bool hdr;
uniform float exposure;

void main(){
    const float gamma = 2.2;

    vec3 hdrColor = texture(hdrBuffer, texCoord).rgb;
    if (hdr) {
        //reinhard
        //vec3 result = hdrColor / (hdrColor + vec3(1.0));
        //exposure
        vec3 result = vec3(1.0) - exp(-hdrColor * exposure);
        //also gamma coorect while we're at it
        result = pow(result, vec3(1.0 / gamma));
        fragColor = vec4(result, 1.0);
    } else {
        vec3 result = pow(hdrColor, vec3(1.0 / gamma));
        fragColor = vec4(result, 1.0);
    }
}
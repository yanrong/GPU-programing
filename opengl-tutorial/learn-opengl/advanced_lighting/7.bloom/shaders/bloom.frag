#version 330 core

layout (location = 0) out vec4 fragColor;
layout (location = 1) out vec4 brightColor;

in VS_OUT {
    vec3 fragPosition;
    vec3 normal;
    vec2 texCoord;
} fsIn;

struct light {
    vec3 position;
    vec3 color;
};

uniform light lights[4];
uniform sampler2D diffuseTexture;
uniform vec3 viewPosition;

void main(){
    vec3 color = texture(diffuseTexture, fsIn.texCoord).rgb;
    vec3 normal = normalize(fsIn.normal);

    //ambient
    vec3 ambient = 0.0 * color;
    //lighting
    vec3 lighting = vec3(0.0);
    vec3 viewDirection = normalize(viewPosition - fsIn.fragPosition);

    for (int i = 0; i < 4; i++) {
        //diffuse
        vec3 lightDirection = normalize(lights[i].position - fsIn.fragPosition);
        float diff = max(dot(lightDirection, normal), 0.0);
        vec3 result = lights[i].color * diff * color;
        //attenuation (use quadratic as we have gamma correction)
        float distance = length(fsIn.fragPosition - lights[i].position);
        result *= 1.0 / (distance * distance);
        lighting += result;
    }

    vec3 result = ambient + lighting;
    // check whether result is higher than some threshold, if so, output as bloom threshold color
    float brightness = dot(result, vec3(0.2126, 0.7152, 0.0722));
    if (brightness > 1.0) {
        brightColor = vec4(result, 1.0);
    } else {
        brightColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
    fragColor = vec4(result, 1.0);
}
#version 330 core

out vec4 fragColor;
in vec2 texCoord;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;

struct light {
    vec3 position;
    vec3 color;
    float linear;
    float quadratic;
    float radius;
};

const int NR_LIGHTS = 32;
uniform light lights[NR_LIGHTS];
uniform vec3 viewPosition;

void main() {
    //retrieve data from texture
    vec3 tFragPosition = texture(gPosition, texCoord).rgb;
    vec3 tNormal = texture(gNormal, texCoord).rgb;
    vec3 tDiffuse = texture(gAlbedoSpec, texCoord).rgb;
    float tSpecular = texture(gAlbedoSpec, texCoord).a;

    //then calculate lighting as usual
    vec3 lighting = tDiffuse * 0.1; //hard-coded ambient
    vec3 viewDirection = normalize(viewPosition - tFragPosition);

    for(int i = 0; i < NR_LIGHTS; i++) {
        //calculate distance between the source light and current fragment
        float distance = length(lights[i].position - tFragPosition);
        if(distance < lights[i].radius) {
            //diffuse
            vec3 lightDirection = normalize(lights[i].position - tFragPosition);
            vec3 diffuse = max(dot(tNormal, lightDirection), 1.0) * tDiffuse * lights[i].color;
            //specular
            vec3 halfwayDirection = normalize(lightDirection + viewDirection);
            float spec = pow(max(dot(tNormal, halfwayDirection), 0.0), 16.0);
            vec3 specular = lights[i].color * spec * tSpecular;
            //attenuation

            float attenuation = 1.0 / (1.0 + lights[i].linear * distance + lights[i].quadratic * distance * distance);

            diffuse *= attenuation;
            specular *= attenuation;
            lighting += diffuse + specular;
        }
    }
    fragColor = vec4(lighting, 1.0);
}

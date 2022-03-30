#version 330 core

out vec4 fragColor;
in vec2 texCoord;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedo;
uniform sampler2D ssao;

struct myLight {
    vec3 position;
    vec3 color;
    float linear;
    float quadratic;
};

uniform myLight light;

void main(){
    //retrieve data from gbuffer
    vec3 tFragPosition = texture(gPosition, texCoord).rgb;
    vec3 tNormal = texture(gNormal, texCoord).rgb;
    vec3 tDiffuse = texture(gAlbedo, texCoord).rgb;
    float ambientOcclusion = texture(ssao, texCoord).r;

    //then calculate lighting as usual
    vec3 ambient = vec3(0.3 * tDiffuse * ambientOcclusion);
    vec3 lighting = ambient;
    vec3 viewDirection = normalize(-tFragPosition); // in view space, view position at (0, 0, 0)
    //diffuse
    vec3 lightDirection = normalize(light.position - tFragPosition);
    vec3 diffuse = max(dot(tNormal, lightDirection), 0.0) * tDiffuse * light.color;
    //specular
    vec3 halfwayDirection = normalize(lightDirection + viewDirection);
    float spec = pow(max(dot(tNormal, halfwayDirection), 0.0), 8.0);
    vec3 specular = light.color * spec;

    //attenuation
    float distance = length(light.position - tFragPosition);
    float attenuation = 1.0 / (1.0 + light.linear * distance + light.quadratic * (distance * distance));

    diffuse *= attenuation;
    specular *= attenuation;
    lighting += diffuse + specular;

    fragColor = vec4(lighting, 1.0);
}
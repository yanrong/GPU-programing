#version 330 core
out vec4 fColor;

struct Material {
    sampler2D diffTexture;
    sampler2D specTexture;
    float shininess;
};

struct Light {
    vec3 position;
    vec3 direction;
    float cutOff;
    float outerCutOff;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    float constant;
    float linear;
    float quadratic;
};

in vec3 fPosition;
in vec3 fNormal;
in vec2 fTexCoords;

uniform vec3 viewPosition;
uniform Material material;
uniform Light light;

void main() {
    //ambient
    vec3 ambient = light.ambient * texture(material.diffTexture, fTexCoords).rgb;

    //diffuse
    vec3 normal = normalize(fNormal);
    vec3 lightDir = normalize(light.position - fPosition);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = light.diffuse * diff * texture(material.diffTexture, fTexCoords).rgb;

    //specular
    vec3 viewDir = normalize(viewPosition - fPosition);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    vec3 specular = light.specular * spec * texture(material.specTexture, fTexCoords).rgb;

    //spotlight (soft edges)
    float theta = dot(lightDir, normalize(-light.direction));
    float epsilon = (light.cutOff - light.outerCutOff);
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
    diffuse *= intensity;
    specular *= intensity;

    //attenuation
    float distance = length(light.position - fPosition);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;

    vec3 result = ambient + diffuse + specular;
    fColor = vec4(result, 1.0);
}
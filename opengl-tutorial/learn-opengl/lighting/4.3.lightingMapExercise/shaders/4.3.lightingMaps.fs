#version 330 core
out vec4 fColor;

struct Material {
    sampler2D diffTexture;
    sampler2D specTexture;
    sampler2D emissionTexture;
    float shininess;
};

struct Light {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
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

    //emission
    vec3 emission = texture(material.emissionTexture, fTexCoords).rgb;

    vec3 result = ambient + diffuse + specular + emission;
    fColor = vec4(result, 1.0);
}
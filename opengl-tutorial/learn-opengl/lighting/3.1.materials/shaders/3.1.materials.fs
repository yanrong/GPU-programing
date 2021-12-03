#version 330 core
out vec4 fColor;

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
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

uniform vec3 viewPosition;
uniform Material material;
uniform Light light;

void main() {
    //ambient
    vec3 ambient = light.ambient * material.ambient;
    //diffuse
    vec3 normal = normalize(fNormal);
    vec3 lightDir = normalize(light.position - fPosition);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = light.diffuse * (diff * material.diffuse);
    //specular
    vec3 viewDir = normalize(viewPosition - fPosition);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    vec3 specular = light.specular * (spec * material.specular);

    vec3 result = ambient + diffuse + specular;
    fColor = vec4(result, 1.0);
}
#version 330 core
out vec4 fColor;

struct Material {
    sampler2D diffTexture;
    sampler2D specTexture;
    float shininess;
};

struct directionLight {
    vec3 direction;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct pointLight {
    vec3 position;

    float constant;
    float linear;
    float quadratic;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct spotLight {
    vec3 position;
    vec3 direction;

    float cutOff;
    float outerCutOff;

    float constant;
    float linear;
    float quadratic;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

#define NR_POINT_LIGHTS 4

in vec3 fPosition;
in vec3 fNormal;
in vec2 fTexCoords;

uniform vec3 viewPosition;
uniform directionLight myDirLight;
uniform pointLight myPointLights[NR_POINT_LIGHTS];
uniform spotLight mySpotLight;
uniform Material material;

//function prototypes
vec3 calcDirectionLight(directionLight light, vec3 normal, vec3 viewDirection);
vec3 calcPointLight(pointLight light, vec3 normal, vec3 fPosition, vec3 viewDirection);
vec3 calcSpotLight(spotLight light, vec3 normal, vec3 fPosition, vec3 viewDirection);

void main() {
    //properties
    vec3 normal = normalize(fNormal);
    vec3 viewDirection = normalize(viewPosition - fPosition);

    /*
    * Our lighting is set up in 3 phases: directional, point lights and an optional flashlight
    * For each phase, a calculate function is defined that calculates the corresponding color
    * per lamp. In the main() function we take all the calculated colors and sum them up for
    * this fragment's final color.
    */
    //first: directional lighting
    vec3 result = calcDirectionLight(myDirLight, normal, viewDirection);
    //second point light
    for (int i = 0; i < NR_POINT_LIGHTS; i++) {
        result += calcPointLight(myPointLights[i], normal, fPosition, viewDirection);
    }
    //thrid spot light
    result += calcSpotLight(mySpotLight, normal, fPosition, viewDirection);

    fColor = vec4(result, 1.0);
}

vec3 calcDirectionLight(directionLight light, vec3 normal, vec3 viewDirection)
{
    vec3 lightDirection = normalize(-light.direction);
    //diffuse
    float diff = max(dot(normal, lightDirection), 0.0);
    //specular
    vec3 reflectDirection = reflect(-lightDirection, normal);
    float spec = pow(max(dot(reflectDirection, viewDirection), 0.0), material.shininess);
    //mix
    vec3 ambient = light.ambient * vec3(texture(material.diffTexture, fTexCoords));
    vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffTexture, fTexCoords));
    vec3 specular = light.specular * spec * vec3(texture(material.specTexture, fTexCoords));
    return (ambient + diffuse + specular);
}

vec3 calcPointLight(pointLight light, vec3 normal, vec3 fPosition, vec3 viewDirection)
{
    vec3 lightDirection = normalize(light.position - fPosition);
    //diffuse
    float diff = max(dot(normal, lightDirection), 0.0);
    //specular
    vec3 reflectDirection = reflect(-lightDirection, normal);
    float spec = pow(max(dot(reflectDirection, viewDirection), 0.0), material.shininess);
    //attenuation
    float distance = length(light.position - fPosition);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    //mix
    vec3 ambient = light.ambient * vec3(texture(material.diffTexture, fTexCoords));
    vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffTexture, fTexCoords));
    vec3 specular = light.specular * spec * vec3(texture(material.specTexture, fTexCoords));

    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;

    return (ambient + diffuse + specular);
}

vec3 calcSpotLight(spotLight light, vec3 normal, vec3 fPosition, vec3 viewDirection)
{
    vec3 lightDirection = normalize(light.position - light.direction);
    //diffuse
    float diff = max(dot(normal, lightDirection), 0.0);
    //specular
    vec3 reflectDirection = reflect(-lightDirection, normal);
    float spec = pow(max(dot(reflectDirection, viewDirection), 0.0), material.shininess);

    //attenuation
    float distance = length(light.position - fPosition);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    //spot light intensity
    float theta = dot(lightDirection, normalize(-light.direction));
    float epsilon = light.cutOff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);

    //mix
    vec3 ambient = light.ambient * vec3(texture(material.diffTexture, fTexCoords));
    vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffTexture, fTexCoords));
    vec3 specular = light.specular * spec * vec3(texture(material.specTexture, fTexCoords));

    ambient *= attenuation * intensity;
    diffuse *= attenuation * intensity;
    specular *= attenuation * intensity;

    return (ambient + diffuse + specular);
}

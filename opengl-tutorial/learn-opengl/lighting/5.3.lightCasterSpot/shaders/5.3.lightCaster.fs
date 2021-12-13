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

    vec3 lightDir = normalize(light.position - fPosition);

    //check if lighting is inside the spotlight core
    float theta = dot(lightDir, normalize(-light.direction));
    //if the object in torch light angle
    if(theta > light.cutOff) {
        //ambient
        vec3 ambient = light.ambient * texture(material.diffTexture, fTexCoords).rgb;

        //diffuse
        vec3 normal = normalize(fNormal);
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 diffuse = light.diffuse * diff * texture(material.diffTexture, fTexCoords).rgb;

        //specular
        vec3 viewDir = normalize(viewPosition - fPosition);
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
        vec3 specular = light.specular * spec * texture(material.specTexture, fTexCoords).rgb;

        //attenuation
        float distance = length(light.position - fPosition);
        float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));

        // remove attenuation from ambient, as otherwise at large distances the light would be darker inside than outside the
        diffuse *= attenuation;
        specular *= attenuation;

        vec3 result = ambient + diffuse + specular;
        fColor = vec4(result, 1.0);
    } else {
        // else, use ambient light so scene isn't completely dark outside the spotlight.
        fColor = vec4(light.ambient * texture(material.diffTexture, fTexCoords).rgb, 1.0);
    }

}
#version 330 core

out vec4 fColor;

in VS_OUT {
    vec3 fragPosition;
    vec3 fNormal;
    vec2 fTexCoord;
} fsIn;

uniform sampler2D floorTexture;

uniform vec3 lightPositions[4];
uniform vec3 lightColors[4];
uniform vec3 viewPosition;
uniform bool gamma;

vec3 blinnPhong(vec3 normal, vec3 fragPos, vec3 lightPos, vec3 lightColor)
{
    //diffuse
    vec3 lightDirection = normalize(lightPos - fragPos);
    float diff = max(dot(lightDirection, normal), 0.0);
    vec3 diffuse = diff * lightColor;
    //specular
    vec3 viewDirection = normalize(viewPosition - fragPos);
    vec3 reflectDirection = reflect(-lightPos, normal);
    float spec = 0.0;
    vec3 halfwayDirection = normalize(lightDirection + viewDirection);
    spec = pow(max(dot(normal, halfwayDirection), 0.0), 64.0);
    vec3 specular = spec * lightColor;
    //simple attenuation
    float maxDistance = 1.5;
    float distance = length(lightPos - fragPos);
    float attenuation = 1.0 / (gamma ? distance * distance : distance);

    diffuse *= attenuation;
    specular *= attenuation;

    return diffuse + specular;
}

void main(){
    vec3 color = texture(floorTexture, fsIn.fTexCoord).rgb;
    vec3 lighting = vec3(0.0);

    for (int i = 0; i < 4 ; i++) {
        lighting += blinnPhong(normalize(fsIn.fNormal), fsIn.fragPosition, lightPositions[i], lightColors[i]);
    }

    color *= lighting;
    if (gamma) {
        color = pow(color, vec3(1.0 / 2.2));
    }

    fColor = vec4(color, 1.0);
}
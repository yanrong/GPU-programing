#version 330 core

out vec4 fColor;
in VS_OUT {
    vec3 fragPos;
    vec3 normal;
    vec2 texCoord;
    vec4 fragPosLightSpace;
} fsIn;

uniform sampler2D diffuseTexture;
uniform sampler2D shadowMap;
uniform vec3 lightPosition;
uniform vec3 viewPosition;

float shadowCalculation(vec4 fragPosLightSpace)
{
    //perform perspective divide
    vec3 projCoord = fragPosLightSpace.xyz / fragPosLightSpace.w;
    //transform to [0,1] range
    projCoord = projCoord * 0.5 + 0.5; // origin in [-1, 1], x / 2 in [-0.5 , 0.5]
    //get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(shadowMap, projCoord.xy).r;
    //get depth of current fragment from light's perspective
    float currentDepth = projCoord.z;

    //calculate bias (base on depth map resolution and slope)
    vec3 normal = normalize(fsIn.normal);
    vec3 lightDirection = normalize(lightPosition - fsIn.fragPos);
    float bias = max(0.05 * (1.0 - dot(normal, lightDirection)), 0.005);
    //check whether current frag position is in shadow
    //float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;
    //PFC
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float pcfDepth = texture(shadowMap, projCoord.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }
    shadow /= 9.0;

    if (projCoord.z > 1.0) {
        shadow = 0.0;
    }

    return shadow;
}

void main(){
    vec3 color = texture(diffuseTexture, fsIn.texCoord).rgb;
    vec3 normal = normalize(fsIn.normal);
    vec3 lightColor = vec3(0.3);

    //ambient
    vec3 ambient = 0.3 * lightColor;
    //diffuse
    vec3 lightDirection = normalize(lightPosition - fsIn.fragPos);
    float diff = max(dot(lightDirection, normal), 0.0);
    vec3 diffuse = diff * lightColor;
    //specular
    vec3 viewDirection = normalize(viewPosition - fsIn.fragPos);
    vec3 reflectDirection = reflect(-lightDirection, normal);
    float spec = 0.0;
    vec3 halfWayDirection = normalize(lightDirection + viewDirection);
    spec = pow(max(dot(normal, halfWayDirection), 0.0), 64.0);
    vec3 specular = spec * lightColor;

    //calculate shadow
    float shadow = shadowCalculation(fsIn.fragPosLightSpace);
    vec3 lighting = (ambient + (1.0 - shadow) * (diffuse + specular)) * color;

    fColor = vec4(lighting, 1.0);
}
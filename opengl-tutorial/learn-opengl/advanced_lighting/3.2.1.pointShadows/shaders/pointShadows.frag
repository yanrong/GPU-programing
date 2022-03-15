#version 330 core

out vec4 fragColor;
in VS_OUT {
    vec3 fragPosition;
    vec3 normal;
    vec2 texCoord;
} fsIn;

uniform sampler2D diffuseTexture;
uniform samplerCube depthMap;

uniform vec3 lightPosition;
uniform vec3 viewPosition;
uniform float farPlane;
uniform bool shadows;

float shadowCalculation(vec3 fragPosition)
{
    //get vector between fragment position and light position
    vec3 fragToLight = fragPosition - lightPosition;
    //is the fragment to light vector to sample from the depth map
    float closestDepth = texture(depthMap, fragToLight).r;
    //it is currently in linear range between [0,1], restore it to origin value
    closestDepth *= farPlane;
    //now get current linear depth as the length between the fragment and light position
    float currentDepth = length(fragToLight);
    //test for shadows
    float bias = 0.05; //we use a much larger bias since depth is now in [nearPlane, farPlane] range
    float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;

    //FOR DEBUG :display closestDepth as DEBUG
    //fragColor = vec4(vec3(closestDepth / farPlane), 1.0);

    return shadow;
}

void main(){
    vec3 color = texture(diffuseTexture, fsIn.texCoord).rgb;
    vec3 normal = normalize(fsIn.normal);
    vec3 lightColor = vec3(0.3);

    //ambient
    vec3 ambient = 0.3 * lightColor;
    //diffuse
    vec3 lightDirection = normalize(lightPosition - fsIn.fragPosition);
    float diff = max(dot(lightDirection, normal), 0.0);
    vec3 diffuse = diff * lightColor;
    //specular
    vec3 viewDirection = normalize(viewPosition - fsIn.fragPosition);
    vec3 reflectDirection = reflect(-lightDirection, normal);
    float spec = 0.0;
    vec3 halfwayDirection = normalize(lightDirection + viewDirection);
    spec = pow(max(dot(normal, halfwayDirection), 0.0), 64.0);
    vec3 specular = spec * lightColor;

    //calculate shadow
    float shadow = shadows ? shadowCalculation(fsIn.fragPosition) : 0.0;
    vec3 lighting = (ambient + (1.0 - shadow) * (diffuse + specular)) * color;

    fragColor = vec4(lighting, 1.0);
}


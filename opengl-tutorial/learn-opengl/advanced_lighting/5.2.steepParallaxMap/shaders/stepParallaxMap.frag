#version 330 core

out vec4 fragColor;

in VS_OUT{
    vec3 fragPosition;
    vec2 texCoord;
    vec3 tangentLightPosition;
    vec3 tangentViewPosition;
    vec3 tangentFragPosition;
} fsIn;

uniform sampler2D diffuseMap;
uniform sampler2D normalMap;
uniform sampler2D depthMap;

uniform float heightScale;

vec2 parallaxMap(vec2 texCoord, vec3 viewDirection)
{
    //number of depth layers
    const float minLayers = 8;
    const float maxLayers = 32;
    //interpolate in range [max min] according to view direction and normal
    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0, 0.0, 1.0), viewDirection)));
    //calculate the size of  each layers
    float layerDepth = 1.0 / numLayers;
    //depth of current layer
    float currentLayderDepth = 0.0;
    //the amount to shift the texture coordinate per layer(from the vector P)
    vec2 P = viewDirection.xy / viewDirection.z * heightScale;
    vec2 deltaTexCoord = P / numLayers;

    //get initial values
    vec2 currentTexCoord = texCoord;
    float currentDepthMapValue = texture(depthMap, currentTexCoord).r;

    while (currentLayderDepth < currentDepthMapValue) {
        //shift texture coordinates along direction of P
        currentTexCoord -= deltaTexCoord;
        //get depthmap value a current texture
        currentDepthMapValue = texture(depthMap, currentTexCoord).r;
        //get dept of next layer
        currentLayderDepth += layerDepth;
    }

    return currentTexCoord;
}

void main(){
    //offset texture coordinates with parallax map
    vec3 viewDirection = normalize(fsIn.tangentViewPosition - fsIn.tangentFragPosition);
    vec2 texCoord = fsIn.texCoord;

    texCoord = parallaxMap(fsIn.texCoord, viewDirection);
    if (texCoord.x > 1.0 || texCoord.y > 1.0 || texCoord.x < 0.0 || texCoord.y < 0.0) {
        discard;
    }
    //obtain normal from normal map
    vec3 normal = texture(normalMap, texCoord).rgb;
    //transform normal vector to[-1, 1]
    normal = normalize(normal * 2.0 - 1.0);// this normal is in tangent space

    //get diffuse color
    vec3 color = texture(diffuseMap, texCoord).rgb;
    //ambient
    vec3 ambient = 0.1 * color;
    //diffuse
    vec3 lightDirection = normalize(fsIn.tangentLightPosition - fsIn.tangentFragPosition);
    float diff = max(dot(lightDirection, normal), 0.0);
    vec3 diffuse = diff * color;
    //specular
    vec3 reflectDirection = reflect(-lightDirection, normal);
    vec3 halfwayDirection = normalize(lightDirection + viewDirection);
    float spec = pow(max(dot(normal, halfwayDirection), 0.0), 32.0);

    vec3 specular = vec3(0.2) * spec;
    fragColor = vec4(ambient + diffuse + specular, 1.0);
}
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
    float height = texture(depthMap, texCoord).r;
    return texCoord - viewDirection.xy * (height * heightScale);
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
    vec3 reflaceDirection = reflect(-lightDirection, normal);
    vec3 halfwayDirection = normalize(lightDirection + viewDirection);
    float spec = pow(max(dot(normal, halfwayDirection), 0.0), 32.0);

    vec3 specular = vec3(0.2) * spec;
    fragColor = vec4(ambient + diffuse + specular, 1.0);
}
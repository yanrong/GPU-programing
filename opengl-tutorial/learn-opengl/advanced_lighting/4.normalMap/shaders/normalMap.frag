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

uniform vec3 lightPosition;
uniform vec3 viewPosition;

void main(){
    //obtain normal from normal map in range [0,1]
    vec3 normal = texture(normalMap, fsIn.texCoord).rgb;
    //transform normal vector to[-1, 1]
    normal = normalize(normal * 2.0 - 1.0);// this normal is in tangent space

    //get diffuse color
    vec3 color = texture(diffuseMap, fsIn.texCoord).rgb;
    //ambient
    vec3 ambient = 0.1 * color;
    //diffuse
    vec3 lightDirection = normalize(fsIn.tangentLightPosition - fsIn.tangentFragPosition);
    float diff = max(dot(lightDirection, normal), 0.0);
    vec3 diffuse = diff * color;
    //specular
    vec3 viewDirection = normalize(fsIn.tangentViewPosition - fsIn.tangentFragPosition);
    vec3 reflaceDirection = reflect(-lightDirection, normal);
    vec3 halfwayDirection = normalize(lightDirection + viewDirection);
    float spec = pow(max(dot(normal, halfwayDirection), 0.0), 32.0);

    vec3 specular = vec3(0.2) * spec;
    fragColor = vec4(ambient + diffuse + specular, 1.0);
}
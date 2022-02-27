#version 330 core

out vec4 fColor;

in VS_OUT {
    vec3 fragPosition;
    vec3 Normal;
    vec2 texCoord;
} fsIn;

uniform sampler2D floorTexture;
uniform vec3 lightPosition;
uniform vec3 viewPosition;
uniform bool blinn;

void main(){
    vec3 color = texture(floorTexture, fsIn.texCoord).rgb;
    //ambient
    vec3 ambient = 0.05 * color;
    //diffuse
    vec3 lightDirection = normalize(lightPosition - fsIn.fragPosition);
    vec3 normal = normalize(fsIn.Normal);
    float diff = max(dot(lightDirection, normal), 0.0);
    vec3 diffuse = diff * color;
    //specular
    vec3 viewDirection = normalize(viewPosition - fsIn.fragPosition);
    vec3 reflectDirection = reflect(-lightDirection, normal);
    float spec = 0.0;

    if (blinn) {
        vec3 halfwayDirection = normalize(lightDirection + viewDirection);
        spec = pow(max(dot(normal, halfwayDirection), 0.0), 32.0);
    } else {
        vec3 reflectDirection = reflect(-lightDirection, normal);
        spec = pow(max(dot(viewDirection, reflectDirection), 0.0), 8.0);
    }

    vec3 specular = vec3(0.3) * spec; //assuming bright white light color
    fColor = vec4(ambient + diffuse + specular, 1.0);
}
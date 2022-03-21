#version 330 core
out vec4 fragColor;

in VS_OUT {
    vec3 fragPosition;
    vec3 normal;
    vec2 texCoord;
} fsIn;

struct light {
    vec3 Position;
    vec3 Color;
};

uniform light lights[16];
uniform sampler2D diffuseTexture;
uniform vec3 viewPosition;

void main(){
    vec3 color = texture(diffuseTexture, fsIn.texCoord).rgb;
    vec3 normal = normalize(fsIn.normal);

    //ambient
    vec3 ambient = 0.0 * color;
    //lightin
    vec3 lighting = vec3(0.0);
    for (int i = 0; i < 16 ; i++) {
        //diffuses
    }
}
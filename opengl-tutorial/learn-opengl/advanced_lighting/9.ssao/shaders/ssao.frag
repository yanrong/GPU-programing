#version 330 core

out float fragColor;

in vec2 texCoord;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D texNoise;

uniform vec3 samples[64];

int kernelSize = 64;
float radius = 0.5;
float bias = 0.025;

//tile noise texture over screen based on screen dimensions divided by noise size
const vec2 noiseScale = vec2(800.0 / 4.0, 600.0 / 4.0);

uniform mat4 projection;

void main(){
    //get input for SSAO algorithm
    vec3 fragPosition = texture(gPosition, texCoord).xyz;
    vec3 normal = normalize(texture(gNormal, texCoord).rgb);
    vec3 randomVec = normalize(texture(texNoise, texCoord * noiseScale).xyz);
    //create TBN change of basis matrix: from tangent-space to view-space
    vec3 tangent = normalize(randomVec - normal * dot(randomVec , normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);

    //iterate over the sample kernel and calculate occlusion factor
    float occlusion = 0.0;
    for (int i = 0; i < kernelSize; i++) {
        //get sample position
        vec3 samplePosition = TBN * samples[i]; // from tagent space to view space
        samplePosition = fragPosition + samplePosition * radius;

        //projection sample position(to sample texture, to get position on screen/texture)
        vec4 offset = vec4(samplePosition, 1.0);
        offset = projection * offset; //from view to clip-space
        offset.xyz /= offset.w; // perspective divide
        offset.xyz = offset.xyz * 0.5 + 0.5; // transform to 0.0 - 1.0

        //get sample depth
        float sampleDepth = texture(gPosition, offset.xy).z;// get depth value of kernel sample
        //range check and accumulate
        float rangeCheck = smoothstep(0.0, 1.0, radius / abs(fragPosition.z - sampleDepth));
        occlusion += (sampleDepth >= samplePosition.z + bias ? 1.0 : 0.0) * rangeCheck;
    }

    occlusion = 1.0 - (occlusion / kernelSize);

    fragColor = occlusion;
}
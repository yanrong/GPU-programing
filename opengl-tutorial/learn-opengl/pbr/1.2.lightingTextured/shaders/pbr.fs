#version 330 core

out vec4 fragColor;

in vec2 texCoord;
in vec3 worldPosition;
in vec3 normal;

//material parameters
uniform sampler2D albedoMap;
uniform sampler2D normalMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;

//lights
uniform vec3 lightPosition[4];
uniform vec3 lightColor[4];

uniform vec3 camPosition;

const float PI = 3.14159265359;
// Easy trick to get tangent-normals to world-space to keep PBR code simplified.
// Don't worry if you don't get what's going on; you generally want to do normal
// mapping the usual way for performance anways; I do plan make a note of this
// technique somewhere later in the normal mapping tutorial
vec3 getNormalFromMap()
{
    vec3 tangentNormal = texture(normalMap, texCoord).xyz * 2.0 - 1.0;

    vec3 Q1 = dFdx(worldPosition);
    vec3 Q2 = dFdy(worldPosition);
    vec2 st1 = dFdx(texCoord);
    vec2 st2 = dFdy(texCoord);

    vec3 N = normalize(normal);
    vec3 T = normalize(Q1 * st2.t - Q2 * st1.t);
    vec3 B = -normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * tangentNormal);
}

//calculate normal distribution function D with Trowbridge-Reitz GGX
float distributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom = a2;
    float denom = (NdotH2 *(a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

//calculate Geometry function with Schlick-GGX combination the GGX and Schlick-Beckmann
float geometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0); //k direct
    float k = (r * r) / 8.0;

    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);

    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

//Fresnel equation
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main(){

    vec3 albedo = pow(texture(albedoMap, texCoord).rgb, vec3(2.2));
    float metallic = texture(metallicMap, texCoord).r;
    float roughness = texture(roughnessMap, texCoord).r;
    float ao = texture(aoMap, texCoord).r;
    vec3 N = getNormalFromMap();
    vec3 V = normalize(camPosition - worldPosition);

    //calculate reflectance at normal incidence; if dia-electric(like plastic) use F0
    // of 0.04 and if it's a metal, use the albedo color as F0(metallic workflow)
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic); //interpolate

    //reflectance equation
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < 4; i++) {
        //calculate per-light radiance
        vec3 L = normalize(lightPosition[i] - worldPosition);
        vec3 H = normalize(V + L); //halfway vector
        float distance = length(lightPosition[i] - worldPosition);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = lightColor[i] * attenuation;

        //Cook-Torrance BRDF(bidirectional reflective distribution function)
        float NDF = distributionGGX(N, H ,roughness);
        float G = geometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 numerator = NDF * G * F;
        // + 0.0001 to prevent divide by zero
        float denominator = 4 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular = numerator / denominator;

        // KS is equal to Fresnel
        vec3 kS = F;
        // for energy conservation, the diffuse and specular light can't be above 1.0
        //(unless the surface emits light); to preserve this relationship the diffuse
        // component (kD) should equal 1.0 - kS.
        vec3 kD = vec3(1.0) - kS;
        // multiply kD by the inverse metalness such that only non-metals have diffuse
        //lighting, or a linear blend if partly metal (pure metals have no diffuse light).
        kD *= 1.0 - metallic;

        //scale light by NdotL
        float NdotL = max(dot(N, L), 0.0);

        //add to outgoing radiance Lo
        //note that we already multiplied the BRDF by the Fresnel(kS) so we won't multiply by kS again
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;
    }

    //ambient lighting(note that the nex IBL tutorial will replace this ambient lighting with enviroment lgiting)
    vec3 ambient = vec3(0.03) * albedo * ao;
    vec3 color = ambient + Lo;

    //HDR tonemapping
    color = color / (color + vec3(1.0));
    //gamma correct
    color = pow(color, vec3(1.0 / 2.2));

    fragColor = vec4(color, 1.0);
}

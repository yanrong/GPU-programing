#version 330 core

out vec4 fragColor;
in vec3 worldPosition;

uniform samplerCube environmentMap;

void main()
{
    vec3 envColor = texture(environmentMap, worldPosition).rgb;

    //HDR tonemap and gamma correct
    envColor = envColor / (envColor + vec3(1.0));
    envColor = pow(envColor, vec3(1.0 / 2.2));

    fragColor = vec4(envColor, 1.0);
}

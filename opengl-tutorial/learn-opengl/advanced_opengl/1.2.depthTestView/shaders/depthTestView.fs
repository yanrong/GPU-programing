#version 330 core

out vec4 fColor;

uniform sampler2D texture1;

float near = 0.1;
float far = 100.0;

float linearizeDepth(float depth)
{
    float z = depth * 2.0 - 1.0; // to NDC
    return (2.0 * near * far) / (far + near - z * (far - near));
}

void main(){
     // divide by far to get depth in range [0,1] for visualization purposes
    float depth = linearizeDepth(gl_FragColor.z) / far;
    fColor = vec4(vec3(depth), 1.0);
}
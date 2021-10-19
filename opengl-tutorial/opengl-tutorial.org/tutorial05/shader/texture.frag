#version 330 core
//Interpolate value from the vertices shader
in vec2 uv;
//Output data
out vec3 color;
//Values that stay constant for the whole mesh
uniform sampler2D textureSample;

void main(){
    //Output color = color of the texture at special UV
    color = texture(textureSample, uv).rgb;
}
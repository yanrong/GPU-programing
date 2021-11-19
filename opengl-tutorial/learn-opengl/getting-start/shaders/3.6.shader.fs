#version 330 core
out vec4 fragColor;
in vec3 ourPosition;

void main()
{
     // note how the position value is linearly interpolated to get all the different colors
    fragColor = vec4(ourPosition, 1.0);
}

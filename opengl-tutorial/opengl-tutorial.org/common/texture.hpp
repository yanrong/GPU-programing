#ifndef TEXTURE_HPP
#define TEXTURE_HPP
//Build texture from BMP file
GLuint loadBMP(const char *path);

/*Since GLFW3, glfwLoadTexture2D() has been removed, you have to use another loading library
* or do it yourself , Loadd a .TAG file using GLFW's own loader
* GLuint loadTag_glfw(const char * image);
*/

//Load a .DDS file use GLFW's own loader
GLuint loadDDS(const char *path);
#endif
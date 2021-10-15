#include <stdio.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "texture.hpp"

GLuint loadbBMP(const char * path)
{
    unsigned char header[54]; // Each BMP file begins by a 54-bytes header
    unsigned int dataPos; //the offset at beginning of file indicate valild data
    unsigned int width, height;
    unsigned int imgSize;
    unsigned char * data; //Actual RGB data

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        printf("image file could not be opened\n");
        return 0;
    }
    //Must be sure the file header is exist
    if (fread(header, 1, 54, fp) != 54) {
        printf("not a BMP file\n");
        fclose(fp);
        return 0;
    }

    //Check the bmp file
    if (header[0] != 'B' || header[1] != 'M') {
        printf("invalid BMP header \n");
        fclose(fp);
        return 0;
    }
    //Make sure this is a  24bpp format
    if( *(int *)&(header[0x1E]) != 0) { //no pixel array compression used
        printf("Not a correct BMP(compressed) file\n");
        fclose(fp);
        return 0;
    }
    
    if( *(int *)&(header[0x1C]) != 0) { // number of per pixel byte
        printf("Not a correct BMP BPP file\n");
        fclose(fp);
        return 0;
    }

    dataPos = *(int *)&(header[0x0A]); //start from header next
    imgSize = *(int *)&(header[0x22]); //Size of the raw bitmap data (including padding)
    width   = *(int *)&(header[0x12]); //Width
    height  = *(int *)&(header[0x16]); //Head

    //Some BMP file are misformatted, guess missig information
    if (imgSize == 0) imgSize = width * height * 3;
    if (dataPos == 0) dataPos = 54; // The BMP header is done that way

    //Create a buffer
    data =new unsigned char [imgSize];

    //Read the actual data from the file into buffer
    fread(data, 1, imgSize, fp);
    fclose(fp);

    //Create one OpenGL texture
    GLuint textureID;
    glGenTextures(1, &textureID);

    //Bind the nealy crated texture.all future texture functions will modify this texture
    glBindTexture(GL_TEXTURE_2D, textureID);
    //Give the image to OpenGL
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, data);
    //copy the data finished, free source
    delete []data;
    


}
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "texture.hpp"

/*
* For get the correctly source data from specific file to OpenGL texture
* The Following two sinpet code show how to read the raw data from BMP file
* and read from DSS file
*/
GLuint loadBMP(const char * path)
{
    unsigned char header[54]; // Each BMP file begins by a 54-bytes header
    unsigned int dataPos; //the offset at beginning of file indicate valild data
    unsigned int width, height;
    unsigned int imgSize;
    unsigned char * data; //Actual RGB data

    printf("Reading image %s, \n", path);
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
    
    if( *(int *)&(header[0x1C]) != 24) { // number of per pixel byte
        printf("Not a correct BMP BPP file\n");
        fclose(fp);
        return 0;
    }

    dataPos = *(int *)&(header[0x0A]); //start from header next
    imgSize = *(int *)&(header[0x22]); //Size of the raw bitmap data (including padding)
    width   = *(int *)&(header[0x12]); //Width
    height  = *(int *)&(header[0x16]); //Head

    //Some BMP file are misformatted, guess missig information, 3 -RGB componet
    if (imgSize == 0) imgSize = width * height * 3;
    if (dataPos == 0) dataPos = 54; // The BMP header is done that way

    //Create a buffer
    data = new unsigned char [imgSize];

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
    // Poor filtering, or ...
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    // setup for Texture filter
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    //Which require mipmaps. Generate them automatically.
    glGenerateMipmap(GL_TEXTURE_2D);

    return textureID;
}

/*
* Since GLFW 3, glfwLoadTexture2D() has been removed. You have to another texture loading library
* The general way to build a texture by following steps
* //Create One OpenGL texture
* GLuint textureID;
* glGenTexture(1, &textureID);
* //"Bind" the newly created texture, all further texture functions will modify this texture
* glBindTexture(GL_TEXTURE_2D, textureID);
*
* //Read the file, call glTexImage2D() with right paramters
* glfwLoadTexture2D(path, 0);
* //Nice trilinear filtering.
* glTexParmeteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
* glTexParmeteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
* glTexParmeteri(GL_TEXTURE_2D, GL_TEXTURE_MGA_FILTER, GL_LINEAR);
* glTexParmeteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
* glGenerateMipmap(GL_TEXTURE_2D);
* // Return the ID of the exture we just created
* return textureID;
*/


/*Load texture data from DDS file*/
#define FOURCC_DXT1 0x31545844
#define FOURCC_DXT3 0x33545844
#define FOURCC_DXT5 0x35545844

GLuint loadDDS(const char *path)
{
    unsigned char header[124];

    FILE *fp;

    /* try to open the file */
    fp = fopen(path, "rb");
    if (fp == NULL) {
        printf("%s couold not be opened\n, You must ensure the file is exist and correctly", path);
        return 0;
    }

    char filecode[4];
    fread(filecode, 1, 4, fp);
    if(strncmp(filecode, "DDS ", 4) != 0){
        fclose(fp);
        return 0;
    }

    /* ??? get the surface dec */
    fread(&header, 124, 1, fp);

    unsigned int height         = *(unsigned int *)&(header[8]);
    unsigned int width          = *(unsigned int *)&(header[12]);
    unsigned int linearSize     = *(unsigned int *)&(header[16]);
    unsigned int mipMapCount    = *(unsigned int *)&(header[24]);
    unsigned int fourCC         = *(unsigned int *)&(header[80]);

    unsigned char *buffer;
    unsigned int bufSize;
    /* how big is it going to be includeing all mipmaps? */
    bufSize = mipMapCount > 1 ? linearSize * 2 : linearSize;
    buffer = (unsigned char *)malloc(bufSize * sizeof(unsigned char));
    fread(buffer, 1, bufSize, fp);
    /* close the file pointer */
    fclose(fp);

    unsigned int components = (fourCC == FOURCC_DXT1) ? 3 : 4;
    unsigned int format;
    switch (fourCC)
    {
    case FOURCC_DXT1:
        format = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
        break;
    case FOURCC_DXT3:
        format = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
        break;
    case FOURCC_DXT5:
        format = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
        break;    
    default:
        free(buffer);
        break;
    }

    // Create  one OpenGL texture
    GLuint textureID;
    glGenTextures(1, &textureID);
    // Bind the newly created texture: all future texture functions will modify this texture
    glBindTexture(GL_TEXTURE_2D, textureID);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    unsigned int blockSize = (format == GL_COMPRESSED_RGBA_S3TC_DXT1_EXT) ? 8 : 16;
    unsigned int offset = 0;

    /* load the mipmaps */
    for(unsigned int level = 0; level < mipMapCount && (width || height); level++)
    {
        unsigned int size = ((width + 3) /4) * ((height + 3 ) /4) * blockSize;
        glCompressedTexImage2D(GL_TEXTURE_2D, level, format, width, height,
            0, size, buffer + offset);
        
        offset += size;
        width   /= 2;
        height  /= 2;

        //Deal with Non-Power-Of-Two textures. This code is not included in webpage to reduce clutter
        if (width < 1) width = 1;
        if (height < 1) height = 1;
    }

    free(buffer);
    return textureID;
}
#include <stdio.h>
#include <stdlib.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../common/shader.hpp"
#include "../common/texture.hpp"
#include "../common/controls.hpp"

using namespace glm;

// Our vertices. Three consecutive floats give a 3D vertex; Three consecutive vertices give a triangle.
// A cube has 6 faces with 2 triangles each, so this makes 6*2=12 triangles, and 12*3 vertices
static const GLfloat g_vertex_buffer_data[] = {
    -1.0f,-1.0f,-1.0f, // triangle 1 : begin
    -1.0f,-1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f, // triangle 1 : end
     1.0f, 1.0f,-1.0f, // triangle 2 : begin
    -1.0f,-1.0f,-1.0f,
    -1.0f, 1.0f,-1.0f, // triangle 2 : end
     1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f,-1.0f,
     1.0f,-1.0f,-1.0f,
     1.0f, 1.0f,-1.0f,
     1.0f,-1.0f,-1.0f,
    -1.0f,-1.0f,-1.0f,
    -1.0f,-1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f,-1.0f,
     1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f,-1.0f, 1.0f,
     1.0f,-1.0f, 1.0f,
     1.0f, 1.0f, 1.0f,
     1.0f,-1.0f,-1.0f,
     1.0f, 1.0f,-1.0f,
     1.0f,-1.0f,-1.0f,
     1.0f, 1.0f, 1.0f,
     1.0f,-1.0f, 1.0f,
     1.0f, 1.0f, 1.0f,
     1.0f, 1.0f,-1.0f,
    -1.0f, 1.0f,-1.0f,
     1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f,
     1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f,
     1.0f,-1.0f, 1.0f
};

// Two UV coordinatesfor each vertex. They were created with Blender.
static const GLfloat g_uv_buffer_data[] = {
	0.000059f, 1.0f-0.000004f,
	0.000103f, 1.0f-0.336048f,
	0.335973f, 1.0f-0.335903f,
	1.000023f, 1.0f-0.000013f,
	0.667979f, 1.0f-0.335851f,
	0.999958f, 1.0f-0.336064f,
	0.667979f, 1.0f-0.335851f,
	0.336024f, 1.0f-0.671877f,
	0.667969f, 1.0f-0.671889f,
	1.000023f, 1.0f-0.000013f,
	0.668104f, 1.0f-0.000013f,
	0.667979f, 1.0f-0.335851f,
	0.000059f, 1.0f-0.000004f,
	0.335973f, 1.0f-0.335903f,
	0.336098f, 1.0f-0.000071f,
	0.667979f, 1.0f-0.335851f,
	0.335973f, 1.0f-0.335903f,
	0.336024f, 1.0f-0.671877f,
	1.000004f, 1.0f-0.671847f,
	0.999958f, 1.0f-0.336064f,
	0.667979f, 1.0f-0.335851f,
	0.668104f, 1.0f-0.000013f,
	0.335973f, 1.0f-0.335903f,
	0.667979f, 1.0f-0.335851f,
	0.335973f, 1.0f-0.335903f,
	0.668104f, 1.0f-0.000013f,
	0.336098f, 1.0f-0.000071f,
	0.000103f, 1.0f-0.336048f,
	0.000004f, 1.0f-0.671870f,
	0.336024f, 1.0f-0.671877f,
	0.000103f, 1.0f-0.336048f,
	0.336024f, 1.0f-0.671877f,
	0.335973f, 1.0f-0.335903f,
	0.667969f, 1.0f-0.671889f,
	1.000004f, 1.0f-0.671847f,
	0.667979f, 1.0f-0.335851f
};

int main(int argc, char *argv[])
{
    GLFWwindow *window;

    GLuint vertexArrayID;
    GLuint vertexBuffer;
    GLuint uvBuffer;

    GLuint programID;
    GLuint matrixID;
    GLuint texture;
    GLuint textureID;

    // Initialise GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialization GLFW\n");
        getchar();
        return -1;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); //Make MacOS happy, fuck it
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Open a window and create its OpenGL context
    window = glfwCreateWindow(1024, 600, "Tutorial 06 - KeyBoard and Mouse", NULL, NULL);
    if (window == NULL) {
        fprintf(stderr, "Failed to openg GLFW, If you have an Intel GPU, they are not 3.3 compatible.
        Try the 2.1 version of the tutorials.\n");
        getchar();
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialze GLFW
    glewExperimental = true; //Needed for core profile
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialzed GLEW\n");
        getchar();
        glfwTerminate();
        return -1;
    }
    // Ensuer we can handle the escape key
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    // Hind the mouse and enable umlimited mouvement
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    // Set the mouse at the center of the screen
    glfwPollEvents();
    glfwSetCursorPos(window, 1024/2, 768 /2);

    // Dark blue background
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    // Access fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);
    // Cull triangles which normal is not towards the camera
    glEnable(GL_CULL_FACE);

    // Create vertext array for hold the predefined data
    glGenVertexArray(1, &vertexArrayID);
    glBindVertexArray(vertexArrayID);
    /*************************************************/
    // Create vertext buffer and UV buffer
    glGenBuffers(1, &vertexBuffer);
    // Associate it with the OpenGL
    glBindBUffer(GL_ARRAY_BUFFER, vertexBuffer);
    // Fill it with the data
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);
    /*************************************************/
    glGenBuffers(1, &uvBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_uv_buffer_data), g_uv_buffer_data, GL_STATIC_DRAW);

    // Create and compile our GLSL program from the shaders
    programID = loadShader("./shader/transform.vert", "./shader/texture.frag");
    // Get a handle for out "mvp" uniform
    matrixID = glGetUniformLocation(programID, "mvp");
    // Load the texture
    texture = loadDDS("uvtemplate.DDS");
    // Get a handle for our texture uniform
    textureID = glGetUniformLocation(programID. "texture");

    do {
        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // Use our shader
        glUseProgram(programID);

        // Compute the MVP matrix from the keybaord and mouse
        computeMatrixFromInputs();
        glm::mat4 projectionMatrix = getProjectionMatrix();
        glm::mat4 viewMatrix = getViewMatrix;
        glm::mat4 modelMatrix = glm::mat4(1, 0);
        glm::mat4 mvp = projectionMatrix * viewMatrix *modelMatrix;

        // Send out transformation to the current bound shade,in the "MVP" uniform
        glUniformMatrix4fv(matrixID, 1, GL_FALSE, &mvp[0][0]);
        //Bind out texture in Texture Unit 0
        glActiveTextrue(GL_TEXTURE);
        glBindTexture(GL_TEXTURE_2D, texture);
        // Set our "textureSampleer" sampler to use Texture unit 0
        glUniform1i(textureID, 0);

        // 1st attribute buffer: vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
        glVertexAttribPointer(
            0,          // attribute. No particular reason for 0, but must match the layout in the shader.
            3,          // size
            GL_FLOAT,   // type
            GL_FALSE,   //normal
            0,          //stride
            (void *)0   //array buffer offset
        );

        // 2st attribute buffer: UVs
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
        glVertexAttribPointer(
            1,          // attribute. No particular reason for 0, but must match the layout in the shader.
            2,          // size
            GL_FLOAT,   // type
            GL_FALSE,   //normal
            0,          //stride
            (void *)0   //array buffer offset
        );

        //Draw the triangle !
        glDrawArrays(GL_TRIANGLES, 0 , 12 * 3);  // 12*3 indices starting at 0 -> 12 triangles

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        //swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

    } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
            glfwWindowShouldClose(window) == 0);
    //Clean up VBO and shader
    glDeleteBuffers(1, &vertexBuffer);
    glDeleteBuffers(1, &uvBuffer);
    glDeleteProgram(programID);
    glDeleteTextures(1, &textureID);
    glDeleteVertexArrays(1, &vertexArrayID);

    // Close OpenGL window and terminate GLFW
    glfwTerminate();
}
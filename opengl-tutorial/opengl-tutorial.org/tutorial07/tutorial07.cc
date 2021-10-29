#include <cstdio>
#include <cstdlib>
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../common/shader.hpp"
#include "../common/texture.hpp"
#include "../common/controls.hpp"
#include "../common/objloader.hpp"

using namespace glm;

int main(int argc, char* argv[])
{
    GLFWwindow* window;
    GLuint programID, matrixID, vertexArrayID;
    GLuint texture, textureID;
    GLuint vertexBuffer, uvBuffer;

    //Initalise GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        getchar();
        return -1;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); //set OpenGL verion for 3.3 compatible
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); //Fuck make MacOS happy
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    //Crate a window
    window = glfwCreateWindow(1024, 768, "Tutorial 07 - Model Loading", NULL, NULL);
    if (window == NULL) {
        fprintf(stderr, "Failed to open & create GLFW window\n");
        getchar();
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    //Initialise GLEW
    glewExperimental = true; //Needed for core profile
    if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return -1;
    }

    //Dispose the keyboard event
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    //Set the mouse at the center of the window
    glfwPollEvents(); //receive the moust and keyboard event
    glfwSetCursorPos(window, 1024 / 2, 768 / 2);

    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
    glEnable(GL_DEPTH_TEST); //set depth test
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);
    // Cull triangles which normal is not towards the camera
    glEnable(GL_CULL_FACE);

    //Create an compile our GLSL program from the shaders
    programID = loadShader("./shader/transform.vert", "./shader/texture.frag");
    //Get a handle for our mvp uniform
    matrixID = glGetUniformLocation(programID, "mvp");
    //Load the Texture
    texture = loadDDS("uvmap.DDS");
    //Get a handle for out "textureSampler" uniform
    textureID = glGetUniformLocation(programID, "textureSampler");

    //Read out .obj file
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec3> normals; // Won't be used at the moment.
    bool res = loadOBJ("cube.obj", vertices, uvs, normals);

    //Generate the vertexs & buffers
    glGenVertexArrays(1, &vertexArrayID);
    glBindVertexArray(vertexArrayID);

    glGenBuffers(1, &vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3),  &vertices[0], GL_STATIC_DRAW);

    glGenBuffers(1, &uvBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
    glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(glm::vec2),  &uvs[0], GL_STATIC_DRAW);

    do {
        //Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        //Use our shader
        glUseProgram(programID);

        //Compute the mvp matrix from keyboard
        computeMatricesFromInputs(window);
        glm::mat4 projectionMatrix = getProjectionMatrix();
        glm::mat4 viewMatrix = getViewMatrix();
        glm::mat4 modelMatrix = glm::mat4(1.0);
        glm::mat4 mvp = projectionMatrix * viewMatrix * modelMatrix;

        //Send out transformation to the current bound shader in the "mvp" uniform
        glUniformMatrix4fv(matrixID, 1, GL_FALSE, &mvp[0][0]);

        //Bind our texture in Texture Unit 0
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        //Set our "textureSampler" sampler to use Texture Unit 0
        glUniform1i(textureID, 0);

        // 1rst attribute buffer :vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
        glVertexAttribPointer(
            0,                  // attribute
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
        );

        // 2nd attribute buffer : UVs
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
		glVertexAttribPointer(
			1,                                // attribute
			2,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);

        //Draw the tirangle
        glDrawArrays(GL_TRIANGLES, 0 , vertices.size());
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        //Swap buffer for smooth display
        glfwSwapBuffers(window);
        glfwPollEvents();

    } while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
                glfwWindowShouldClose(window) == 0);

    //Clean VBO and shader
    glDeleteBuffers(1, &vertexBuffer);
    glDeleteBuffers(1, &uvBuffer);
    glDeleteProgram(programID);
    glDeleteTextures(1, &texture);
    glDeleteVertexArrays(1, &vertexArrayID);

    //Close OpenGl window and terminate GLFW
    glfwTerminate();

    return 0;
}
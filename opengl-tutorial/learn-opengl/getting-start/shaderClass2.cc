#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cstdlib>

#include "common/shader_s.hpp"

static void processInput(GLFWwindow *window);
static void frambufferSizeCallback(GLFWwindow *window, int width, int height);

//settings
const unsigned int SRC_WIDTH = 800;
const unsigned int SRC_HEIGHT = 600;

// set up vertex data (and buffer(s)) and configure vertex attributes
float vertices[] = {
    // positions         // colors
    0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f,  // bottom right
    -0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, // bottom left
    0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 1.0f    // top
};

int main(int argc, char *argv[])
{
    GLFWwindow *window;
    GLuint VBO, VAO;
    //glfw initialized and configure
    if (!glfwInit()) {
        std::cout << "GLFW init error\n" << std::endl;
        exit(-1);
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    //glfw create window
    window = glfwCreateWindow(SRC_WIDTH, SRC_HEIGHT, "Learn OpenGL", NULL, NULL);
    if (!window) {
        std::cout << "GLFW create window error\n" <<std::endl;
        glfwTerminate();
        exit(-1);
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, frambufferSizeCallback);

    //glade: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        exit(-1);
    }

    //build and compile our shader program
    Shader ourShader("./shaders/3.5.shader.vs", "./shaders/3.3.shader.fs");

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    //bind VAO first, then bind and set VBO and set attributes

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    //position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    //color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    //VAO unbind is not unnecessary
    //glBinderVertexArray(0);

    //render loop
    while (!glfwWindowShouldClose(window)) {
        //input
        processInput(window);
        //render
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ourShader.use();
        //set a offset for X position
        ourShader.setFloat("xOffset", 0.5f);

        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0 , 3);

        //glfw swap buffer and poll input event
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    //optional: de-allocate all resource
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    //glfw terminate, free resource
    glfwTerminate();
    return 0;
}

//process keyboard input
static void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) ==  GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

//glfw whenever the window size change, all this callbakc function to adjust display
static void frambufferSizeCallback(GLFWwindow *window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays
    glViewport(0, 0, width, height);
}
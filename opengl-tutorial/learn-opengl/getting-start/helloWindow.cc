#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

void frameBufferSizeCallback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

//settings
const unsigned int SRC_WIDTH = 800;
const unsigned int SRC_HEIGHT = 600;

int main(int argc, char *argv[])
{
    GLFWwindow *window;

    if (!glfwInit()) {
        std::cout << "init glfw error" << std::endl;
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); //set up opengl version 3.3
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); //setup core profile
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    //create a window
    window = glfwCreateWindow(SRC_WIDTH, SRC_HEIGHT, "hello Window", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);//set the window as the OpenGL context
    glfwSetFramebufferSizeCallback(window, frameBufferSizeCallback);

    //glad load all OpenGL function pointer
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    //render loop
    while (!glfwWindowShouldClose(window)) {
        //Input
        processInput(window);
        //test if window set color OK
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        //glfw swap buffer and poll IO events(keys press/release mouse move etc)
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    //glfw terminate, cleaning all previous allocated GLFW resource
    glfwTerminate();
    return 0;
}

//process all input: qurey GLFW whether relevant keys are pressed/released this frame and recat accordingly
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

//glfw: whenever the window size changed(by OS or user resize), this callbakc will be executed
void frameBufferSizeCallback(GLFWwindow *window, int width, int height)
{
    //make suer the view port match the new window dimesions,note that width
    //and height will be significantly larger than specificed retina displays.
    glViewport(0, 0, width, height);
}
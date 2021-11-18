#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>

void frameBufferSizeCallback(GLFWwindow *window, int widht, int height);
void processInput(GLFWwindow *window);

//settings
const unsigned int SRC_WIDTH = 800;
const unsigned int SRC_HEIGHT = 600;

static const char *vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 Pos;\n"
    "layout (location = 1) in vec3 Color;\n"
    "out vec3 ourColor;\n"
    "void main() {\n"
    "   gl_Position = vec4(Pos, 1.0);\n"
    "   ourColor = Color;"
    "}\0";

static const char *fragmentShaderSource = "#version 330 core\n"
    "out vec4 fragColor;\n"
    "in vec3 ourColor;\n"
    "void main() {\n"
    "   fragColor = vec4(ourColor, 1.0f);\n"
    "}\n\0";

// set up vertex data (and buffer(s)) and configure vertex attributes
static float vertices[] = {
    // positions         // colors
    0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f,  // bottom right
    -0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, // bottom left
    0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 1.0f    // top
};

int main(int argc, char *argv[])
{
    GLFWwindow *window;
    unsigned int vertexShader, fragmentShader;
    unsigned int shaderProgram;
    unsigned int VBO, VAO;
    int success;
    char infoLog[512];

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

    //create and compile our shader program
    //vertex shader
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success); //get compilation status
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout <<"Error::Shader::Vertex::Compilation Error\n" << infoLog << std::endl;
    }
    //fragment shader
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success); //get compilation status
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout <<"Error::Shader::Fragment::Compilation Error\n" << infoLog << std::endl;
    }

    //create a Program and link shader
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success); //get link status
    if (!success) { //we also check linking error
        glGetProgramInfoLog(shaderProgram, 521, NULL, infoLog);
        std::cout << "Error::Sahder::Program::Link Error\n" << infoLog << std::endl;
    }

    //after link , delete shader for free resource
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    //bind the VAO first then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    //position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    //color attribute, set the buffer offset by 3 * sizeof(float)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
    // glBindVertexArray(0)

    // as we only have a single shader, we could also just activate our shader once beforehand if we want to
    glUseProgram(shaderProgram);

    while (!glfwWindowShouldClose(window)) {
        // process input
        processInput(window);
        //render
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        //draw the triangle
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        //glfw swap for smooth display
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    // optional: de-allocate all resources once they've outlived their purpose:
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    // glfw : terminate clearing all previous resource
    glfwTerminate();
    return 0;
}

void frameBufferSizeCallback(GLFWwindow *window, int widht, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, widht, height);
}

//query GLFW whether relevant keys are pressed/released this frame and react accordingly
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

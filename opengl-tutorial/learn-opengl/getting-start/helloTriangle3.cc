#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

void frameBufferSizeCallback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

//settings
const unsigned int SRC_WIDTH = 800;
const unsigned int SRC_HEIGHT = 600;

const char *vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 Pos;\n"
    "void main() {\n"
    "gl_Position = vec4(Pos.x, Pos.y, Pos.z, 1.0);\n"
    "}\n";

const char *fragmentShaderSource1 = "#version 330 core\n"
    "out vec4 fragColor;\n"
    "void main() {\n"
    "fragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
    "}\n";

const char *fragmentShaderSource2 = "#version 330 core\n"
    "out vec4 fragColor;\n"
    "void main() {\n"
    "fragColor = vec4(1.0f, 1.0f, 0.0f, 1.0f);\n"
    "}\n";

// set up vertex data (and buffer(s)) and configure vertex attributes
// ------------------------------------------------------------------
float firstTriangle[] = {
    -0.9f, -0.5f, 0.0f, // left
    -0.0f, -0.5f, 0.0f, // right
    -0.45f, 0.5f, 0.0f, // top
};

float secondTriangle[] = {
    0.0f, -0.5f, 0.0f, // left
    0.9f, -0.5f, 0.0f, // right
    0.45f, 0.5f, 0.0f  // top
};

int main(int argc, char *argv[])
{
    GLFWwindow *window;
    int status;
    unsigned int vertexShader;
    unsigned int fragmentShaderOrange, fragmentShaderYellow;
    unsigned int shaderProgramOrange, shaderProgramYellow;
    unsigned int VBOs[2], VAOs[2];
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

    //build and compile our shader program
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    fragmentShaderOrange = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShaderOrange, 1, &fragmentShaderSource1, NULL);
    glCompileShader(fragmentShaderOrange);

    fragmentShaderYellow = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShaderYellow, 1, &fragmentShaderSource2, NULL);
    glCompileShader(fragmentShaderYellow);

    //link shaders
    shaderProgramOrange = glCreateProgram();
    glAttachShader(shaderProgramOrange, vertexShader);
    glAttachShader(shaderProgramOrange, fragmentShaderOrange);
    glLinkProgram(shaderProgramOrange);

    shaderProgramYellow = glCreateProgram();
    glAttachShader(shaderProgramYellow, vertexShader);
    glAttachShader(shaderProgramYellow, fragmentShaderYellow);
    glLinkProgram(shaderProgramYellow);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShaderOrange);
    glDeleteShader(fragmentShaderYellow);

    glGenVertexArrays(2, VAOs);
    glGenBuffers(2, VBOs); // we can also generate multiple VAOs or buffers at the same time

    //first triangle setup
    glBindVertexArray(VAOs[0]);
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(firstTriangle), firstTriangle, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3 ,GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    // no need to unbind at all as we directly bind a different VAO the next few lines
    //glBindVertexArray(0);

    //secong triangle setup
    glBindVertexArray(VAOs[1]);
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[1]); //bind to different VAO and VBO
    glBufferData(GL_ARRAY_BUFFER, sizeof(secondTriangle), secondTriangle, GL_STATIC_DRAW);
    // because the vertex data is tightly packed we can also specify 0 as the vertex attribute's stride to let OpenGL figure it out
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glEnableVertexAttribArray(0);
    // no need to unbind at all as we directly bind a different VAO the next few lines
    //glBindVertexArray(0);

    // uncomment this call to draw in wireframe polygons.
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    //render loop
    while (!glfwWindowShouldClose(window)) {
        //Input
        processInput(window);
        //test if window set color OK
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // now when we draw the triangle we first use the vertex and orange fragment
        // shader from the first program
        glUseProgram(shaderProgramOrange);
        // draw first triangle using the data from VAO[0]
        glBindVertexArray(VAOs[0]);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // when we draw the second triangle we want to use a different shader program
        // so we switch to the shader program with our yellow fragment shader.
        glUseProgram(shaderProgramYellow);
        // draw second triangle using the data from VAO[1]
        glBindVertexArray(VAOs[1]);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // glfw swap buffer and poll IO events(keys press/release mouse move etc)
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    //Optional de-allocate all reources once they've outlived their purpose
    glDeleteVertexArrays(2, VAOs);
    glDeleteBuffers(2, VBOs);
    glDeleteProgram(shaderProgramOrange);
    glDeleteProgram(shaderProgramYellow);

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
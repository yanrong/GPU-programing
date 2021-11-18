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

const char *fragmentShaderSource = "#version 330 core\n"
    "out vec4 fragColor;\n"
    "void main() {\n"
    "fragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
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
    unsigned int vertexShader, fragmentShader, shaderProgram;
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
    //check shader compile status
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &status);
    if (!status) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "Error::Shader::Vertex::Compilation Failed\n" << infoLog << std::endl;
    }

    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    //check shader compile status
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &status);
    if (!status) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "Error::Shader::Fragment::Compilation Failed\n" << infoLog << std::endl;
    }

    //link shaders
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    //check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &status);
    if (!status) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "Error::Shader::Program::Link Failed\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    glGenVertexArrays(2, VAOs);
    glGenBuffers(2, VBOs);
    //bin the Vertex Array Object first, then bind and set vertex buffers(s)
    //and the configure vertex attributes
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
    // because the vertex data is tightly packed we can also specify 0 as the vertex
    // attribute's stride to let OpenGL figure it out
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glEnableVertexAttribArray(0);
    // not really necessary as well, but beware of calls that could affect VAOs while
    // this one is bound (like binding element buffer objects, or enabling/disabling vertex attributes)
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

        //draw out first trangle
        glUseProgram(shaderProgram);
        //draw first triangle using the data from VAO[0]
        glBindVertexArray(VAOs[0]);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        //draw second triangle using the data from VAO[1]
        glBindVertexArray(VAOs[1]);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        //glfw swap buffer and poll IO events(keys press/release mouse move etc)
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    //Optional de-allocate all reources once they've outlived their purpose
    glDeleteVertexArrays(2, VAOs);
    glDeleteBuffers(2, VBOs);
    glDeleteProgram(shaderProgram);

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
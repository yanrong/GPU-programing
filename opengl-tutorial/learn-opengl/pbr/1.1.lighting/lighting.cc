#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "common/fileSystem.hpp"
#include "common/shader_s.hpp"
#include "common/camera.hpp"
#include "common/model.hpp"

static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
static void mouseCallback(GLFWwindow* window, double xPos, double yPos);
static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
static void processInput(GLFWwindow* window);
static GLuint loadTexture(const char* path);
static void renderSphere();

//set screen width and height
const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;

//camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = 800.0f / 2.0;
float lastY = 600.0f / 2.0;
bool firstMouse = true;

//timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

GLuint sphereVAO = 0, indexCount;

int main(int argc, char *argv[])
{
    GLFWwindow *window;
    int nrColumn = 7, nrRow = 7;
    float spacing = 2.5;

    //glfw initialized and configure
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    //glfw window create
    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Learn OpenGL", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window!" << std::endl;
        glfwTerminate();
        return -1;
    }
    //set upt current widow as OpenGL render contex
    glfwMakeContextCurrent(window);

    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetScrollCallback(window, scrollCallback);

    //tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    //glad: load all OpenGL function pointer
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    //configure globla opengl state
    glEnable(GL_DEPTH_TEST);

    //build and compile shader
    Shader shader("shaders/pbr.vs", "shaders/pbr.fs");
    shader.use();
    shader.setVec3("albedo", 0.5f, 0.0f, 0.0f);
    shader.setFloat("ao", 1.0f);

    //lights attribute
    glm::vec3 lightPosition[] = {
        glm::vec3(-10.0f,  10.0f, 10.0f),
        glm::vec3( 10.0f,  10.0f, 10.0f),
        glm::vec3(-10.0f, -10.0f, 10.0f),
        glm::vec3( 10.0f, -10.0f, 10.0f),
    };

    glm::vec3 lightColor[] = {
        glm::vec3(300.0f, 300.0f, 300.0f),
        glm::vec3(300.0f, 300.0f, 300.0f),
        glm::vec3(300.0f, 300.0f, 300.0f),
        glm::vec3(300.0f, 300.0f, 300.0f)
    };

    //initialize static shader uniforms
    glm::mat4 projection = glm::perspective(glm::radians(camera.zoom), float(SCR_WIDTH) / float(SCR_HEIGHT), 0.1f, 100.0f);
    //shader.use(); // necessary ???, glUseProgram has been called above
    shader.setMat4("projection", projection);

    //render loop
    while (!glfwWindowShouldClose(window)) {
        //per frame time logic
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        //input
        processInput(window);
        //render
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //shader.use();
        glm::mat4 view = camera.getViewMatrix();
        shader.setMat4("view", view);
        shader.setVec3("camPosition", camera.position);

        // render rows * column number of sphere with varying metallic/roughness
        // values scaled by rows and columns respectively
        glm::mat4 model = glm::mat4(1.0f);

        for (int row = 0; row < nrRow; row++) {
            shader.setFloat("metallic", float(row) / float(nrRow));
            for (int col = 0; col < nrColumn; col++) {
                // we clamp the roughness to 0.05 - 1.0 as perfectly smooth surfaces (roughness of 0.0)
                //tend too look a bit off on direct lighting
                shader.setFloat("roughness", glm::clamp(float(col) / float(nrColumn), 0.05f, 1.0f));

                model = glm::mat4(1.0f);
                model = glm::translate(model, glm::vec3((col - (nrColumn / 2)) * spacing, (row - (nrRow / 2)) * spacing, 0.0f));
                shader.setMat4("model", model);
                renderSphere();
            }
        }

        //render light source (simple re-render sphere at light position)
        //this lool a bit off as we use the same shader, but it'll make
        // their positions obvious and keep the codeprint small.
        for (int i = 0; i < sizeof(lightPosition) / sizeof(lightPosition[0]); i++) {
            //glm::vec3 newPosition = lightPosition[i] + glm::vec3(sin(glfwGetTime() * 5.0) * 5.0, 0.0, 0.0);
            glm::vec3 newPosition = lightPosition[i];
            shader.setVec3("lightPosition[" + std::to_string(i) + "]", newPosition);
            shader.setVec3("lightColor[" + std::to_string(i) + "]", lightColor[i]);

            model = glm::mat4(1.0f);
            model = glm::translate(model, newPosition);
            model = glm::scale(model, glm::vec3(0.5f));
            shader.setMat4("model", model);
            renderSphere();
        }

        //glfw swapbuffer and IO poll Event
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.processKeyboardOpt(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.processKeyboardOpt(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.processKeyboardOpt(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.processKeyboardOpt(RIGHT, deltaTime);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
void mouseCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.processMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.processMouseScroll(yoffset);
}

void renderSphere()
{
    GLuint VBO, EBO;
    std::vector<glm::vec3> position;
    std::vector<glm::vec2> uv;
    std::vector<glm::vec3> normal;
    std::vector<unsigned int> indices;
    std::vector<float> data;
    const unsigned int X_SEGMENTS = 64;
    const unsigned int Y_SEGMENTS = 64;
    const float PI = 3.14159265359;
    bool oddRow = false;

    //generate a sphere coordinate, each time in loop, do much calculate at here may not a good design
    for (int x = 0; x <= X_SEGMENTS; x++) {
        for (int y = 0; y <= Y_SEGMENTS; y++) {
            float xSegment = float(x) / float(X_SEGMENTS);
            float ySegment = float(y) / float (Y_SEGMENTS);
            float xPosition = std::cos(xSegment * 2.0f * PI) * std::sin(ySegment * PI);
            float yPosition = std::cos(ySegment * PI);
            float zPosition = std::sin(xSegment * 2.0f * PI) * std::sin(ySegment * PI);

            position.push_back(glm::vec3(xPosition, yPosition, zPosition));
            uv.push_back(glm::vec2(xSegment, ySegment));
            normal.push_back(glm::vec3(xPosition, yPosition, zPosition));
        }
    }

    for (int y = 0; y < Y_SEGMENTS; y++) {
        if (!oddRow){ // even row
            for (int x = 0; x <= X_SEGMENTS; x++) {
                indices.push_back(y * (X_SEGMENTS + 1) + x);
                indices.push_back((y + 1) * (X_SEGMENTS + 1) + x);
            }
        } else { //odd row
            for (int x = X_SEGMENTS; x >= 0 ; x--) {
                indices.push_back((y + 1) * (X_SEGMENTS + 1) + x);
                indices.push_back(y * (X_SEGMENTS + 1) + x);
            }
            oddRow = !oddRow;
        }
    }

    indexCount = indices.size();
    for (int i = 0; i < position.size(); i++) {
        data.push_back(position[i].x);
        data.push_back(position[i].y);
        data.push_back(position[i].z);

        if(normal.size() > 0) {
            data.push_back(normal[i].x);
            data.push_back(normal[i].y);
            data.push_back(normal[i].z);
        }

        if(uv.size() > 0) {
            data.push_back(uv[i].x);
            data.push_back(uv[i].y);
        }
    }

    if (sphereVAO == 0) {
        glGenVertexArrays(1, &sphereVAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
        glBindVertexArray(sphereVAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), &data[0], GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

        unsigned int stride = (3 + 2 + 3) * sizeof(float);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void *)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void *)(3 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void *)(6 * sizeof(float)));

        glBindVertexArray(0);
    }

    glBindVertexArray(sphereVAO);
    glDrawElements(GL_TRIANGLE_STRIP, indexCount, GL_UNSIGNED_INT, 0);
}

//utility function for loading a 2D texture
static GLuint loadTexture(const char* path)
{
    GLuint textureID;
    int width, height, nrComponents;
    GLenum format;

    glGenTextures(1, &textureID);
    unsigned char *data = stbi_load(path, &width, &height, &nrComponents, 0);

    if (data != NULL) {
        if (nrComponents == 1) {
            format = GL_RED;
        } else if (nrComponents == 3) {
            format = GL_RGB;
        } else if (nrComponents == 4) {
            format = GL_RGBA;
        }

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    } else {
        std::cerr << "load texture Error!" << path << std::endl;
    }

    stbi_image_free(data);

    return textureID;
}

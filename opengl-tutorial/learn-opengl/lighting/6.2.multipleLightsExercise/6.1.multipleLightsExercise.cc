#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
//Open STB ON
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include "common/camera.hpp"
#include "common/shader_s.hpp"
#include "common/fileSystem.hpp"

#define EXERCISE_NUM 1

static void framebufferSizeCallback(GLFWwindow* window, int height, int width);
static void mouseCallback(GLFWwindow* window, double xPos, double yPos);
static void scrollCallback(GLFWwindow* window, double xOffset, double yOffset);
static void processInput(GLFWwindow *window);
static GLuint loadTexture(const char *path);

//settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

//camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

//timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// set up vertex data (and buffer(s)) an d configure vertex attributes
float vertices[] = {
    // positions          // normals           // texture coords
    -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f,
     0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  0.0f,
     0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  1.0f,
     0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  1.0f,
    -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  1.0f,
    -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f,
    -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,
     0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  0.0f,
     0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  1.0f,
     0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  1.0f,
    -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  1.0f,
    -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,
    -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  0.0f,
    -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  1.0f,
    -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
    -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
    -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  0.0f,
    -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  0.0f,
     0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,
     0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  1.0f,
     0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
     0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
     0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  0.0f,
     0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  1.0f,
     0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  1.0f,
     0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  0.0f,
     0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  0.0f,
    -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  0.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  1.0f,
    -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,
     0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  1.0f,
     0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  0.0f,
     0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  0.0f,
    -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  0.0f,
    -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f
};

// lighting
glm::vec3 lightPosition(1.2f, 1.0f, 2.0f);

int main(int argc, char *argv[])
{
    GLFWwindow* window;
    GLuint VBO, lightVAO, objectVAO;
    GLuint diffuseMap, specularMap, emissionMap;

    // positions all containers
    glm::vec3 cubePositions[] = {
        glm::vec3( 0.0f,  0.0f,  0.0f),
        glm::vec3( 2.0f,  5.0f, -15.0f),
        glm::vec3(-1.5f, -2.2f, -2.5f),
        glm::vec3(-3.8f, -2.0f, -12.3f),
        glm::vec3( 2.4f, -0.4f, -3.5f),
        glm::vec3(-1.7f,  3.0f, -7.5f),
        glm::vec3( 1.3f, -2.0f, -2.5f),
        glm::vec3( 1.5f,  2.0f, -2.5f),
        glm::vec3( 1.5f,  0.2f, -1.5f),
        glm::vec3(-1.3f,  1.0f, -1.5f)
    };

    // positions of the point lights
    glm::vec3 pointLightPositions[] = {
        glm::vec3( 0.7f,  0.2f,  2.0f),
        glm::vec3( 2.3f, -3.3f, -4.0f),
        glm::vec3(-4.0f,  2.0f, -12.0f),
        glm::vec3( 0.0f,  0.0f, -3.0f)
    };

    glm::vec3 pointLightColors[] = {
        glm::vec3(1.0f, 0.6f, 0.0f),
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(1.0f, 1.0, 0.0),
        glm::vec3(0.2f, 0.2f, 1.0f)
    };

    //glfw initialize
    if (glfwInit() != GLFW_TRUE) {
        std::cerr << "GLFW initialize error" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Learn OpenGL", NULL, NULL);
    if (!window) {
        std::cerr <<"GLFW create widow failed" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetScrollCallback(window, scrollCallback);

    //capture the mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    //glade, load all GL function pointer
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    //enable depth test
    glEnable(GL_DEPTH_TEST);

    Shader objectShader("shaders/6.1.multipleLights.vs", "shaders/6.1.multipleLights.fs");
    Shader lightCubeShader("shaders/6.1.lightCube.vs", "shaders/6.1.lightCube.fs");

    //first set up the object VAO
    glGenVertexArrays(1, &objectVAO);
    glGenBuffers(1, &VBO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindVertexArray(objectVAO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void *)(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void *)(3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void *)(6 * sizeof(GLfloat)));
    glEnableVertexAttribArray(2);

    //second set up the lighting cube VAO
    glGenVertexArrays(1, &lightVAO);
    glBindVertexArray(lightVAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    //skip fill data to buffer
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void *)(0));
    glEnableVertexAttribArray(0);

    //load texture
    diffuseMap = loadTexture(fileSystem::getResource("/../resources/textures/container2.png").c_str());
    specularMap = loadTexture(fileSystem::getResource("/../resources/textures/container2_specular.png").c_str());

    objectShader.use();
    objectShader.setInt("material.diffTexture", 0);
    objectShader.setInt("material.specTexture", 1);

    //render loop
    while (!glfwWindowShouldClose(window)) {
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        //clear background
        glClearColor(0.75f, 0.52f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //be sure to activate shader
        objectShader.use();
        objectShader.setVec3("viewPosition", camera.position);
        objectShader.setFloat("material.shininess", 32.0f);
        /*
        * Here we set all the uniforms for the 5/6 types of lights we have. We have to set them manually and index
        * the proper PointLight struct in the array to set each uniform variable. This can be done more code-friendly
        * by defining light types as classes and set their values in there, or by using a more efficient uniform approach
        * by using 'Uniform buffer objects', but that is something we'll discuss in the 'Advanced GLSL' tutorial.
        */

        //direction light
        objectShader.setVec3("myDirLight.direction", -0.2f, -1.0f, -0.3f);
        objectShader.setVec3("myDirLight.ambient", 0.05f, 0.05f, 0.05f);
        objectShader.setVec3("myDirLight.diffuse", 0.4f, 0.4f, 0.4f);
        objectShader.setVec3("myDirLight.specular", 0.5f, 0.5f, 0.5f);
        //point light
        //light 1
        objectShader.setVec3("myPointLights[0].position", pointLightPositions[0]);
        objectShader.setVec3("myPointLights[0].ambient", pointLightColors[0] * 0.1f);
        objectShader.setVec3("myPointLights[0].diffuse", pointLightColors[0]);
        objectShader.setVec3("myPointLights[0].specular", pointLightColors[0]);
        objectShader.setFloat("myPointLights[0].constant", 1.0f);
        objectShader.setFloat("myPointLights[0].linear", 0.09);
        objectShader.setFloat("myPointLights[0].quadratic", 0.032);
        //light 2
        objectShader.setVec3("myPointLights[1].position", pointLightPositions[1]);
        objectShader.setVec3("myPointLights[1].ambient", pointLightColors[1] * 0.1f);
        objectShader.setVec3("myPointLights[1].diffuse", pointLightColors[1]);
        objectShader.setVec3("myPointLights[1].specular", pointLightColors[1]);
        objectShader.setFloat("myPointLights[1].constant", 1.0f);
        objectShader.setFloat("myPointLights[1].linear", 0.09);
        objectShader.setFloat("myPointLights[1].quadratic", 0.032);
        //light 3
        objectShader.setVec3("myPointLights[2].position", pointLightPositions[2]);
        objectShader.setVec3("myPointLights[2].ambient", pointLightColors[2] * 0.1f);
        objectShader.setVec3("myPointLights[2].diffuse", pointLightColors[2]);
        objectShader.setVec3("myPointLights[2].specular", pointLightColors[2]);
        objectShader.setFloat("myPointLights[2].constant", 1.0f);
        objectShader.setFloat("myPointLights[2].linear", 0.09);
        objectShader.setFloat("myPointLights[2].quadratic", 0.032);
        //light 3
        objectShader.setVec3("myPointLights[3].position", pointLightPositions[3]);
        objectShader.setVec3("myPointLights[3].ambient", pointLightColors[3] * 0.1f);
        objectShader.setVec3("myPointLights[3].diffuse", pointLightColors[3]);
        objectShader.setVec3("myPointLights[3].specular", pointLightColors[3]);
        objectShader.setFloat("myPointLights[3].constant", 1.0f);
        objectShader.setFloat("myPointLights[3].linear", 0.09);
        objectShader.setFloat("myPointLights[3].quadratic", 0.032);

        //spot light
        objectShader.setVec3("mySpotLight.position", camera.position);
        objectShader.setVec3("mySpotLight.direction", camera.front);
        objectShader.setVec3("mySpotLight.ambient", 0.0f, 0.0f, 0.0f);
        objectShader.setVec3("mySpotLight.diffuse", 1.0f, 1.0f, 1.0f);
        objectShader.setVec3("mySpotLight.specular", 1.0f, 1.0f, 1.0f);
        objectShader.setFloat("mySpotLight.constant", 1.0f);
        objectShader.setFloat("mySpotLight.linear", 0.09);
        objectShader.setFloat("mySpotLight.quadratic", 0.032);
        objectShader.setFloat("mySpotLight.cutOff", glm::cos(glm::radians(12.5f)));
        objectShader.setFloat("mySpotLight.outerCutOff",  glm::cos(glm::radians(15.0f)));

        //view/projection transformations
        glm::mat4 projection = glm::perspective(glm::radians(camera.zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.getViewMatrix();
        objectShader.setMat4("projection", projection);
        objectShader.setMat4("view", view);
        //world transformation, identity
        glm::mat4 model = glm::mat4(1.0f);
        objectShader.setMat4("model", model);

        //bind diffuse map
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, diffuseMap);
        //bind specular map
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, specularMap);

        //render(Draw) the cube objection
        glBindVertexArray(objectVAO);
        for (int i = 0; i < 10; i++) {
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, cubePositions[i]);
            float angle = 20.0f * i;
            model = glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
            objectShader.setMat4("model", model);

            glDrawArrays(GL_TRIANGLES, 0, 36);
        }

        //draw the lamp object
        lightCubeShader.use();
        lightCubeShader.setMat4("projection", projection);
        lightCubeShader.setMat4("view", view);
        glBindVertexArray(lightVAO);
        for (int i = 0; i < 4; i++)
        {
            model = glm::mat4(1.0f);
            model = glm::translate(model, pointLightPositions[i]);
            model = glm::scale(model, glm::vec3(0.2f)); //a smaller cube
            lightCubeShader.setMat4("model", model);

            glDrawArrays(GL_TRIANGLES, 0, 36);
        }
        //glfw swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &objectVAO);
    glDeleteVertexArrays(1, &lightVAO);
    glDeleteBuffers(1, &VBO);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

void framebufferSizeCallback(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void mouseCallback(GLFWwindow *window, double xPos, double yPos)
{
    if (firstMouse) {
        lastX = xPos;
        lastY = yPos;
        firstMouse = false;
    }

    float xOffset = xPos - lastX;
    float yOffset = lastY - yPos; // reversed since y-coordinates go from bottom to top

    lastX = xPos;
    lastY = yPos;

    camera.processMouseMovement(xOffset, yOffset);
}

void scrollCallback(GLFWwindow *widnow, double xOffset, double yOffset)
{
    camera.processMouseScroll(yOffset);
}

void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }

    if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        camera.processKeyboardOpt(FORWARD, deltaTime);
    }
    if(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS){
        camera.processKeyboardOpt(BACKWARD, deltaTime);
    }
    if(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS){
        camera.processKeyboardOpt(LEFT, deltaTime);
    }
    if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS){
        camera.processKeyboardOpt(RIGHT, deltaTime);
    }
}

static GLuint loadTexture(const char *path)
{
    GLuint textureID;
    int width, height, nrChannels;
    unsigned char *data = NULL;

    glGenTextures(1, &textureID);

    //load image, crate texture and generate mipmaps
    data = stbi_load(path, &width, &height, &nrChannels, 0);

    if (data != NULL) {
        GLenum format;
        if (nrChannels == 1) {
            format = GL_RED;
        } else if (nrChannels == 3) {
            format = GL_RGB;
        } else if (nrChannels == 4) {
            format = GL_RGBA;
        }
        glBindTexture(GL_TEXTURE_2D, textureID);
        //set the texture wrapping parameters
        // note that we set the container wrapping method to GL_CLAMP_TO_EDGE
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        //set texture filtering paramters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    } else {
        std::cout << "Failed to load texture :" << path<<std::endl;
    }
    stbi_image_free(data);

    return textureID;
}
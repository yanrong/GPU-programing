#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cstdlib>
#include <cmath>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "common/fileSystem.hpp"
#include "common/shader_s.hpp"
#include "common/camera.hpp"
#include "common/model.hpp"

static void framebufferSizeCallback(GLFWwindow* window, int height, int width);
static void mouseCallback(GLFWwindow* window, double xPos, double yPos);
static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
static void processInput(GLFWwindow* window);

//settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

//camera
Camera camera(glm::vec3(0.0f, 0.0f, 155.0f));
float lastX = (float)SCR_WIDTH / 2.0;
float lastY = (float)SCR_HEIGHT / 2.0;
bool firstMouse = true;

//timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

int main(int argc, char *argv[])
{
    GLFWwindow* window;
    GLuint instanceVBO;

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

    //glad, load all GL function pointer
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    //enable depth test
    glEnable(GL_DEPTH_TEST);

    //buidl and compile shaders
    Shader asteroidShader("shaders/asteroids.vs", "shaders/asteroids.fs");
    Shader planetShader("shaders/planet.vs", "shaders/planet.fs");

    //load models
    model rock(fileSystem::getResource("../resources/objects/rock/rock.obj"));
    model planet(fileSystem::getResource("../resources/objects/planet/planet.obj"));

    //generate a larage list of semi-random model transformation matrices
    int amount = 100000;
    glm::mat4* modelMatrices = new glm::mat4[amount];
    srand(glfwGetTime());
    float radius = 150.0;
    float offset = 25.0f;
    for (int i = 0; i < amount; i++) {
        glm::mat4 model = glm::mat4(1.0f);
        //1. translation: displace along circle with 'radius' in range [-offset, offset]
        float angle = (float)i / (float) amount * 360.0f;
        float displacement = (rand() % (int)(2 * offset * 100)) / 100.0f - offset;
        float x = sin(angle) * radius + displacement;
        displacement = (rand() % (int)(2 * offset * 100)) / 100.0f - offset;
        float y = displacement * 0.4f; //ket height of asteroid field small compare to to width of x and z
        displacement = (rand() % (int)(2 * offset * 100)) / 100.0f - offset;
        float z = cos(angle) * radius + displacement;
        model = glm::translate(model, glm::vec3(x, y, z));

        //2. scale: scale between 0.05 and 0.25f
        float scale = (rand() % 20) / 100.0f + 0.05;
        model = glm::scale(model, glm::vec3(scale));

        //3. rotation: and random rotation aroud a (semi) randomly picked rotation axis vecrot
        float rotAngle = (rand() % 360);
        model = glm::rotate(model, rotAngle, glm::vec3(0.4f, 0.6f, 0.8f));

        //4. now add to list of matrices
        modelMatrices[i] = model;
    }

    //configure instance array

    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, amount * sizeof(glm::mat4), modelMatrices, GL_STATIC_DRAW);

    // set transformation matrices as an instance vertex attribute (with divisor 1)
    // note: we're cheating a little by taking the, now publicly declared, VAO of the model's mesh(es) and adding new
    // vertexAttribPointers normally you'd want to do this in a more organized fashion, but for learning purposes this will do.
    for (int i = 0; i < rock.meshes.size(); i++) {
        GLuint VAO = rock.meshes[i].VAO;
        glBindVertexArray(VAO);
        //set attribute pointer for matrix (4 times vec4)
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)0);
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)(sizeof(glm::vec4)));
        glEnableVertexAttribArray(5);
        glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)(2 * sizeof(glm::vec4)));
        glEnableVertexAttribArray(6);
        glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)(3 * sizeof(glm::vec4)));

        glVertexAttribDivisor(3, 1);
        glVertexAttribDivisor(4, 1);
        glVertexAttribDivisor(5, 1);
        glVertexAttribDivisor(6, 1);

        glBindVertexArray(0);
    }

    while (!glfwWindowShouldClose(window)) {
        //pre-frame time logic
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        //input
        processInput(window);

        //render
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //configure transformation matrices
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 1.0f, 100.0f);
        glm::mat4 view = camera.getViewMatrix();

        asteroidShader.use();
        asteroidShader.setMat4("projection", projection);
        asteroidShader.setMat4("view", view);
        planetShader.use();
        planetShader.setMat4("projection", projection);
        planetShader.setMat4("view", view);

        //draw planet
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0.0f, -3.0f, 0.0f));
        model = glm::scale(model , glm::vec3(4.0f, 4.0f, 4.0f));
        planetShader.setMat4("model", model);
        planet.draw(planetShader);

        //draw meteorites
        asteroidShader.use();
        asteroidShader.setInt("textureDiffuse1", 0);
        glActiveTexture(GL_TEXTURE0);
        //// note: we also made the textures_loaded vector public (instead of private) from the model class.
        glBindTexture(GL_TEXTURE_2D, rock.texturesLoaded[0].id);
        for (int i = 0; i < rock.meshes.size(); i++) {
            glBindVertexArray(rock.meshes[i].VAO);
            glDrawElementsInstanced(GL_TRIANGLES, rock.meshes[i].indices.size(), GL_UNSIGNED_INT, 0, amount);
            glBindVertexArray(0);
        }

        //swap buffer and listen IO event
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    delete []modelMatrices;
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

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.processMouseScroll(yoffset);
}
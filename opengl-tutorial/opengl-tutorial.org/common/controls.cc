#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "controls.hpp"

using namespace glm;

static glm::mat4 viewMatrix;
static glm::mat4 projectionMatrix;

// Initial position : on + Z
static glm::vec3 position = glm::vec3(0 ,0, 5);
// Initial horizontal angle :toward - Z
static float horizontalAngle = 3.14f;
// Initial Vertical angle : none
static float verticalAngle = 0.0f;
// Initial Field of View
static float initialFov = 45.0f;

static float speed = 3.0f; // 3 units / second
static float mouseSpeed = 0.005f;

void computeMatricesFromInputs(GLFWwindow *window)
{
    double xpos, ypos;
    int width, height;
    float FoV;
    // glfwGetTime is call only once, the fist time this function is called
    static double lastTime = glfwGetTime();
    // Compute time difference between current and last
    double currentTime = glfwGetTime();
    float deltaTime = (float)(currentTime - lastTime);

    // Get Mouse position
    glfwGetCursorPos(window, &xpos, &ypos);
    glfwGetWindowSize(window, &width, &height);
    // Reset moust position for the next frame
    glfwSetCursorPos(window, width / 2, height / 2);
    // Compute new orientation
    horizontalAngle += mouseSpeed * float(width / 2 - xpos);
    verticalAngle   += mouseSpeed * float(height / 2 - ypos);

    // Direction : Spherical coordinates to Cartesian coordinate conversion
    glm::vec3 direction(
        cos(verticalAngle) * sin(horizontalAngle),
        sin(verticalAngle),
        cos(verticalAngle) * cos(horizontalAngle)
    );
    //Right vector
    glm::vec3 right = glm::vec3(
        sin(horizontalAngle - 3.14f / 2.0f),
        0,
        cos(horizontalAngle - 3.14f / 2.0f)
    );
    // Up vector
    glm::vec3 up = glm::cross(right, direction);

    //Move forward
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        position += direction * deltaTime * speed;
    }
    //Move backward
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        position -= direction * deltaTime * speed;
    }
    //Strafe right
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        position += right * deltaTime * speed;
    }
    //Strafe left
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        position -= right * deltaTime * speed;
    }
    /* - 5 * glfwGetMouseWheel(); // Now GLFW 3 requires setting up a callback for this.
    * It's a bit too complicated for this beginner's tutorial, so it's disabled instead.
    */
    FoV = initialFov;
    //Projection matrix :45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
    projectionMatrix = glm::perspective(glm::radians(FoV), 4.0f / 3.0f, 0.1f, 100.0f);
    //Camera matrix
    viewMatrix = glm::lookAt(
        position,
        position + direction,
        up
    );
    // Update the last time to NOW
    lastTime = currentTime;
}

glm::mat4 getViewMatrix() {
    return viewMatrix;
}

glm::mat4 getProjectionMatrix() {
    return projectionMatrix;
}
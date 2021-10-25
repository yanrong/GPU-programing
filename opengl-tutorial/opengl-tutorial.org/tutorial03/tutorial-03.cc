#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "../common/shader.hpp"

using namespace glm;

static const GLfloat g_vertex_buffer_data[] = {
	-1.0f, -1.0f, 0.0f,
	1.0f, -1.0f, 0.0f,
	0.0f, 1.0f, 0.0f,
};

int main(int argc, char* argv[])
{
	GLFWwindow *window;
	GLuint vertexBuffer;
	GLuint vertextArrayID;

	if(!glfwInit()){
		fprintf(stderr, "Failed to initialize GLFW\n");
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); //opengl version 3.3
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);// To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL

	window = glfwCreateWindow(800, 600, "Tutorial 03", NULL, NULL);
	if(window == NULL){
		fprintf(stderr, "Failed to openg GLFW window, If you have an Intel GPU,they are not 3.3 compatible.Try the 2.1 version of the tutorials.\n");
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	//Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if(glewInit()  != GLEW_OK){
		fprintf(stderr, "Failed to intialize GLEW\n");
		glfwTerminate();
		return -1;
	}
	//Ensure we can capture the esacap key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    //Dark blue background
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

	//Generate Vertex Array Objection(VAO)
	glGenVertexArrays(1, &vertextArrayID);
	glBindVertexArray(vertextArrayID);

	//Create and Compile our GLSL program from the shaders
	GLuint programID = loadShader("./shader/transform.vert", "./shader/color.frag");

    //Get a handle for our "MVP" uniform
    GLuint matrixID = glGetUniformLocation(programID, "mvp");
    //Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 4.0f / 3.0f, 0.1f, 100.0f);
    //Or, for an ortho camera :
    // glm::mat4 projection = glm::ortho(-10.0f,10.0f,-10.0f,10.0f,0.0f,100.0f); // In world coordinates

    //Camera matrix
    glm::mat4 view = glm::lookAt(
        glm::vec3(4, 3, 3), // Camera is at (4,3,3), in World Space
        glm::vec3(0, 0, 0), // and looks at the origin
        glm::vec3(0, 1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
    );
    //Model matrix : an identity matrix(model will be at origin)
    glm::mat4 model = glm::mat4(1.0f);
    //Our ModelViewProjection : multiplication is our 3 matrices
    glm::mat4 mvp = projection * view * model; //Remeber the sequence
    // 1rst attribute buffer : vertices
    glGenBuffers(1, &vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

    do{
		//clear the screen
		glClear(GL_COLOR_BUFFER_BIT);

		glUseProgram(programID);

        // Send our transformation to the currently bound shader,
		// in the "MVP" uniform
		glUniformMatrix4fv(matrixID, 1, GL_FALSE, &mvp[0][0]);

		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
		glVertexAttribPointer(0, //attribute 0. No particular reason for 0, but must match the layout in the shader
		    3,
            GL_FLOAT,
            GL_FALSE,
            0,
            (void *) 0
        );

        //Draw the triangle !
		glDrawArrays(GL_TRIANGLES, 0, 3); // Starting from vertex 0; 3 vertices total
		glDisableVertexAttribArray(0);

		//swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();
	} while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS
		&& glfwWindowShouldClose(window) == 0);

	//clean VBO
	glDeleteBuffers(1, &vertexBuffer);
	glDeleteVertexArrays(1, &vertextArrayID);
	glDeleteProgram(programID);
	// Close OpenGL window and terminate GLFW
	glfwTerminate();
	return 0;
}
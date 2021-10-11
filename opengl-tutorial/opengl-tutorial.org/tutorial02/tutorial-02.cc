#define GLEW_STATIC
#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "../common/loadShader.hpp"
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
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL 

	window = glfwCreateWindow(800, 600, "Tutorial 01", NULL, NULL);
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
		return 0;
	}
	//Ensure we can capture the esacap key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

	//Generate Vertext Buffer Objection(VBO)
	glGenBuffers(1, &vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);
	
	//Generate Vertex Array Objection(VAO)
	glGenVertexArrays(1, &vertextArrayID);
	glBindVertexArray(vertextArrayID);

	//Create and Compile our GLSL program from the shaders
	GLuint programID = loadShader("./shader/vertextShader.vert", "./shader/fragmentShader.frag");

	do{
		//clear the screen
		glClear(GL_COLOR_BUFFER_BIT);

		glUseProgram(programID);

		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
		//attribute size, type, normalized, stride, array buffer offset
		/*arguments index, size, type, stride, array buffer offset*/
		glVertexAttribPointer(0, //attribute 0. No particular reason for 0, but must match the layout in the shader
		3, GL_FLOAT, GL_FALSE, 0, (void *) 0);
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
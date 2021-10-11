#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

using namespace glm;

int main(int argc, char* argv[])
{
	glewExperimental = true;
	if(!glfwInit()){
		fprintf(stderr, "Failed to initialize GLFW\n");
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); //opengl version 3.3
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL 

	GLFWwindow *window;
	window = glfwCreateWindow(1024, 768, "Tutorial 01", NULL, NULL);
	if(window == NULL){
		fprintf(stderr, "Failed to openg GLFW window, If you have an Intel GPU,they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window); //Initialize GLFW
	glewExperimental = true;
	if(glewInit()  != GLEW_OK){
		fprintf(stderr, "Failed to intialize GLEW\n");
		return 0;
	}
	//Ensure we can capture the esacap key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	do{
		//clear the screen
		glClear(GL_COLOR_BUFFER_BIT);
		//do nothing
		//swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();
	}while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);
	
	// Close OpenGL window and terminate GLFW
	glfwTerminate();
	return 0;
}
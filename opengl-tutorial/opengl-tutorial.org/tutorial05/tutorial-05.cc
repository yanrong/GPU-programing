#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../common/texture.hpp"
#include "../common/loadShader.hpp"
using namespace glm;

// Our vertices. Three consecutive floats give a 3D vertex; Three consecutive vertices give a triangle.
// A cube has 6 faces with 2 triangles each, so this makes 6*2=12 triangles, and 12*3 vertices
static const GLfloat g_vertex_buffer_data[] = {
    -1.0f,-1.0f,-1.0f, // triangle 1 : begin
    -1.0f,-1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f, // triangle 1 : end
     1.0f, 1.0f,-1.0f, // triangle 2 : begin
    -1.0f,-1.0f,-1.0f,
    -1.0f, 1.0f,-1.0f, // triangle 2 : end
     1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f,-1.0f,
     1.0f,-1.0f,-1.0f,
     1.0f, 1.0f,-1.0f,
     1.0f,-1.0f,-1.0f,
    -1.0f,-1.0f,-1.0f,
    -1.0f,-1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f,-1.0f,
     1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f,-1.0f, 1.0f,
     1.0f,-1.0f, 1.0f,
     1.0f, 1.0f, 1.0f,
     1.0f,-1.0f,-1.0f,
     1.0f, 1.0f,-1.0f,
     1.0f,-1.0f,-1.0f,
     1.0f, 1.0f, 1.0f,
     1.0f,-1.0f, 1.0f,
     1.0f, 1.0f, 1.0f,
     1.0f, 1.0f,-1.0f,
    -1.0f, 1.0f,-1.0f,
     1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f,
     1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f,
     1.0f,-1.0f, 1.0f
};

// Two UV coordinatesfor each vertex. They were created with Blender.
static const GLfloat g_uv_buffer_data[] = {
	0.000059f, 1.0f-0.000004f,
	0.000103f, 1.0f-0.336048f,
	0.335973f, 1.0f-0.335903f,
	1.000023f, 1.0f-0.000013f,
	0.667979f, 1.0f-0.335851f,
	0.999958f, 1.0f-0.336064f,
	0.667979f, 1.0f-0.335851f,
	0.336024f, 1.0f-0.671877f,
	0.667969f, 1.0f-0.671889f,
	1.000023f, 1.0f-0.000013f,
	0.668104f, 1.0f-0.000013f,
	0.667979f, 1.0f-0.335851f,
	0.000059f, 1.0f-0.000004f,
	0.335973f, 1.0f-0.335903f,
	0.336098f, 1.0f-0.000071f,
	0.667979f, 1.0f-0.335851f,
	0.335973f, 1.0f-0.335903f,
	0.336024f, 1.0f-0.671877f,
	1.000004f, 1.0f-0.671847f,
	0.999958f, 1.0f-0.336064f,
	0.667979f, 1.0f-0.335851f,
	0.668104f, 1.0f-0.000013f,
	0.335973f, 1.0f-0.335903f,
	0.667979f, 1.0f-0.335851f,
	0.335973f, 1.0f-0.335903f,
	0.668104f, 1.0f-0.000013f,
	0.336098f, 1.0f-0.000071f,
	0.000103f, 1.0f-0.336048f,
	0.000004f, 1.0f-0.671870f,
	0.336024f, 1.0f-0.671877f,
	0.000103f, 1.0f-0.336048f,
	0.336024f, 1.0f-0.671877f,
	0.335973f, 1.0f-0.335903f,
	0.667969f, 1.0f-0.671889f,
	1.000004f, 1.0f-0.671847f,
	0.667979f, 1.0f-0.335851f
};

int main(int argc, char* argv[])
{
	GLFWwindow *window;
	GLuint vertexBuffer;
    GLuint uvBuffer;
	GLuint vertextArrayID;

	if(!glfwInit()){
		fprintf(stderr, "Failed to initialize GLFW\n");
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); //opengl version 3.3
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);// To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    //Open a window and create its OpenGL context
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

    //Enable depth test
    glEnable(GL_DEPTH_TEST);
    //Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);

	//Generate Vertex Array Objection(VAO)
	glGenVertexArrays(1, &vertextArrayID);
	glBindVertexArray(vertextArrayID);

	//Create and Compile our GLSL program from the shaders
	GLuint programID = loadShader("./shader/transform.vert", "./shader/texture.frag");

    //Get a handle for our "MVP" uniform
    GLuint matrixID = glGetUniformLocation(programID, "mvp");
    //Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 4.0f / 3.0f, 0.1f, 100.0f);

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

    //Load the texture using any two methods
    GLuint texture = loadBMP("uvtemplate.bmp");

    //Get a handle for our "textureSampler" uniform
    GLuint textureID = glGetUniformLocation(programID, "textureSampler");
    // 1rst attribute buffer : vertices
    glGenBuffers(1, &vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

    glGenBuffers(1, &uvBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_uv_buffer_data), g_uv_buffer_data, GL_STATIC_DRAW);

    do{
		//clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        //Use our shader
		glUseProgram(programID);

        // Send our transformation to the currently bound shader,
		// in the "MVP" uniform
		glUniformMatrix4fv(matrixID, 1, GL_FALSE, &mvp[0][0]);

        //Bind our texture in Texture Unit 0
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        //Set our "textureSample" sampler to use Texture
        glUniform1i(textureID, 0);

        // 1rst attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
        glVertexAttribPointer(
			0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);

        // 2dn attribute buffer, UVs
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
        glVertexAttribPointer(
			1,                  // attribute. No particular reason for 0, but must match the layout in the shader.
			2,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);
        //Draw the triangle !
		glDrawArrays(GL_TRIANGLES, 0, 12 * 3); // Starting from vertex 0; 3 vertices total

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);

		//swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();
	} while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS /*check the ESCA key*/
		&& glfwWindowShouldClose(window) == 0);

	//clean VBO and shader
	glDeleteBuffers(1, &vertexBuffer);
    glDeleteBuffers(1, &uvBuffer);
	glDeleteVertexArrays(1, &vertextArrayID);
    glDeleteTextures(1, &texture);
	glDeleteProgram(programID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();
	return 0;
}

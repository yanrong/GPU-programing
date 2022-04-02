#include <iostream>
#include <cstring>

#include <cmath>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/matrix.hpp>
#include <glm/gtc/type_ptr.hpp>

static void renderSceneCB();
static void initializeGLUTCallbacks();
static void createVertexBuffer();
static void addShaders(GLuint program, const char* souce, GLenum type);
static void compileShader();

GLuint VBO, worldLocation;

int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(1024, 768);
    glutInitWindowPosition(100, 100);

    initializeGLUTCallbacks();

    //musty be done after glut is initialized
    GLenum res = glewInit();
    if (res!= GLEW_OK) {
        std::cerr << "Error :" << glewGetErrorString(re) << std::endl;
        return -1;
    }

    std::cout <<"GL version :" << glGetString(GL_VERSION);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    createVertexBuffer();
    compileShader();

    glutMainLoop();
    return 0;
}

static void renderSceneCB()
{
    glClear(GL_COLOR_BUFFER_BIT);

    static GLfloat scale = 0.0f;

    scale += 0.001f;
    GLfloat vector[] = {{sin(scale), 0.0f, 0.0f, 0.0f},
                        {0.0f, sin(scale), 0.0f, 0.0f},
                        {0.0f, 0.0f, sin(scale), 0.0f},
                        {0.0f, 0.0f, 0.0f, sin(scale)}};
    glm::mat4 diagonal = glm::make_mat4(vector);

    glUniformMatrix4fv(worldLocation, 1, GL_TRUE, glm::value_ptr(diagonal));

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);

    glDrawArrays(GL_TRIANGLES, 0, 3);

    glDisableVertexAttribArray(0);
    glutSwapBuffers();
}

static void initializeGLUTCallbacks()
{
    glutDisplayFunc(renderSceneCB);
    glutIdleFunc(renderSceneCB);
}

static void createVertexBuffer()
{
    GLFloat vertices[] = {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    };

    glGenVertexArrays(1, VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
}

static void addShaders(GLuint program, const char* souce, GLenum type)
{
    GLuint shader = glCreateShader(type);
    GLint lenght, success;

    if (shader == 0) {
        std::cerr << "Error creating shader type:" << type << std::endl;
        return ;
    }

    length = strlen(source);

    glShaderSource(shader, 1, source, length);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[1024];
        glGetShaderInfoLog(shader, 1024, NULL, infoLog);
        std::cerr << "Error compile the shader source, Type " << type << " " << infoLog << std::endl;
        return ;
    }

    glAttachShader(program, shader);

}

static void compileShader(const char* vertex, const char *frag)
{
    GLuint program = glCreateProgram();

    if (program == NULL) {
        std::cerr << "Error creating shader program" << std::endl;
        return ;
    }


}
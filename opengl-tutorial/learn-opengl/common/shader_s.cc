#include <glad/glad.h>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "shader_s.hpp"

Shader::Shader(const char* vertexPath, const char* fragmentPath)
{
    std::string vertexString, fragmentString;
    std::ifstream vShaderFStream, fShaderFStream;
    std::stringstream vShaderSStream, fShaderSStream;

    unsigned int vertex, fragment;

    //ensure ifstream objects can throw exception
    vShaderFStream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fShaderFStream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
        //openg files, associate a file stream to a specify file
        vShaderFStream.open(vertexPath);
        fShaderFStream.open(fragmentPath);

        //read file's buffer contents into string stream
        vShaderSStream << vShaderFStream.rdbuf();
        fShaderSStream << fShaderFStream.rdbuf();

        //close file handlers
        vShaderFStream.close();
        fShaderFStream.close();
        //contvert stream into string
        vertexString = vShaderSStream.str();
        fragmentString = fShaderSStream.str();
    } catch (std::ifstream::failure & e) {
        std::cout << "Error::Shader::File Not Successful Read" << std::endl;
    }

    const char* vShaderCode = vertexString.c_str();
    const char* fShaderCode = fragmentString.c_str();
    // compile shaders
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);
    checkCompileErrors(vertex, "VERTEX");

    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);
    checkCompileErrors(fragment, "VERTEX");

    //shader Program
    Id = glCreateProgram();
    glAttachShader(Id, vertex);
    glAttachShader(Id, fragment);
    glLinkProgram(Id);
    checkCompileErrors(Id, "PROGRAM");

    //delete the shaders as they're linked into our program, ew needn't it anymore
    glDeleteShader(vertex);
    glDeleteShader(fragment);
}

Shader::Shader(const char* vertexPath, const char* fragmentPath, const char *geometryPath)
{
    std::string vertexString, fragmentString, geometryString;
    std::ifstream vShaderFStream, fShaderFStream, gShaderFStream;
    std::stringstream vShaderSStream, fShaderSStream, gShaderSStream;

    unsigned int vertex, fragment, geometry;

    //ensure ifstream objects can throw exception
    vShaderFStream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fShaderFStream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    gShaderFStream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
        //openg files, associate a file stream to a specify file
        vShaderFStream.open(vertexPath);
        fShaderFStream.open(fragmentPath);
        gShaderFStream.open(geometryPath);

        //read file's buffer contents into string stream
        vShaderSStream << vShaderFStream.rdbuf();
        fShaderSStream << fShaderFStream.rdbuf();
        gShaderSStream << gShaderFStream.rdbuf();

        //close file handlers
        vShaderFStream.close();
        fShaderFStream.close();
        gShaderFStream.close();

        //contvert stream into string
        vertexString = vShaderSStream.str();
        fragmentString = fShaderSStream.str();
        geometryString = gShaderSStream.str();
    } catch (std::ifstream::failure & e) {
        std::cout << "Error::Shader::File Not Successful Read" << std::endl;
    }

    const char* vShaderCode = vertexString.c_str();
    const char* fShaderCode = fragmentString.c_str();
    const char* gShaderCode = geometryString.c_str();

    // compile shaders
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);
    checkCompileErrors(vertex, "VERTEX");

    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);
    checkCompileErrors(fragment, "VERTEX");

    geometry = glCreateShader(GL_GEOMETRY_SHADER);
    glShaderSource(geometry, 1, &gShaderCode, NULL);
    glCompileShader(geometry);
    checkCompileErrors(geometry, "GEOMETRY");

    //shader Program
    Id = glCreateProgram();
    glAttachShader(Id, vertex);
    glAttachShader(Id, fragment);
    glAttachShader(Id, geometry);
    glLinkProgram(Id);
    checkCompileErrors(Id, "PROGRAM");

    //delete the shaders as they're linked into our program, ew needn't it anymore
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    glDeleteShader(geometry);
}

//activate the shader
void Shader::use()
{
    glUseProgram(Id);
}

//utility uniform functions
void Shader::setBool(const std::string &name, bool value) const
{
    glUniform1i(glGetUniformLocation(Id, name.c_str()), (int)value);
}

void Shader::setInt(const std::string &name, int value) const
{
    glUniform1i(glGetUniformLocation(Id, name.c_str()), value);
}

void Shader::setFloat(const std::string &name, float value) const
{
    glUniform1f(glGetUniformLocation(Id, name.c_str()), value);
}

void Shader::setVec2(const std::string &name, const glm::vec2 &vector2) const
{
    glUniform2fv(glGetUniformLocation(Id, name.c_str()), 1, &vector2[0]);
}

void Shader::setVec2(const std::string &name, float x, float y) const
{
    glUniform2f(glGetUniformLocation(Id, name.c_str()), x, y);
}

void Shader::setVec3(const std::string &name, const glm::vec3 &vector3) const
{
    glUniform3fv(glGetUniformLocation(Id, name.c_str()), 1, &vector3[0]);
}

void Shader::setVec3(const std::string &name, float x, float y, float z) const
{
    glUniform3f(glGetUniformLocation(Id, name.c_str()), x, y, z);
}

void Shader::setVec4(const std::string &name, const glm::vec4 &vector4) const
{
    glUniform4fv(glGetUniformLocation(Id, name.c_str()), 1, &vector4[0]);
}

void Shader::setVec4(const std::string &name, float x, float y, float z, float w) const
{
    glUniform4f(glGetUniformLocation(Id, name.c_str()), x, y, z, w);
}

void Shader::setMat2(const std::string &name, const glm::mat2 &matrix2) const
{
    glUniformMatrix2fv(glGetUniformLocation(Id, name.c_str()), 1, GL_FALSE, &matrix2[0][0]);
}

void Shader::setMat3(const std::string &name, const glm::mat3 &matrix3) const
{
    glUniformMatrix3fv(glGetUniformLocation(Id, name.c_str()), 1, GL_FALSE, &matrix3[0][0]);
}

void Shader::setMat4(const std::string &name, const glm::mat4 &matrix4) const
{
    glUniformMatrix4fv(glGetUniformLocation(Id, name.c_str()), 1, GL_FALSE, &matrix4[0][0]);
}

//utility function for checking shader compilation status
void Shader::checkCompileErrors(unsigned int shader, std::string type)
{
    int status;
    char infoLog[1024];
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
        if (!status) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog  << std::endl;
        }
    } else {
        glGetProgramiv(shader, GL_LINK_STATUS, &status);
        if (!status) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << std::endl;
        }
    }
}
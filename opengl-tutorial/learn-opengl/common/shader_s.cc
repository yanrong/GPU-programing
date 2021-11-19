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
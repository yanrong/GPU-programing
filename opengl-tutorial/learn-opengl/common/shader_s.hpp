#ifndef SHADER_H
#define SHADER_H

class Shader {
public:
    unsigned int Id;
    Shader(const char *vertexPath, const char *fragmentPath);
    void use();
    void setBool(const std::string &name, bool value);
    void setInt(const std::string &name, int value);
    void setFloat(const std::string &name, float value);
private:
    void checkCompileErrors(unsigned int shader, std::string type);
}
#endif // SHDER_H
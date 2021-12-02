#ifndef SHADER_H
#define SHADER_H

class Shader {
public:
    unsigned int Id;
    Shader(const char *vertexPath, const char *fragmentPath);
    void use();
    void setBool(const std::string &name, bool value) const;
    void setInt(const std::string &name, int value) const;
    void setFloat(const std::string &name, float value) const;
    void setVec2(const std::string &name, const glm::vec2 &vector2) const;
    void setVec2(const std::string &name, float x, float y) const;
    void setVec3(const std::string &name, const glm::vec3 &vector3) const;
    void setVec3(const std::string &name, float x, float y, float z) const;
    void setVec4(const std::string &name, const glm::vec4 &vector4) const;
    void setVec4(const std::string &name, float x, float y, float z, float w) const;
    void setMat2(const std::string &name, const glm::mat2 &matrix2) const;
    void setMat3(const std::string &name, const glm::mat3 &matrix3) const;
    void setMat4(const std::string &name, const glm::mat4 &matrix4) const;

private:
    void checkCompileErrors(unsigned int shader, std::string type);
};

#endif // SHDER_H
#ifndef _MESH_H_
#define _MESH_H_

#include "shader_s.hpp"

#define MAX_BONE_INFLUENCE 4
struct vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec3 tangent;
    glm::vec3 bitAngent;

    int mBoneIds[MAX_BONE_INFLUENCE];
    float mWeights[MAX_BONE_INFLUENCE];
};

struct texture {
    unsigned int id;
    std::string type;
    std::string path;
};

class mesh {
public:
    std::vector<vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<texture> textures;
    GLuint VAO;

    mesh(std::vector<vertex> vertices, std::vector<unsigned int> indices, std::vector<texture> textures);
    void draw(Shader &shader);
private:
    GLuint VBO, EBO;
    void setupMesh();
};

#endif /* end of _MESH_H_ */
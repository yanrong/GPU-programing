#include <string>
#include <vector>

#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "mesh.hpp"

mesh::mesh(std::vector<vertex> vertices, std::vector<unsigned int> indices, std::vector<texture> textures)
{
    this->vertices = vertices;
    this->indices = indices;
    this->textures = textures;

    //get all data and setup the mesh for rendering
    setupMesh();
}

void mesh::draw(Shader &shader)
{
    //bind the appropriate textures
    unsigned int diffuseNR  = 1;
    unsigned int specularNR = 1;
    unsigned int normalNR   = 1;
    unsigned int heightNR   = 1;

    for (int i = 0; i < textures.size(); i++)
    {
        glActiveTexture(GL_TEXTURE0 + i); //active the texture
        std::string number;
        std::string name = textures[i].type;
        if (name == "texture_diffuse") {
            number = std::to_string(diffuseNR++); // transfer unsigned int to string
        } else if (name == "texture_specular") {
            number = std::to_string(specularNR++);
        } else if (name == "texture_normal") {
            number = std::to_string(normalNR++);
        } else if (name == "texture_height") {
            number = std::to_string(heightNR++);
        }

        //now set the sampler to the correct texture
        glUniform1i(glGetUniformLocation(shader.Id, (name + number).c_str()), i);
        //bind to texture
        glBindTexture(GL_TEXTURE_2D, textures[i].id);
    }

    //draw mesh, EBO
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    //set the active texture to OpenGL default
    glActiveTexture(GL_TEXTURE0);
}

void mesh::setupMesh()
{
    //create buffer/arrays
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    //load data to to vertex buffers
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // A great thing about structs is that their memory layout is sequential for all its items.
    // The effect is that we can simply pass a pointer to the struct and it translates perfectly to a glm::vec3/2 array which
    // again translates to 3/2 floats which translates to a byte array.
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertex), &vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

    //setup the vertext attribute pointers
    //vertex position
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)0);
    //vertex normal
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, normal));
    //vertex texture coordinate
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, texCoord));
    //vertex tangent
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, tangent));

    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, bitAngent));
    //ids
    glEnableVertexAttribArray(5);
    glVertexAttribIPointer(5, 4, GL_INT, sizeof(vertex), (void *)offsetof(vertex, mBoneIds));
    //weights
    glEnableVertexAttribArray(6);
    glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, mWeights));

    glBindVertexArray(0); //release current vertex array
}
#include "mesh.hpp"

mesh::mesh(std::vector<vertex> vertices, vector<unsigned int> indices, vector<texture> textures)
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
    GLuint diffuseNR = 1;
    GLuint specularNR = 1;
    GLuint normalNR = 1;
    GLuint heightNR = 1;

    for (int i = 0; i < textures.size(); i++)
    {
        glActiveTexture(GL_TEXTURE0 + i); //active the texture
        string number;
        string name = textures[i].type;
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
}
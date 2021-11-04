#include <vector>
#include <map>
#include <cstring>
#include <glm/glm.hpp>
#include "vboindexer.hpp"

// if v1 can be considered equal to v2, return true
bool is_near(float v1, float v2) {
    return fabs(v1 - v2) < 0.01f;
}

//Search through all already-exported virtices for similar one
//Similar = same positon + same UVs + same normal
bool getSimilarVertexIndex(glm::vec3& in_vectex,
        glm::vec2& in_uv,
        glm::vec3& in_normal,
        std::vector<glm::vec3>& out_vertices,
        std::vector<glm::vec2>& out_uvs,
        std::vector<glm::vec3>& out_normals,
        unsigned short& result)
{
    //Lame linear search
    for (int i = 0; i < out_vertices.size(); i++) {
        if (is_near(in_vectex.x, out_vertices[i].x) &&
            is_near(in_vectex.y, out_vertices[i].y) &&
            is_near(in_vectex.z, out_vertices[i].z) &&
            is_near(in_uv.x, out_uvs[i].x) &&
            is_near(in_uv.y, out_uvs[i].y) &&
            is_near(in_normal.x, out_normals[i].x) &&
            is_near(in_normal.y, out_normals[i].y) &&
            is_near(in_normal.z, out_normals[i].z)) {
                result = i;
                return true;
            }
    }
    return false;
}

void indexVBOSlow(std::vector<glm::vec3>& in_vertices,
        std::vector<glm::vec2>& in_uvs,
        std::vector<glm::vec3>& in_normals,

        std::vector<unsigned short>& out_indices,
        std::vector<glm::vec3>& out_vertices,
        std::vector<glm::vec2>& out_uvs,
        std::vector<glm::vec3>& out_normals)
{
    //Iterate each input vertex
    for (int i = 0; i < in_vertices.size(); i++) {
        //Try to find s similar vertex in out_XXX
        unsigned short index;
        bool found = getSimilarVertexIndex(in_vertices[i], in_uvs[i], in_normals[i],
                                    out_vertices, out_uvs, out_normals, index);
        if (found) { //A similar vertex already in output vertex, use it
            out_indices.push_back(index);
        } else { // If not, it needs to be added in the output data.
            out_vertices.push_back(in_vertices[i]);
            out_uvs.push_back(in_uvs[i]);
            out_normals.push_back(in_normals[i]);
            out_indices.push_back((unsigned short)out_vertices.size() - 1);
        }
    }
}

struct packedVertex {
    glm::vec3 position;
    glm::vec2 uv;
    glm::vec3 normal;

    bool operator<(const packedVertex that) const {
        return memcmp((void*) this, (void*)& that, sizeof(packedVertex)) > 0;
    }
};

bool getSimilarVertexIndexFast(packedVertex &packed,
    std::map<packedVertex, unsigned short>& vertexToOutIndex,
    unsigned short & result)
{
    std::map<packedVertex, unsigned short>::iterator it = vertexToOutIndex.find(packed);
    if (it == vertexToOutIndex.end()) {
        return false;
    } else {
        result = it->second;
        return true;
    }
}

void indexVBO(std::vector<glm::vec3>& in_vertices,
        std::vector<glm::vec2>& in_uvs,
        std::vector<glm::vec3>& in_normals,

        std::vector<unsigned short>& out_indices,
        std::vector<glm::vec3>& out_vertices,
        std::vector<glm::vec2>& out_uvs,
        std::vector<glm::vec3>& out_normals)
{
    std::map<packedVertex, unsigned short> vertexToOutIndex;
    //For each input vertex
    for (int i = 0; i < in_vertices.size(); i++) {
        packedVertex packed = {in_vertices[i], in_uvs[i], in_normals[i]};

        //Try to find a similar vertex in out_XXX
        unsigned short index;
        bool found = getSimilarVertexIndexFast(packed, vertexToOutIndex, index);
        if (found) { // A similar vertex is already in VBO, use it instead
            out_indices.push_back(index);
        } else { //If not, it needs to be added int the output data
            out_vertices.push_back(in_vertices[i]);
            out_uvs.push_back(in_uvs[i]);
            out_normals.push_back(in_normals[i]);
            unsigned short newIndex = (unsigned short)out_vertices.size() - 1;
            out_indices.push_back(newIndex);
            vertexToOutIndex[packed] = newIndex;
        }
    }
}

void indexVBO_TBN(std::vector<glm::vec3>& in_vertices,
        std::vector<glm::vec2>& in_uvs,
        std::vector<glm::vec3>& in_normals,
        std::vector<glm::vec3>& in_tangents,
        std::vector<glm::vec3>& in_bitangents,

        std::vector<unsigned short>& out_indices,
        std::vector<glm::vec3>& out_vertices,
        std::vector<glm::vec2>& out_uvs,
        std::vector<glm::vec3>& out_normals,
        std::vector<glm::vec3>& out_tangents,
        std::vector<glm::vec3>& out_bitangents)
{
    //For each input vertex
    for (int i = 0; i < in_vertices.size(); i++) {
        //Try toe find a similar vertex in out_XXX
        unsigned short index;
        bool found = getSimilarVertexIndex(in_vertices[i], in_uvs[i], in_normals[i],
                        out_vertices, out_uvs, out_normals, index);
        if (found) {//A similar vertex is already in the VBO, use it instead!
            out_indices.push_back(index);

            //Average the tangents and bitangents
            out_tangents[index] += in_tangents[i];
            out_bitangents[index] += in_bitangents[i];
        } else {
            out_vertices.push_back(in_vertices[i]);
            out_uvs.push_back(in_uvs[i]);
            out_normals.push_back(in_normals[i]);
            out_tangents.push_back(in_tangents[i]);
            out_bitangents.push_back(in_bitangents[i]);
            out_indices.push_back((unsigned short)out_vertices.size() - 1);
        }
    }
}

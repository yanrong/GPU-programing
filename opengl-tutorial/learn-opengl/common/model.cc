#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>

#include "mesh.hpp"
#include "model.hpp"

model::model(std::string const &path, bool gamma)
{
    gammaCorrection = gamma;
    loadModel(path);
}

void model::draw(Shader &shader)
{
    for(unsigned int i = 0; i < meshes.size(); i++) {
        meshes[i].draw(shader);
    }
}

void model::loadModel(std::string const &path)
{
    //read file via ASSIMP
    Assimp::Importer importer;
    const aiScene* iScene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
    //check for errors
    if (!iScene || iScene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !iScene->mRootNode) {
        std::cout << "Error::AssImp:: " << importer.GetErrorString() << std::endl;
        return ;
    }

    //retrieve the directory path of the filepath
    directory = path.substr(0, path.find_last_of('/'));
    //process assimp's root node
    processNode(iScene->mRootNode, iScene);
}

void model::processNode(aiNode* iNode, const aiScene* iScene)
{
    //proces each mesh located at the current node
    for (int i = 0; i < iNode->mNumMeshes; i++) {
        //the node object only contains indices to index the actual objects in the secen.
        //the secn contains all the data, node is just keep stuff organized
        aiMesh* iMesh = iScene->mMeshes[iNode->mMeshes[i]];
        meshes.push_back(processMesh(iMesh, iScene));
    }
    //after we've processed all of the meshed, recursively process each of the children nodes
    for (int i = 0; i < iNode->mNumChildren; i++) {
        processNode(iNode->mChildren[i], iScene);
    }
}

mesh model::processMesh(aiMesh* iMesh, const aiScene* iScene)
{
    //data to file
    std::vector<vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<texture> textures;

    //walk through each of the mesh's vertices
    for (int i = 0; i < iMesh->mNumVertices; i++) {
        vertex myVertex;
        glm::vec3 vector3;

        //position
        vector3.x = iMesh->mVertices[i].x;
        vector3.y = iMesh->mVertices[i].y;
        vector3.z = iMesh->mVertices[i].z;
        myVertex.position = vector3;
        //load normal if exist
        if (iMesh->HasNormals()) {
            vector3.x = iMesh->mNormals[i].x;
            vector3.y = iMesh->mNormals[i].y;
            vector3.z = iMesh->mNormals[i].z;
            myVertex.normal = vector3;
        }
        //load texture coordinate
        if (iMesh->mTextureCoords[0]) {
            glm::vec2 vector2;
            // a vertex can contain up to 8 different texture coordinates. We thus make the assumption that we won't
            // use models where a vertex can have multiple texture coordinates so we always take the first set (0).
            vector2.x = iMesh->mTextureCoords[0][i].x;
            vector2.y = iMesh->mTextureCoords[0][i].y;
            myVertex.texCoord = vector2;

            //tangent
            vector3.x = iMesh->mTangents[i].x;
            vector3.y = iMesh->mTangents[i].y;
            vector3.z = iMesh->mTangents[i].z;
            myVertex.tangent = vector3;

            //bitangent
            vector3.x = iMesh->mBitangents[i].x;
            vector3.y = iMesh->mBitangents[i].y;
            vector3.z = iMesh->mBitangents[i].z;
            myVertex.bitAngent = vector3;
        } else {
            myVertex.texCoord = glm::vec2(0.0f, 0.0f);
		}
        
		vertices.push_back(myVertex);
        
    }

    //now walk through each of the mesh's faces, (a face is a mesh its triangle)
    // and retrieve the corresponding vertex indices.
    for (int i = 0; i < iMesh->mNumFaces; i++) {
        aiFace face = iMesh->mFaces[i];
        //retrieve all indices of the face and store them in the indices vector
        for (int j = 0; j < face.mNumIndices; j++) {
            indices.push_back(face.mIndices[j]);
        }
    }

    //process materila
    aiMaterial* material = iScene->mMaterials[iMesh->mMaterialIndex];
    //assume a convention for sampler name in shaders, eache diffuse texture should be
    //named as texture_diffuse{N}, N represent a sequential number ranging from 1 to MAX_SAMPLER_NUM
    //diffuse : texture_diffuseN, specular: texture_specularN, normal : texture_normalN
    //1. diffuse maps
    std::vector<texture> diffuseMaps = loadMaterialTextures(material, aiTextureType_DIFFUSE, "texture_diffuse");
    textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());
    //2. specular maps
    std::vector<texture> specularMaps = loadMaterialTextures(material, aiTextureType_SPECULAR, "texture_specular");
    textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());
    //3. normal maps
    std::vector<texture> normalMaps = loadMaterialTextures(material, aiTextureType_HEIGHT, "texture_normal");
    textures.insert(textures.end(), normalMaps.begin(), normalMaps.end());
    //4. height maps
    std::vector<texture> heightMaps = loadMaterialTextures(material, aiTextureType_AMBIENT, "texture_height");
    textures.insert(textures.end(), heightMaps.begin(), heightMaps.end());

    //return a mesh objection create from the extracted mesh data
    return mesh(vertices, indices, textures);
}

std::vector<texture> model::loadMaterialTextures(aiMaterial* material, aiTextureType type, std::string typeName)
{
    std::vector<texture> textures;
    for (int i = 0; i< material->GetTextureCount(type); i++) {
        aiString iString;
        material->GetTexture(type, i, &iString);
        //check if texture was load before and if so, continue to next interation,skip loading a new texture
        bool skip = false;
        for (int j = 0; j < texturesLoaded.size(); j++) {
            if (std::strcmp(texturesLoaded[j].path.data(), iString.C_Str()) == 0) {
                textures.push_back(texturesLoaded[j]);
                skip = true;
                break; // a texture with the same file path has been already loaded, skip it
            }
        }

        if (!skip) {
            texture tmpTexture;
            tmpTexture.id = textureFromFile(iString.C_Str(), this->directory);
            tmpTexture.type = typeName;
            tmpTexture.path = iString.C_Str();
            textures.push_back(tmpTexture);
            //store it as texture loaded for entire model, to ensure we won't unnecesery load duplicate textures.
            texturesLoaded.push_back(tmpTexture);
        }
    }

    return textures;
}

unsigned int textureFromFile(const char* path, const std::string &directory, bool gamma)
{
    std::string filename = std::string(path);
    filename = directory + '/' + filename;

    unsigned int textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &nrComponents, 0);
    if (data) {
        GLenum format;
        if (nrComponents == 1) {
            format = GL_RED;
        } else if (nrComponents == 3) {
            format = GL_RGB;
        } else if (nrComponents == 4) {
            format = GL_RGBA;
        }

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    } else {
        std::cout << "Texture failed to load to path: " << path << std::endl;
    }
    //release the texture data
    stbi_image_free(data);

    return textureID;
}
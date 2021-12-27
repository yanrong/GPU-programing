#ifndef _MODEL_H_
#define _MODEL_H_

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
//Open STB ON
#include <stb/stb_image.h>
#include "shader_s.hpp"

unsigned int textureFromFile(const char* path, const std::string &directory, bool gamma = false);

class model
{
public:
    std::vector<texture> texturesLoaded;
    std::vector<mesh> meshes;
    std::string directory;
    bool gammaCorrection;

    model(std::string const &path, bool gamma = false);
    void draw(Shader &shader);

private:
    void loadModel(std::string const &path);
    void processNode(aiNode* node, const aiScene* scene);
    mesh processMesh(aiMesh* mesh, const aiScene* scene);
    std::vector<texture> loadMaterialTextures(aiMaterial* material, aiTextureType type, std::string typeName);
};

#endif /* end of _MODEL_H_ */
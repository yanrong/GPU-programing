#ifndef _MODEL_H_
#define _MODEL_H_

unsigned int textureFromFile(const char* path, const std::string &directory, bool gamma = false);

class model
{
public:
    std::vector<texture> texturesLoaded;
    std::vector<mesh> meshes;
    std::string directory;
    bool gammaCorrection;

    model(std::string const &path, bool gamma);
    void draw(Shader &shader);

private:
    void loadModel(std::string const &path);
    void processNode(aiNode* node, const aiScene* scene);
    void processMesh(aiMesh* mesh, const aiScene* scene);
    vector<texture> loadMaterialTextures(aiMaterial* material, aiTextureType type, std::string typeName);
};

#endif /* end of _MODEL_H_ */
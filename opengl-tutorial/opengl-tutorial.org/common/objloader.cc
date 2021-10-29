#include <vector>
#include <cstdio>
#include <cstring>
#include <string>
#include <glm/glm.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "objloader.hpp"

// Very, VERY simple OBJ loader.
// Here is a short list of features a real function would provide :
// - Binary files. Reading a model should be just a few memcpy's away, not parsing a file at runtime. In short : OBJ is not very great.
// - Animations & bones (includes bones weights)
// - Multiple UVs
// - All attributes should be optional, not "forced"
// - More stable. Change a line in the OBJ file and it crashes.
// - More secure. Change another line and you can inject code.
// - Loading from memory, stream, etc

bool loadOBJ(const char *path,
            std::vector<glm::vec3> &out_vertices,
            std::vector<glm::vec2> &out_uvs,
            std::vector<glm::vec3> &out_normals
            )
{
    printf("loading OBJ files %s...\n", path);

    std::vector<unsigned int> vertexIndices, uvIndices, normalIndices;
    std::vector<glm::vec3> tmp_vertices;
    std::vector<glm::vec2> tmp_uvs;
    std::vector<glm::vec3> tmp_normals;

    FILE* fp = fopen(path, "r");
    if (fp == NULL) {
        printf("Impossible to open the file!\n");
        getchar();
        return false;
    }

    while (true) {
        char lineHeader[128];
        //read the first word of the line
        int res = fscanf(fp, "%s", lineHeader);
        if (res == EOF) {
            break; //when meet EOF, quit
        }

        if (strcmp(lineHeader,"v") == 0) {
            glm::vec3 vertex;
            fscanf(fp, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z);
            tmp_vertices.push_back(vertex);
        } else if (strcmp(lineHeader, "vt") == 0) {
			glm::vec2 uv;
			fscanf(fp, "%f %f\n", &uv.x, &uv.y );
			uv.y = -uv.y; // Invert V coordinate since we will only use DDS texture, which are inverted. Remove if you want to use TGA or BMP loaders.
			tmp_uvs.push_back(uv);
		} else if (strcmp(lineHeader, "vn") == 0) {
			glm::vec3 normal;
			fscanf(fp, "%f %f %f\n", &normal.x, &normal.y, &normal.z );
			tmp_normals.push_back(normal);
		} else if (strcmp(lineHeader, "f") == 0) {
            std::string vertex1, vertex2, vertex3;
            unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
            int matches = fscanf(fp, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIndex[0], &uvIndex[0], &normalIndex[0],
                                    &vertexIndex[1], &uvIndex[1], &normalIndex[1],
                                    &vertexIndex[2], &uvIndex[2], &normalIndex[2]);
            if (matches != 9) {
                printf("File can't ve read by our simple paraser\n");
                fclose(fp);
                return false;
            }
            vertexIndices.push_back(vertexIndex[0]);
            vertexIndices.push_back(vertexIndex[1]);
            vertexIndices.push_back(vertexIndex[2]);
            uvIndices.push_back(uvIndex[0]);
            uvIndices.push_back(uvIndex[1]);
            uvIndices.push_back(uvIndex[2]);
            normalIndices.push_back(normalIndex[0]);
            normalIndices.push_back(normalIndex[1]);
            normalIndices.push_back(normalIndex[2]);
        } else {
            //probable a comment, eat up the rest of line, make nonsense code
            char stupidBuffer[1000];
            fgets(stupidBuffer, 100, fp);
        }
    }

    for (int i = 0; i < vertexIndices.size(); i++) {
        // get the indices of its attributes
        unsigned int vertexIndex = vertexIndices[i];
        unsigned int uvIndex = uvIndices[i];
        unsigned int normalIndex = normalIndices[i];

        //get the attributes thanks to the index
        glm::vec3 vertex = tmp_vertices[vertexIndex - 1];
        glm::vec2 uv = tmp_uvs[uvIndex - 1];
        glm::vec3 normal = tmp_normals[normalIndex - 1];

        //put the attributes in buffers
        out_vertices.push_back(vertex);
        out_uvs.push_back(uv);
        out_normals.push_back(normal);
    }

    fclose(fp);
    return true;
}

bool loadAssImp(const char *path,
                std::vector<unsigned short> &indices,
                std::vector<glm::vec3> &vertices,
                std::vector<glm::vec2> &uvs,
                std::vector<glm::vec3> &normals
                )
{
    Assimp::Importer importer;
    /*aiProcess_JoinIdenticalVertices | aiProcess_SortByPType*/
    const aiScene* scene = importer.ReadFile(path, 0);
    if (!scene) {
        fprintf(stderr, importer.GetErrorString());
        getchar();
        return false;
    }
    // In this simple example code we always use the 1rst mesh (in OBJ files there is often only one anyway)
    const aiMesh* mesh = scene->mMeshes[0];

    //fill vertices positions
    vertices.reserve(mesh->mNumVertices);
    for(int i = 0; i < mesh->mNumVertices; i++) {
        aiVector3D pos = mesh->mVertices[i];
        vertices.push_back(glm::vec3(pos.x, pos.x, pos.z));
    }

    // Fill vertices texture coordinates
	uvs.reserve(mesh->mNumVertices);
	for (int i = 0; i < mesh->mNumVertices; i++){
		aiVector3D UVW = mesh->mTextureCoords[0][i]; // Assume only 1 set of UV coords; AssImp supports 8 UV sets.
		uvs.push_back(glm::vec2(UVW.x, UVW.y));
	}

	// Fill vertices normals
	normals.reserve(mesh->mNumVertices);
	for(unsigned int i=0; i<mesh->mNumVertices; i++){
		aiVector3D n = mesh->mNormals[i];
		normals.push_back(glm::vec3(n.x, n.y, n.z));
	}

	// Fill face indices
	indices.reserve(3*mesh->mNumFaces);
	for (unsigned int i=0; i<mesh->mNumFaces; i++){
		// Assume the model has only triangles.
		indices.push_back(mesh->mFaces[i].mIndices[0]);
		indices.push_back(mesh->mFaces[i].mIndices[1]);
		indices.push_back(mesh->mFaces[i].mIndices[2]);
	}

	// The "scene" pointer will be deleted automatically by "importer"
	return true;
}
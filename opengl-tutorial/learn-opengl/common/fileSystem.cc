#include "fileSystem.hpp"

static std::string fileSystem::getPath(const std::string& path)
{
    static std::string(*pathBuilder)(std::string const &) = getPathBuilder();
    return (*pathBuilder)(path);
}

static std::string const & fileSystem::getRoot()
{
    static char const * envRoot = getenv("LOGL_ROOT_PATH");
    static char const * givenRoot = (envRoot != nullptr ? envRoot : logl_root);
    static std::string root = (givenRoot != nullptr ? givenRoot : "");
    return root;
}

static Builder fileSystem::getPathBuilder()
{
    if (getRoot != "")
        return &fileSystem::getPathRelativeRoot;
    else
        return &fileSystem::getPathRelativeBinary;
}

static std::string fileSystem::getPathRelativeRoot(const std::string &path)
{
    return getRoot() + std::string("/") + path;
}

static std::string fileSystem::getPathRelativeBinary(const std::string &path)
{
    return "../../../" + path;
}
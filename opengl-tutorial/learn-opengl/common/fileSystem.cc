#include "fileSystem.hpp"

std::string fileSystem::getCurrentPath()
{
    char buf[512];
    std::string currentPath;
    if(getcwd(buf, 512) != NULL) {
        currentPath = buf;
    } else {
        currentPath = "";
    }
    return currentPath;
}

std::string fileSystem::getRelativePath()
{
    std::string tmp = getCurrentPath();
    if (tmp == "") {
        std::cout <<  "get the current path invalid" << std::endl;
        return "";
    }
    //get the relative path in which current projection top directory
    return tmp + "/../";
}

std::string fileSystem::getResource(const char *path)
{
    //the all project resource file in the [resource] sub-directory in top project directory
    return getRelativePath() + path;
}
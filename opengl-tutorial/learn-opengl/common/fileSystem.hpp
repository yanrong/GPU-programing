#ifndef FILESYSTEM_H
#define FILESYSTEM_H

#include <string>
#include <iostream>
#include <unistd.h>
//Open the STB on


class fileSystem{
public:
    static std::string getResource(const char *path);

private:
    static std::string getRelativePath(void);
    static std::string getCurrentPath(void);
};
#endif
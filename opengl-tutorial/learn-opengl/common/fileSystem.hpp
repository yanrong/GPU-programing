#ifndef FILESYSTEM_H
#define FILESYSTEM_H

class fileSystem{
public:
    static std::string getResource(const char *path);

private:
    static std::string getRelativePath(void);
    static std::string getCurrentPath(void);
};
#endif
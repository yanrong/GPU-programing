#include <unistd.h>
char buf[512];

char * get_logl_root(char *buf)
{
    if(getcwd(buf, 512) == NULL)
        return NULL;
    return buf;
}

const char * logl_root = get_logl_root(buf);
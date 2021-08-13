#include <stdio.h>

GLenum errorCheck ()
{
   GLenum code;
   const GLubyte *string;

   code = glGetError ();
   if (code != GL_NO_ERROR)
   {
      string = gluErrorString (code);
      fprintf( stderr, "OpenGL error: %s\n", string );
   }

   return code;
}

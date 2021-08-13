#include <stdio.h>
#include <stdlib.h>

/*
** Create a null-terminated string from the contents of a file
** whose name is supplied as a parameter.  Return a pointer to
** the string, unless something goes wrong, in which case return
** a null pointer.
*/

GLchar *readTextFile( const char *name ) {
   FILE *fp;
   GLchar *content = NULL;
   int count=0;

   /* verify that we were actually given a name */
   if (name == NULL)  return NULL;

   /* attempt to open the file */
   fp = fopen( name, "rt" );   /* open the file */
   if (fp == NULL ) return NULL;

   /* determine the length of the file */
   fseek (fp, 0, SEEK_END);
   count = ftell (fp);
   rewind( fp );

   /* allocate a buffer and read the file into it */
   if( count > 0 ) {
      content = (GLchar *) malloc (sizeof(char) * (count+1));
      if( content != NULL ) {
         count = fread (content, sizeof(char), count, fp);
         content[count] = '\0';
      }
   }

   fclose (fp);

   return content;
}

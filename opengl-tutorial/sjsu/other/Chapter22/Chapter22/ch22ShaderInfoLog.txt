GLint length;
GLsizei num;
char *log;

glGetShaderiv (vertShader, GL_INFO_LOG_LENGTH, &length);
if( length > 0 ) {
   log = (char *) malloc (sizeof(char) * length);
   glGetShaderInfoLog (vertShader, length, &num, log);
   fprintf (stderr, "%s\n", log);
}

glGetProgramiv (program, GL_INFO_LOG_LENGTH, &length);
if( length > 0 ) {
   log = (char *) malloc (sizeof(char) * length);
   glGetProgramInfoLog (program, length, &num, log);
   fprintf (stderr, "%s\n", log);
}

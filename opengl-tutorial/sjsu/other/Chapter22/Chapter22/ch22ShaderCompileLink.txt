GLuint vertShader, fragShader, program;
GLchar *vertSource, *fragSource;
GLint status;

/* Create our vertex and fragment shader objects */
vertShader = glCreateShader (GL_VERTEX_SHADER);
fragShader = glCreateShader (GL_FRAGMENT_SHADER);

/* Load the vertex shader code */
vertSource = readTextFile ("simpleShader.vert");
if (vertSource == NULL) {
   fputs ("Failed to read vertex shader\n", stderr);
   exit (EXIT_FAILURE);
}

/* Load the fragment shader code */
fragSource = readTextFile ("simpleShader.frag");
if (fragSource == NULL) {
   fputs ("Failed to read fragment shader\n", stderr);
   exit (EXIT_FAILURE);
}

/* Attach the shader source code to the objects */
glShaderSource (vertShader, 1, (const GLchar **) &vertSource, NULL);
glShaderSource (fragShader, 1, (const GLchar **) &fragSource, NULL);
free (vertSource);
free (fragSource);

/* Compile the shaders and check for compilation errors */
glCompileShader (vertShader);
glCompileShader (fragShader);

glGetShaderiv (vertShader, GL_COMPILE_STATUS, &status);
if (status != GL_TRUE ) {
   fputs ("Error in vertex shader compilation\n", stderr);
   exit (EXIT_FAILURE);
}

glGetShaderiv (fragShader, GL_COMPILE_STATUS, &status);
if (status != GL_TRUE ) {
   fputs ("Error in fragment shader compilation\n", stderr);
   exit (EXIT_FAILURE);
}

/* Create the program object */
program = glCreateProgram ();

/* Attach the compiled shaders to the program */
glAttachShader (program, vertShader);
glAttachShader (program, fragShader);

/* Link the shader program and check for errors */
glLinkProgram (program);

glGetProgramiv (vertShader, GL_LINK_STATUS, &status);
if (status != GL_TRUE ) {
   fputs( "Error when linking shader program\n", stderr );
   exit (EXIT_FAILURE);
}

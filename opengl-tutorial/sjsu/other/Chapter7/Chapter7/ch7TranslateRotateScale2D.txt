glMatrixMode (GL_MODELVIEW);

glColor3f (0.0, 0.0, 1.0);
glRecti (50, 100, 200, 150);       // Display blue rectangle. 

glColor3f (1.0, 0.0, 0.0);
glTranslatef (-200.0, -50.0, 0.0); // Set translation parameters. 
glRecti (50, 100, 200, 150);       // Display red, translated rectangle. 

glLoadIdentity ( );                // Reset current matrix to identity. 
glRotatef (90.0, 0.0, 0.0, 1.0);   // Set 90-deg. rotation about z axis. 
glRecti (50, 100, 200, 150);       // Display red, rotated rectangle. 

glLoadIdentity ( );                // Reset current matrix to identity. 
glScalef (-0.5, 1.0, 1.0);         // Set scale-reflection parameters. 
glRecti (50, 100, 200, 150);       // Display red, transformed rectangle. 

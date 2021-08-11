#include <math.h>
#include <stdio.h>
//#include <stdlib.h>

#ifdef __APPLE__
  #include <OpenGL/gl.h>
  #include <OpenGL/glu.h>
  #include <GLUT/glut.h>
#elif __linux__
  #include <GL/glut.h>
#endif

// A general OpenGL initialization function.  Sets all of the initial parameters.
void InitGL (int width, int height)     // We call this right after our OpenGL window is created.
{
  glClearColor (255.0f, 255.0f, 255.0f, 0.0f); // This will clear the background color to white
  glClearDepth (1.0);           // Enables clearing of the depth buffer
  glDepthFunc (GL_LESS);        // The type of depth test to do
  glEnable (GL_DEPTH_TEST);     // Enables depth testing
  glShadeModel (GL_SMOOTH);     // Enables smooth color shading
  glMatrixMode (GL_PROJECTION);
  glLoadIdentity ();            // Reset the projection matrix
  gluPerspective (45.0f, (GLfloat) width / (GLfloat) height, 0.1f, 100.0f); // Calculate the aspect ratio of the window
  glMatrixMode (GL_MODELVIEW);
}

void ReSizeGLScene (int width, int height)
{
  if (height == 0)                  // Prevent A Divide By Zero If The Window Is Too Small
  {
    height = 1;
  }
  glViewport (0, 0, width, height); // Reset The Current Viewport And Perspective Transformation
  glMatrixMode (GL_PROJECTION);
  glLoadIdentity ();
  gluPerspective (45.0f, (GLfloat) width / (GLfloat) height, 0.1f, 100.0f);
  glMatrixMode (GL_MODELVIEW);
}

void DrawGLScene ()
{
  float radius = 1.0;
  float theta;
  float major_axis = 1.0;
  float minor_axis = 0.5;

  glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear The Screen And The Depth Buffer
  glLoadIdentity ();                                   // Reset The View
// swap buffers to display, since we're double buffered.
  glutSwapBuffers ();
}

void menu (int value)
{
  switch (value) 
  {
    case 1:
      printf ("menu -> sub_menu -> selection 1\n");
    break;
    case 2:
      printf ("menu -> sub_menu -> selection 2\n");
    break;
    case 3:
      printf ("menu -> sub_menu -> selection 3\n");
    break;
    case 4:
      printf ("menu -> sub_menu -> selection 4\n");
    break;
    case 5:
      printf ("menu -> sub_menu -> selection 5\n");
    break;
    case 6:
      printf ("menu -> sub_menu -> selection 6\n");
    break;
    case 7:
      printf ("menu -> sub_menu -> exit\n");
      exit(1);
    break;
    default: 
    break;
  }
  glutPostRedisplay();
}

int main (int argc, char *argv[]) 
{  
  int window;
  int glut_sub_menu, glut_sub_sub_menu;

  glutInit (&argc, argv);  
  glutInitDisplayMode (GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH);  
  glutInitWindowSize (640, 480);
  glutInitWindowPosition (0, 0);  
  window = glutCreateWindow ("Creating GLUT menus");  
  glut_sub_sub_menu = glutCreateMenu (menu);
  glutAddMenuEntry ("selection 1", 1);
  glutAddMenuEntry ("selection 2", 2);
  glutAddMenuEntry ("selection 3", 3);
  glut_sub_menu = glutCreateMenu (menu);
  glutAddMenuEntry("selection 4", 4);
  glutAddMenuEntry("selection 5", 5);
  glutAddSubMenu ("sub menu", glut_sub_sub_menu);
  glutCreateMenu (menu);
  glutAddMenuEntry("selection 6", 6);
  glutAddSubMenu ("menu", glut_sub_menu);
  glutAddMenuEntry("exit", 7);
  glutAttachMenu (GLUT_RIGHT_BUTTON);
  glutDisplayFunc (&DrawGLScene);  
  glutReshapeFunc (&ReSizeGLScene);
  InitGL (640, 480);
  glutMainLoop ();  
}
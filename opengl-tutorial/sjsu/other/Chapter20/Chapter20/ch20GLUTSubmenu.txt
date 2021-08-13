#include <GL/glut.h>

GLsizei winWidth = 400, winHeight = 400;  // Initial display-window size.

GLfloat red = 1.0, green = 1.0, blue = 1.0;    //  Initial color values.
GLenum renderingMode = GL_SMOOTH;               //  Initial fill method.

void init (void)
{
   glClearColor (0.6, 0.6, 0.6, 1.0);  // Set display-window color to gray.

   glMatrixMode (GL_PROJECTION);
   gluOrtho2D (0.0, 300.0, 0.0, 300.0);
}

void mainMenu (GLint renderingOption)
{
   switch (renderingOption) {
      case 1:  renderingMode = GL_FLAT;    break;
      case 2:  renderingMode = GL_SMOOTH;  break;
   }
   glutPostRedisplay ( );
}

/*  Set color values according to the submenu option selected.  */
void colorSubMenu (GLint colorOption)
{
   switch (colorOption) {
      case 1:
         red = 0.0;  green = 0.0;  blue = 1.0;
         break;
      case 2:
         red = 0.0;  green = 1.0;  blue = 0.0;
         break;
      case 3:
         red = 1.0;  green = 1.0;  blue = 1.0;
   }
   glutPostRedisplay ( );
}

void displayTriangle (void)
{
   glClear (GL_COLOR_BUFFER_BIT);

   glShadeModel (renderingMode);     //  Set fill method for triangle.
   glColor3f (red, green, blue); //  Set color for first two vertices.
   glBegin (GL_TRIANGLES);
      glVertex2i (280, 20); 
      glVertex2i (160, 280);
      glColor3f (1.0, 0.0, 0.0);  // Set color of last vertex to red.
      glVertex2i (20, 100);
   glEnd ( );

   glFlush ( );
}

void reshapeFcn (GLint newWidth, GLint newHeight)
{
   glViewport (0, 0, newWidth, newHeight);

   glMatrixMode (GL_PROJECTION);
   glLoadIdentity ( );
   gluOrtho2D (0.0, GLfloat (newWidth), 0.0, GLfloat (newHeight));

   displayTriangle ( );
   glFlush ( );
}

void main (int argc, char **argv)
{
   GLint subMenu;                       //  Identifier for submenu.

   glutInit (&argc, argv);
   glutInitDisplayMode (GLUT_SINGLE | GLUT_RGB);
   glutInitWindowPosition (200, 200);
   glutInitWindowSize (winWidth, winHeight);
   glutCreateWindow ("Submenu Example");

   init ( );
   glutDisplayFunc (displayTriangle);

   subMenu = glutCreateMenu (colorSubMenu);
      glutAddMenuEntry ("Blue", 1);
      glutAddMenuEntry ("Green", 2);
      glutAddMenuEntry ("White", 3);

   glutCreateMenu (mainMenu);      // Create main pop-up menu.
      glutAddMenuEntry ("Solid-Color Fill", 1);
      glutAddMenuEntry ("Color-Interpolation Fill", 2);
      glutAddSubMenu ("Color", subMenu);

   /*  Select menu option using right mouse button.  */
   glutAttachMenu (GLUT_RIGHT_BUTTON);

   glutReshapeFunc (reshapeFcn);

   glutMainLoop ( );
}

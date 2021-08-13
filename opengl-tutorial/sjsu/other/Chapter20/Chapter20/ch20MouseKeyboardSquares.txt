#include <GL/glut.h>
#inclue <stdlib.h>

GLsizei winWidth = 400, winHeight = 300;   // Initial display-window size.
GLint edgeLength = 10;                  // Initial edge length for square.

void init (void)
{
   glClearColor (0.0, 0.0, 1.0, 1.0)   // Set display-window color to blue.

   glMatrixMode (GL_PROJECTION);
   gluOrtho2D (0.0, 200.0, 0.0, 150.0);
}

void displayFcn (void)
{
   glClear (GL_COLOR_BUFFER_BIT);          //  Clear display window.

   glColor3f (1.0, 0.0, 0.0);              //  Set fill color to red.
}

void winReshapeFcn (GLint newWidth, GLint newHeight)
{
   /*  Reset viewport and projection parameters  */
   glViewport (0, 0, newWidth, newHeight);
   glMatrixMode (GL_PROJECTION);
   glLoadIdentity ( );
   gluOrtho2D (0.0, GLdouble (newWidth), 0.0, GLdouble (newHeight));

   /*  Reset display-window size parameters.  */
   winWidth  = newWidth;
   winHeight = newHeight;
}

/*  Display a red square with a selected edge-length size.  */
void fillSquare (GLint button, GLint action, GLint xMouse, GLint yMouse)
{
   GLint x1, y1, x2, y2;

   /*  Use left mouse button to select a position for the
    *  lower-left corner of the square.
    */
   if (button == GLUT_LEFT_BUTTON && action == GLUT_DOWN)
   {
      x1 = xMouse;
      y1 = winHeight - yMouse;
      x2 = x1 + edgeLength;
      y2 = y1 + edgeLength;
      glRecti (x1, y1, x2, y2);
   }
   else
      if (button == GLUT_RIGHT_BUTTON)   //  Use right mouse button to quit.
         exit (0);

   glFlush ( );
}

/*  Use keys 2, 3, and 4 to enlarge the square.  */
void enlargeSquare (GLubyte sizeFactor, GLint xMouse, GLint yMouse)
{
   switch (sizeFactor)
   {
      case '2':
         edgeLength *= 2;
         break;
      case '3':
         edgeLength *= 3;
         break;
      case '4':
         edgeLength *= 4;
         break;
      default:
         break;
   }
}

/*  Use function keys F2 and F4 for reduction factors 1/2 and 1/4.  */
void reduceSquare (GLint reductionKey, GLint xMouse, GLint yMouse)
{
   switch (reductionKey)
   {
      case GLUT_KEY_F2:
         edgeLength /= 2;
         break;
      case GLUT_KEY_F3:
         edgeLength /= 4;
         break;
      default:
         break;
   }
}

void main (int argc, char** argv)
{
   glutInit (&argc, argv);
   glutInitDisplayMode (GLUT_SINGLE | GLUT_RGB);
   glutInitWindowPosition (100, 100);
   glutInitWindowSize (winWidth, winHeight);
   glutCreateWindow ("Display Squares of Various Sizes");

   init ( );
   glutDisplayFunc (displayFcn);
   glutReshapeFunc (winReshapeFcn);
   glutMouseFunc (fillSquare);
   glutKeyboardFunc (enlargeSquare);
   glutSpecialFunc (reduceSquare);

   glutMainLoop ( );
}

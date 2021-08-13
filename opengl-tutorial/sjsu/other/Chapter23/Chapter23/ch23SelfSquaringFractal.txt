#include <GL/glut.h>
#include <stdlib.h>
#include <math.h>

/*  Set initial size of display window.  */
GLsizei winWidth = 600, winHeight = 600;   

/*  Set coordinate limits in complex plane.  */
GLfloat xComplexMin = -0.25, xComplexMax = 1.25;
GLfloat yComplexMin = -0.75, yComplexMax = 0.75;

struct complexNum
{
   GLfloat x, y;
};

void init (void)
{
   /*  Set color of display window to white.  */
   glClearColor (1.0, 1.0, 1.0, 0.0);
}

void plotPoint (complexNum z)
{
   glBegin (GL_POINTS);
      glVertex2f (z.x, z.y);
   glEnd ( );
}

void solveQuadraticEq (complexNum lambda, complexNum * z)
{
   GLfloat lambdaMagSq, discrMag;
   complexNum discr;
   static complexNum fourOverLambda = { 0.0, 0.0 };
   static GLboolean firstPoint = true;

   if (firstPoint) {
      /*  Compute the complex number: 4.0 divided by lambda.  */
      lambdaMagSq = lambda.x * lambda.x + lambda.y * lambda.y;
      fourOverLambda.x =  4.0 * lambda.x / lambdaMagSq;
      fourOverLambda.y = -4.0 * lambda.y / lambdaMagSq;
      firstPoint = false;
   }
   discr.x = 1.0 - (z->x * fourOverLambda.x - z->y * fourOverLambda.y);
   discr.y = z->x * fourOverLambda.y + z->y * fourOverLambda.x;
   discrMag = sqrt (discr.x * discr.x + discr.y * discr.y);

   /*  Update z, checking to avoid the square root of a negative number.  */
   if (discrMag + discr.x < 0)
      z->x = 0;
   else
      z->x = sqrt ((discrMag + discr.x) / 2.0);

   if (discrMag - discr.x < 0)
      z->y = 0;
   else
      z->y = 0.5 * sqrt ((discrMag - discr.x) / 2.0);

   /*  For half the points, use negative root, 
    *  placing point in quadrant 3.  
    */
   if (rand ( ) < RAND_MAX / 2) {
      z->x = -z->x;
      z->y = -z->y;
   }

   /*  When imaginary part of discriminant is negative, point
    *  should lie in quadrant 2 or 4, so reverse sign of x.
    */
   if (discr.y < 0)
      z->x = -z->x;

   /* Complete the calculation for the real part of z. */
   z->x = 0.5 * (1 - z->x);
}

void selfSqTransf (complexNum lambda, complexNum z, GLint numPoints)
{
   GLint k;

   /*  Skip the first few points.  */
   for (k = 0;  k < 10;  k++)
      solveQuadraticEq (lambda, &z);

   /*  Plot the specified number of transformation points.  */
   for (k = 0;  k < numPoints;  k++) {
      solveQuadraticEq (lambda, &z);
      plotPoint (z);
   }
}

void displayFcn (void)
{
   GLint numPoints = 10000;        // Set number of points to be plotted.
   complexNum lambda = { 3.0, 0.0 };  //  Set complex value for lambda.
   complexNum z0 = { 1.5, 0.4 };   //  Set initial point in complex plane.

   glClear (GL_COLOR_BUFFER_BIT);   //  Clear display window.

   glColor3f (0.0, 0.0, 1.0);       //  Set point color to blue.

   selfSqTransf (lambda, z0, numPoints);
   glFlush ( );
}

void winReshapeFcn (GLint newWidth, GLint newHeight)
{
   /*  Maintain an aspect ratio of 1.0, assuming that
    *  width of complex window = height of complex window.
    */
   glViewport (0, 0, newHeight, newHeight);

   glMatrixMode (GL_PROJECTION);
   glLoadIdentity ( );

   gluOrtho2D (xComplexMin, xComplexMax, yComplexMin, yComplexMax);

   glClear (GL_COLOR_BUFFER_BIT);
}

void main (int argc, char** argv)
{
   glutInit (&argc, argv);
   glutInitDisplayMode (GLUT_SINGLE | GLUT_RGB);
   glutInitWindowPosition (50, 50);
   glutInitWindowSize (winWidth, winHeight);
   glutCreateWindow ("Self-Squaring Fractal");

   init ( );
   glutDisplayFunc (displayFcn);
   glutReshapeFunc (winReshapeFcn);

   glutMainLoop ( );
}

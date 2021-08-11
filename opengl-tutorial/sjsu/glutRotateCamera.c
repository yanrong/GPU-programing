#ifdef __APPLE__
  #include <OpenGL/gl.h>
  #include <OpenGL/glu.h>
  #include <GLUT/glut.h>
#elif __linux__
  #include <GL/glut.h>
#endif
#include <math.h>

float camera_angle_degrees = 0;

void init (void)
{
  glShadeModel (GL_SMOOTH);
  glClearColor (1.0f, 1.0f, 1.0f, 0.0f);				
  glClearDepth (1.0f);
  glEnable (GL_DEPTH_TEST);
  glDepthFunc (GL_LEQUAL);
  glEnable (GL_COLOR_MATERIAL);
  glHint (GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
  glEnable (GL_LIGHTING);
  glEnable (GL_LIGHT0);
  GLfloat lightPos[4] = {-1.0, 1.0, 0.5, 0.0};
  glLightfv (GL_LIGHT0, GL_POSITION, (GLfloat *) &lightPos);
  glEnable (GL_LIGHT1);
  GLfloat lightAmbient1[4] = {0.0, 0.0,  0.0, 0.0};
  GLfloat lightPos1[4]     = {1.0, 0.0, -0.2, 0.0};
  GLfloat lightDiffuse1[4] = {0.5, 0.5,  0.3, 0.0};
  glLightfv (GL_LIGHT1,GL_POSITION, (GLfloat *) &lightPos1);
  glLightfv (GL_LIGHT1,GL_AMBIENT, (GLfloat *) &lightAmbient1);
  glLightfv (GL_LIGHT1,GL_DIFFUSE, (GLfloat *) &lightDiffuse1);
  glLightModeli (GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
}

void display (void)
{
  float sphere_radius;
  float red_sphere_position_x, red_sphere_position_y, red_sphere_position_z;
  float green_sphere_position_x, green_sphere_position_y, green_sphere_position_z;
  float blue_sphere_position_x, blue_sphere_position_y, blue_sphere_position_z;
  float camera_position_x, camera_position_y, camera_position_z;
  float center_x, center_y, center_z;
  float camera_angle_radians;

  sphere_radius = 1.0;
  red_sphere_position_y = -6.0f;
  green_sphere_position_y = -6.0f;
  blue_sphere_position_y = -6.0f;
  red_sphere_position_x = 6.0f;
  red_sphere_position_z = -3.25f;
  green_sphere_position_x = 5.0f;
  green_sphere_position_z = -5.0f;
  blue_sphere_position_x = 7.0f;
  blue_sphere_position_z = -5.0f;
  center_x = (red_sphere_position_x + green_sphere_position_x + blue_sphere_position_x) / 3.0f;
  center_y = -6.0f;
  center_z = (red_sphere_position_z + green_sphere_position_z + blue_sphere_position_z) / 3.0f;
  if (camera_angle_degrees >= 360.0f)
  {
    camera_angle_degrees = 0;
  }
  else
  {
    camera_angle_degrees = camera_angle_degrees + 1.0f;
  }
  camera_angle_radians = camera_angle_degrees * M_PI / 180.0f;
  camera_position_x = sin(camera_angle_radians) * 6.0f + 6.0f;
  camera_position_y = -6.0f;
  camera_position_z = cos(camera_angle_radians) * 6.0f - 4.50f;
  glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity ();
  gluLookAt (camera_position_x, camera_position_y, camera_position_z, center_x, center_y, center_z, 0.0f, 1.0f, 0.0f);
  glPushMatrix ();
  glTranslatef (red_sphere_position_x, red_sphere_position_y, red_sphere_position_z);
  glColor3f (1.0f, 0.0f, 0.0f);
  glutSolidSphere (sphere_radius, 50, 50);
  glPopMatrix ();
  glPushMatrix ();
  glTranslatef (green_sphere_position_x, green_sphere_position_y, green_sphere_position_z);
  glColor3f (0.0f, 1.0f, 0.0f);
  glutSolidSphere (sphere_radius, 50, 50);
  glPopMatrix ();
  glPushMatrix ();
  glTranslatef (blue_sphere_position_x, blue_sphere_position_y, blue_sphere_position_z);
  glColor3f (0.0f, 0.0f, 1.0f);
  glutSolidSphere (sphere_radius, 50, 50);
  glPopMatrix ();
  glutSwapBuffers();
  glutPostRedisplay();
}

void reshape (int w, int h)  
{
  glViewport (0, 0, w, h);
  glMatrixMode (GL_PROJECTION); 
  glLoadIdentity ();  
  if (h == 0)  
  { 
    gluPerspective (80, (float) w, 1.0, 5000.0);
  }
  else
  {
    gluPerspective (80, (float) w / (float) h, 1.0, 5000.0);
  }
  glMatrixMode (GL_MODELVIEW);  
  glLoadIdentity (); 
}

int main (int argc, char *argv[]) 
{
  int window;

  glutInit (&argc, argv);
  glutInitDisplayMode (GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH);
  glutInitWindowSize (1280, 720 ); 
  glutInitWindowPosition (0, 0);
  window = glutCreateWindow ("Rotating Camera Demonstration");
  init ();
  glutDisplayFunc (display);  
  glutReshapeFunc (reshape);
  glutMainLoop ();
}

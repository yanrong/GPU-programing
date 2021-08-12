#include <stdio.h>
#ifdef __APPLE__
  #include <OpenGL/gl.h>
  #include <OpenGL/glu.h>
  #include <GLUT/glut.h>
#elif __linux__
  #include <GL/glut.h>
#endif

void mouse(int button, int state, int x, int y)
{
  switch(button){
    case GLUT_LEFT_BUTTON:
      if(state == GLUT_UP){
        printf("left up\n");
      }else if(state == GLUT_DOWN){
        printf("left down\n");
      }
      break;
    case GLUT_MIDDLE_BUTTON:
      if(state == GLUT_UP){
        printf("middle up\n");
      }else if(state == GLUT_DOWN){
        printf("middle down\n");
      }
      break;
    break;
    case GLUT_RIGHT_BUTTON:
      if(state == GLUT_UP){
        printf("right up\n");
      }else if(state == GLUT_DOWN){
        printf("right down\n");
      }
      break;
    break;
    default: break;
  }
}

void keyboard(unsigned char key, int x, int y)
{
  switch(key)
  {
    case 27: // escape key
      exit(1);
    break;
    default: break;
  }
}

void arrow_keys(int key, int x, int y)
{
  switch(key)
  {
    case GLUT_KEY_UP:
      printf("key up\n");
    break;
    case GLUT_KEY_DOWN:
      printf("key down\n");
    break;
    case GLUT_KEY_LEFT:
      printf("key left\n");
    break;
    case GLUT_KEY_RIGHT:
      printf("key right\n");
    break;
    default: break;
  }
}

void motion(int x, int y)
{
  printf("postion (%d, %d)\n", x, y);
}

void reshape()
{

}

void display()
{

}

int main(int argc, char *argv[])
{
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
  glutInitWindowSize(800, 600);
  glutInitWindowPosition(0,0);
  glutCreateWindow("Mouse keyboard click demo");
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutMouseFunc(mouse);
  glutKeyboardFunc(keyboard);
  glutSpecialFunc(arrow_keys);
  //glutMotionFunc(motiong);
  //active the motion function if the mouse within window
  glutPassiveMotionFunc(motion);
  glutMainLoop();
}
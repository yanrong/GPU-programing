varying vec3 light, view;
attribute vec3 tangent;

void main()
{
   vec3 normal = vec3( normalize( gl_NormalMatrix * gl_Normal ) );

   vec3 binorm = normalize( cross( normal, tangent ) );

   view = -normalize( vec3( gl_ModelViewMatrix * gl_Vertex ) );
   light = normalize( vec3( gl_LightSource[0].position ) );

   vec3 tmp;
   tmp.x = dot( light, tangent );
   tmp.y = dot( light, binorm );
   tmp.z = dot( light, normal );
   light = tmp;

   tmp.x = dot( view, tangent );
   tmp.y = dot( view, binorm );
   tmp.z = dot( view, normal );
   view = tmp;

   gl_TexCoord[0] = gl_MultiTexCoord0;
   gl_Position = ftransform();
}

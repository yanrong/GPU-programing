// Phong vertex shader

void main() {
   vec3 normal, lightdir;
   vec4 color;
   float NdotL;

   color = gl_FrontMaterial.ambient * gl_LightSource[0].ambient;

   normal = normalize(gl_NormalMatrix * gl_Normal);
   lightdir = normalize( vec3(gl_LightSource[0].position) );
   NdotL = max( dot(normal, lightdir), 0.0 );

   color += NdotL *
       (gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse);

   if( NdotL > 0.0 )
   {
      vec3 view, reflection;
      float RdotV;

      view = vec3( -normalize(gl_ModelViewMatrix * gl_Vertex) );
      reflection = normalize( reflect(-lightdir, normal) );
      RdotV = max( dot( reflection, view ), 0.0 );

      color += gl_FrontMaterial.specular *
               gl_LightSource[0].specular * 
               pow( RdotV, gl_FrontMaterial.shininess );
   }

   gl_FrontColor = color;
   gl_Position = ftransform();
}

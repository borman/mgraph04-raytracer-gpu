#version 330

// just don't like these OGL's vec"n", cause in (DX,CUDA,Cg) they are float"n" 
//
#define float2 vec2
#define float3 vec3
#define float4 vec4
#define float4x4 mat4
#define float3x3 mat3

struct Material
{
  float3 ambient;
  float3 diffuse;
  float3 specular;
  float shininess;

  float kRefr;
  float3 glow;
};

struct Light
{
  float3 pos;
  float3 color;
  float3 att; // (const, linear, quad)
};

//===========================================================

// Shader in/out
in float2 fragmentTexCoord;
layout(location = 0) out vec4 fragColor;

// External data
uniform int g_screenWidth;
uniform int g_screenHeight;

uniform float3 g_bBoxMin;
uniform float3 g_bBoxMax;

uniform float4x4 g_rayMatrix;
uniform float4 g_bgColor;

uniform float g_time;

uniform Light g_lights[] = {
  Light(
    float3(10.0, 10.0, 10.0), // Position
    float3(1.0, 1.0, 1.0), // Color
    float3(3.0, 0.0, 0.001) // Attenuation
  ),
  Light(
    float3(5.0, 5.0, 7.0), // Position
    float3(0.2, 1.0, 0.2), // Color
    float3(2.0, 0.0, 0.01) // Attenuation
  ),
};

uniform Material g_mats[] = {
  // Ground
  Material(
    float3(0.4, 0.4, 0.4), // Ambient
    float3(1.0, 1.0, 1.0), // Diffuse
    float3(0.0, 0.0, 0.0), // Specular
    10, // Shininess
    1.0, // kRefraction
    float3(0.0, 0.0, 0.0) // Glow
  ),
  // Balls
  Material(
    float3(0.4, 0.1, 0.1), // Ambient
    float3(1.0, 0.1, 0.6), // Diffuse
    float3(1.0, 1.0, 1.0), // Specular
    1000, // Shininess
    0.7, // kRefraction
    float3(1.0, 1.0, 1.0) // Glow
  ),
  // Tiles
  Material(
    float3(0.1, 0.4, 0.1), // Ambient
    float3(0.6, 1.0, 0.1), // Diffuse
    float3(1.0, 1.0, 1.0), // Specular
    1000, // Shininess
    1.0, // kRefraction
    float3(0.0, 0.0, 0.0) // Glow
  )
};

uniform float4 g_fogColor = float4(0.95, 0.95, 1.0, 1.0);

//===========================================================

float maxcomp(float3 p)
{
  return max(p.x, max(p.y, p.z));
}

float sdBox(float3 p, float3 b)
{
  vec3 di = abs(p) - b;
  float mc = maxcomp(di);
  return min(mc, length(max(di, 0.0)));
}

float mball(float k, float3 p)
{
  return pow(dot(p, p), -k/2);
}

float ObjectBalls(float3 p)
{  
  const float tension = 2.5;

  /*
  float2 i;
  const float cellSize = 16;
  p.xy = modf(p.xy/cellSize, i)*cellSize - float2(8, 8);
  float shift = length(i);
  */

  //float vheight = 4.5 + 4*sin(0.1 * g_time);
  const float vheight = 4.5;
  float mballs = mball(tension, p - float3(-1.1, -1.1, 1))
               + mball(tension, p - float3(0, 2.5, vheight*0.33))
               + mball(tension, p - float3(2.5, 0, vheight*0.66))
               + mball(tension, p - float3(0, 0, vheight));
  return pow(mballs, -1/tension) - 1.5;
}

float ObjectTiles(float3 p)
{
  float2 i;
  const float cellSize = 16;
  p.xy = modf(abs(p.xy)/cellSize, i)*cellSize - float2(8, 8);
  return sdBox(p, float3(1, 1, 1));
}

float DistanceField(float3 p)
{
  return min(p.z, min(ObjectBalls(p), ObjectTiles(p)));
}

int MaterialId(float3 p)
{
  if (ObjectBalls(p) < 1e-2)
    return 1;
  if (ObjectTiles(p) < 1e-2)
    return 2;
  return 0;
}

float3 NormalField(float3 p)
{
  const float eps = 1e-4;
  float3 dx = float3(eps, 0, 0);
  float3 dy = float3(0, eps, 0);
  float3 dz = float3(0, 0, eps);

  return normalize(float3(DistanceField(p+dx) - DistanceField(p-dx),
                          DistanceField(p+dy) - DistanceField(p-dy),
                          DistanceField(p+dz) - DistanceField(p-dz)));
}

float4 EnvColor(float3 dir)
{
  const float3 blue = float3(0.2, 0.4, 1.0);
  const float3 tint = float3(0.95, 0.95, 1.0);
  return float4(mix(tint, blue, dir.z), 1);
  //return float4(abs(cross(dir, float3(1,1,1))), 1);
}

//===========================================================

bool RayMarch(float3 pos, float3 dir, float zmin, float zmax, 
              out float3 hit, out float z, out float prox, out int steps)
{
  const float minStepK = 1e-5;
  const float maxStepBase = 1e-2;
  const float maxStepK = 1; // Proportional to z
  const float epsK = 1e-5; // Proportional to z
  const float maxSteps = 1000;

	z = zmin;
  steps = 0;
  prox = 1e38f;
	while(++steps < maxSteps && z < zmax)
	{
	  hit = pos + z*dir;
    float dz = DistanceField(hit);
    if (dz < epsK*z) // Adaptive threshold => fix artefacts on objects far away, speedup possible
    {
      prox = 0;
      return true;
    }
	  z += clamp(dz, minStepK*z, maxStepBase + maxStepK*z);
    prox = min(prox, dz/z);
	}
  // Miss
  z = zmax;
  hit = pos + z*dir;
	return false;
}

float AmbientOcclusion(float3 p, float3 n)
{
  float delta = 0.1f;
  float blend = 0.8f;
  int iter = 10;

  float ao = 0;
  for (int i=iter; i>0; i--)
    ao = ao/2 + max(0.0f, i*delta - DistanceField(p + i*delta*n));

  return 1 - blend*ao;
}

float Diffuse(int matId, float3 norm, float3 toLight)
{
  // Lambert
  return max(0, dot(toLight, norm)); 
}

float Specular(int matId, float3 norm, float3 toEye, float3 toLight)
{
  // Blinn
  return pow(max(0, dot(norm, normalize(toEye+toLight))), g_mats[matId].shininess);
}

float Fresnel(float k, float3 norm, float3 toEye)
{
  float cos1 = dot(norm, toEye);
  float cos2s = 1 - k*k*(1 - cos1*cos1);
  if (cos2s < 0)
    return 1;
  else
  {
    float cos2 = sqrt(cos2s);
    float a = (k*cos1 - cos2) / (k*cos1 + cos2);
    float b = (k*cos2 - cos1) / (k*cos2 + cos1);
    return (a*a + b*b)/2;
  }
}

float Illumination(float3 pos, int id)
{
  float3 _hit;
  float _z;
  int _steps;
  float prox;
  float dist = distance(g_lights[id].pos, pos); // Distance
  if (RayMarch(pos, normalize(g_lights[id].pos-pos), 0, dist, _hit, _z, prox, _steps)) // Shadow
    return 0;
  float att = dot(g_lights[id].att, float3(1, dist, dist*dist)); // Attenuation
  return min(1.0, 5*prox)/att;
}

float Fog(float z)
{
  const float density = 0.0001;
  return clamp(1-exp(-density * z*z), 0, 1);
}

float4 Render(float3 pos, float3 dir, float tmin, float tmax)
{
  float3 hit;
  float _prox;
  float z;
  int steps;
  bool isHit = RayMarch(pos, dir, tmin, tmax, hit, z, _prox, steps);
  float fog = Fog(length((pos-hit).xy));

  float4 resColor;
  if (isHit)
  {
    //return float4(float3(fog), 1.0);
    float3 norm = NormalField(hit);
    float3 toEye = -dir;
    int matId = MaterialId(hit);
    float fresnel = Fresnel(g_mats[matId].kRefr, norm, toEye);

    float3 color = g_mats[matId].ambient * AmbientOcclusion(hit, norm); // Ambient
    color += g_mats[matId].glow * fresnel; // Glow

    for (int i=0; i<g_lights.length(); ++i)
    {
      float3 toLight = normalize(g_lights[i].pos - hit);
      float illum = Illumination(hit + 1e-2*norm, i);
      if (illum > 1e-5)
      {
        float3 shade = g_mats[matId].diffuse * Diffuse(matId, norm, toLight); // Diffuse
        shade += g_mats[matId].specular * Specular(matId, norm, toEye, toLight); // Specular
        color += illum * g_lights[i].color * shade;
      }
    }
   
    resColor = float4(color, 1);
  }
  else
    resColor = EnvColor(dir);

  return mix(resColor, g_fogColor, fog);
}

//===========================================================

bool RayBoxIntersection(float3 ray_pos, float3 ray_dir, float3 boxMin, float3 boxMax, 
                        out float tmin, out float tmax)
{
  float3 inv_ray_dir = float3(1.0f, 1.0f, 1.0f) / ray_dir;
  
  float3 v1 = (boxMin - ray_pos) * inv_ray_dir;
  float3 v2 = (boxMax - ray_pos) * inv_ray_dir;
  float3 lo = min(v1, v2);
  float3 hi = max(v1, v2);

  tmin = min(lo.x, min(lo.y, lo.z));
  tmax = max(hi.x, max(hi.y, hi.z));

  return tmax > 0 && tmin < tmax;
}

float3 Project(float2 xy, float2 wh)
{
	const float fov = 3.141592654f/(2.0f); 
  float3 ray_dir;
  
  ray_dir.xy = xy + float2(0.5, 0.5) - wh/2;
	ray_dir.z = -(wh.x)/tan(fov/2.0f);
	
  return normalize(ray_dir);
}

float4 Sample(float2 xy, float2 wh)
{
  // Ray: from (0,0,0); screen parallel to xy plane
  float3 ray_pos = float3(0,0,0); 
  float3 ray_dir = Project(xy, wh);
 
  // transform ray with matrix
  //
  ray_pos = (g_rayMatrix*float4(ray_pos,1)).xyz;
  ray_dir = float3x3(g_rayMatrix)*ray_dir;

  return Render(ray_pos, ray_dir, 0, 1e3);
}

void main(void)
{	
  float2 wh = float2(g_screenWidth, g_screenHeight); // Screen size
  float2 xy = fragmentTexCoord * wh; // Pixel coords

  fragColor = Sample(xy, wh);
  /*
  // 4x supersampling
  float2 dx = 0.5*float2(1,0);
  float2 dy = 0.5*float2(0,1);
  fragColor = 0.25 * (Sample(xy, wh) + Sample(xy+dx, wh) + Sample(xy+dy, wh) + Sample(xy+dx+dy, wh));
  */

  /*
  // intersect bounding box of the whole scene, if no intersection found return background color
  // 
  float tmin = 1e38f;
  float tmax = 0.0f;

  if(RayBoxIntersection(ray_pos, ray_dir, g_bBoxMin, g_bBoxMax, tmin, tmax))
    fragColor = Render(ray_pos, ray_dir, max(0, tmin), tmax);
  else
    fragColor = float4(0, 0, 0, 0);
    */
}


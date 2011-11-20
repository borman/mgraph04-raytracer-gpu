#version 330

// just don't like these OGL's vec"n", cause in (DX,CUDA,Cg) they are float"n" 
//
#define float2 vec2
#define float3 vec3
#define float4 vec4
#define float4x4 mat4
#define float3x3 mat3

in float2 fragmentTexCoord;

layout(location = 0) out vec4 fragColor;

uniform int g_screenWidth;
uniform int g_screenHeight;

uniform float3 g_bBoxMin;
uniform float3 g_bBoxMax;

uniform float3 g_lightPos;

uniform float4x4 g_rayMatrix;
uniform float4 g_bgColor;

uniform float g_time;

//===========================================================

float metaball(float3 p)
{
  return pow(dot(p, p), 1.5);
}

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

float DistanceField(float3 p)
{
  float s1 = metaball(p - float3(0, 0, 1));
  float s2 = metaball(p - float3(0, 3.5, 1));
  float s3 = metaball(p - float3(3.5, 0, 1));
  //float s3 = pow(sdBox(p - float3(3.5, 0, 1), float3(1, 1, 1)), 2);
  float s4 = metaball(p - float3(0, 0, 4.5 + 2*sin(0.1 * g_time)));
  //float s4 = pow(sdBox(p - float3(0, 0, 4.5 + 2*sin(0.1 * g_time)), float3(1, 1, 1)), 3);
  float s12 = pow(1/(1/s1 + 1/s2 + 1/s3 + 1/s4), 0.333) - 1.5;
  //float s12 = min(s1, s2);
  return min(p.z, s12);
}

float3 NormalField(float3 p)
{
  float eps = 1e-4;
  float3 dx = float3(eps, 0, 0);
  float3 dy = float3(0, eps, 0);
  float3 dz = float3(0, 0, eps);

  return normalize(float3(DistanceField(p+dx) - DistanceField(p-dx),
                          DistanceField(p+dy) - DistanceField(p-dy),
                          DistanceField(p+dz) - DistanceField(p-dz)));
}

//===========================================================

float RayMarchShadow(float3 pos, float3 dir, float tmin, float tmax)
{
  float kSmooth = 5;
	float t = tmin;
  float dens = 1.0f;
	while(t < tmax)
	{
	  float3 p = pos + t*dir;
    float dt = DistanceField(p);
    if (dt < 1e-4f)
      return 0.0f;
    dens = min(dens, kSmooth*dt/t);
	  t += max(1e-4, dt);
	}
	return dens;
}

bool RayMarchHit(float3 pos, float3 dir, float tmin, float tmax, out float3 hit)
{
	float t = tmin;	
	while(t < tmax)
	{
	  hit = pos + t*dir;
    float dt = DistanceField(hit);
    if (dt < 1e-4f)
      return true;
	  t += max(1e-4, dt);
	}
	return false;
}

float AmbientOcclusion(float3 p, float3 n)
{
  float delta = 0.1f;
  float blend = 1.0f;
  int iter = 10;

  float ao = 0;
  for (int i=iter; i>0; i--)
    ao = ao/2 + max(0.0f, i*delta - DistanceField(p + i*delta*n));

  return 1 - blend*ao;
}

float Shade(float3 pos, float3 norm, float3 eye, float3 light)
{
  float3 toLight = normalize(light - pos);
  float blinn = max(0, pow(dot(norm, normalize(eye+toLight)), 1000));
  float lambert = max(0, dot(toLight, norm));
  return (blinn + lambert) * RayMarchShadow(pos, toLight, 1e-2, distance(light, pos));
}

float4 Render(float3 pos, float3 dir)
{
  float3 hit;
  if (RayMarchHit(pos, dir, 1e-2, 1e2, hit))
  {
    float3 norm = NormalField(hit);
    float3 ambient = AmbientOcclusion(hit, norm) * float3(0.3, 0.3, 0.3);
    float3 shade = Shade(hit, norm, -dir, g_lightPos) * float3(1.0, 1.0, 1.0);
    float3 color = ambient + 0.2*shade;
    return float4(color.r, color.g, color.b, 1.0f);
  }
  else
    return float4(0, 0, 0, 0);
}

//===========================================================

bool RayBoxIntersection(float3 ray_pos, float3 ray_dir, float3 boxMin, float3 boxMax, out float tmin, out float tmax)
{
  ray_dir.x = 1.0f/ray_dir.x;
  ray_dir.y = 1.0f/ray_dir.y;
  ray_dir.z = 1.0f/ray_dir.z; 

  float lo = ray_dir.x*(boxMin.x - ray_pos.x);
  float hi = ray_dir.x*(boxMax.x - ray_pos.x);
  
  tmin = min(lo, hi);
  tmax = max(lo, hi);

  float lo1 = ray_dir.y*(boxMin.y - ray_pos.y);
  float hi1 = ray_dir.y*(boxMax.y - ray_pos.y);

  tmin = max(tmin, min(lo1, hi1));
  tmax = min(tmax, max(lo1, hi1));

  float lo2 = ray_dir.z*(boxMin.z - ray_pos.z);
  float hi2 = ray_dir.z*(boxMax.z - ray_pos.z);

  tmin = max(tmin, min(lo2, hi2));
  tmax = min(tmax, max(lo2, hi2));
  
  return (tmin <= tmax) && (tmax > 0.f);
}

float3 Project(float x, float y, float w, float h)
{
	float fov = 3.141592654f/(2.0f); 
  float3 ray_dir;
  
	ray_dir.x = x+0.5f - (w/2.0f);
	ray_dir.y = y+0.5f - (h/2.0f);
	ray_dir.z = -(w)/tan(fov/2.0f);
	
  return normalize(ray_dir);
}

void main(void)
{	
  float w = float(g_screenWidth);
  float h = float(g_screenHeight);
  
  // get curr pixelcoordinates
  //
  float x = fragmentTexCoord.x*w; 
  float y = fragmentTexCoord.y*h;
  
  // generate initial ray
  //
  float3 ray_pos = float3(0,0,0); 
  float3 ray_dir = Project(x,y,w,h);
 
  // transform ray with matrix
  //
  ray_pos = (g_rayMatrix*float4(ray_pos,1)).xyz;
  ray_dir = float3x3(g_rayMatrix)*ray_dir;
 
  // intersect bounding box of the whole scene, if no intersection found return background color
  // 
  float tmin = 1e38f;
  float tmax = 0.0f;
 
  if(RayBoxIntersection(ray_pos, ray_dir, g_bBoxMin, g_bBoxMax, tmin, tmax))
    fragColor = Render(ray_pos, ray_dir);
  else
    fragColor = float4(0, 0, 0, 0);
}
























































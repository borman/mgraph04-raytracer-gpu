//////////////////////////////////////////////////////////////////
// glHelper.h Author: Vladimir Frolov, 2011, Graphics & Media Lab.
//////////////////////////////////////////////////////////////////

#include "glHelper.h"

FullScreenQuad::FullScreenQuad()
{
  float quadPos[] =
  {
    -1.0f,  1.0f,	// v0 - top left corner
    -1.0f, -1.0f,	// v1 - bottom left corner
    1.0f,  1.0f,	// v2 - top right corner
    1.0f, -1.0f	  // v3 - bottom right corner
  };

  m_vertexBufferObject = 0;
  m_vertexLocation = 0; // simple layout, assume have only positions at location = 0

  glGenBuffers(1, &m_vertexBufferObject);                                                   OGL_CHECK_FOR_ERRORS;
  glBindBuffer(GL_ARRAY_BUFFER, m_vertexBufferObject);                                      OGL_CHECK_FOR_ERRORS;
  glBufferData(GL_ARRAY_BUFFER, 4*2*sizeof(GLfloat), (GLfloat*)quadPos, GL_STATIC_DRAW);    OGL_CHECK_FOR_ERRORS;

  glGenVertexArrays(1, &m_vertexArrayObject);                                               OGL_CHECK_FOR_ERRORS;
  glBindVertexArray(m_vertexArrayObject);                                                   OGL_CHECK_FOR_ERRORS;

  glBindBuffer(GL_ARRAY_BUFFER, m_vertexBufferObject);                    OGL_CHECK_FOR_ERRORS;                   
  glEnableVertexAttribArray(m_vertexLocation);                            OGL_CHECK_FOR_ERRORS;
  glVertexAttribPointer(m_vertexLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);   OGL_CHECK_FOR_ERRORS;

  glBindVertexArray(0);
}


FullScreenQuad::~FullScreenQuad()
{
  if(m_vertexBufferObject)
  {
    glDeleteBuffers(1, &m_vertexBufferObject);
    m_vertexBufferObject = 0;
  }
}


void FullScreenQuad::Draw()
{
  glBindVertexArray(m_vertexArrayObject); OGL_CHECK_FOR_ERRORS;
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 8);  OGL_CHECK_FOR_ERRORS;  // 4 vertices with 2 floats per vertex = 8 elements total.
}


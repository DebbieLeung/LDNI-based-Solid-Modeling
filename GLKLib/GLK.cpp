/*
 *  Copyright (C) 2014, Geometric Design and Manufacturing Lab in THE CHINESE UNIVERSITY OF HONG KONG
 *  All rights reserved.
 *   
 *		 http://ldnibasedsolidmodeling.sourceforge.net/
 *  
 *   
 *  Redistribution and use in source and binary forms, with or without modification, 
 *  are permitted provided that the following conditions are met:
 *  
 *  1. Redistributions of source code must retain the above copyright notice, 
 *     this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above copyright notice, 
 *     this list of conditions and the following disclaimer in the documentation 
 *	   and/or other materials provided with the distribution.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 *   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
 *   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
 *   IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
 *   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 *   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
 *   OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
 *   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 *   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
 *   OF SUCH DAMAGE.
 */


#define _CRT_SECURE_NO_DEPRECATE

#include <time.h>

#include "GLK.h"
#include "GLKGLList.h"
#include "GLKGeometry.h"




#define UI_TRANSFORM_INTERVAL 10


//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////




GLK::GLK()
{
	m_currentTool=NULL;
	m_HighLightObj=NULL;
	m_Shading=true;
	m_Mesh=true;
	m_Profile=true;
	m_Contouring = false;


	m_mouseState=0;		m_bCoordDisp=false;
	

	bClip = false;
	clipTrans[0] = 0.0;
	clipTrans[1] = 0.0;
	clipTrans[2] = 0.0;

	ClipEnable[0] = false;
	ClipEnable[1] = false;
	ClipEnable[2] = false;

	eqn[0][0] = 1.0;
	eqn[0][1] = 0.0;
	eqn[0][2] = 0.0;
	eqn[0][3] = 0.0; 

	eqn[1][0] = 0.0;
	eqn[1][1] = 1.0;
	eqn[1][2] = 0.0;
	eqn[1][3] = 0.0;

	eqn[2][0] = 0.0;
	eqn[2][1] = 0.0;
	eqn[2][2] = 1.0;
	eqn[2][3] = 0.0;
}

GLK::~GLK()
{
	ClearAll();
}

//////////////////////////////////////////////////////////////////////
// Implementation
//////////////////////////////////////////////////////////////////////

void GLK::ClearAll()
{
	clear_tools();
//	clear_gllist();

	GLKPOSITION Pos;
	for(Pos=m_displayObjList.GetHeadPosition();Pos!=NULL;)
	{
		GLKEntity *entity=(GLKEntity *)(m_displayObjList.GetNext(Pos));
		if (entity) delete entity;
	}
	ClearDisplayObjList();

	initValue();
	refresh();
}

void GLK::clear_tools()
{
	if (m_currentTool)
		delete m_currentTool;
	m_currentTool=NULL;
}

void GLK::set_tool(GLKMouseTool *tool)
{
	m_currentTool=tool;
}

void GLK::AddDisplayObj(GLKEntity *entity, bool bRefresh)
{
	float newRange = entity->getRange();
	float oldRange = m_Range;
	
	if ((newRange>m_Range) || ((m_displayObjList.GetCount())==0)) {
		m_Range=newRange;
		m_Scaling=m_Scaling*(newRange/oldRange);
	}
	m_displayObjList.AddTail(entity);
	if (bRefresh) refresh();
}

void GLK::DelDisplayObj(GLKEntity *entity)
{
	float newRange,oldRange=m_Range;
	m_Range=1.0f;

	GLKPOSITION Pos;	GLKObList tempList;
	tempList.RemoveAll();
	for(Pos=m_displayObjList.GetHeadPosition();Pos!=NULL;)
	{
		GLKEntity *tempEntity=(GLKEntity *)(m_displayObjList.GetNext(Pos));
		if (tempEntity!=entity) tempList.AddTail(tempEntity);
	}
	m_displayObjList.RemoveAll();	m_displayObjList.AddTail(&tempList);
	
	bool flag=true;
	for(Pos=m_displayObjList.GetHeadPosition();Pos!=NULL;)
	{
		GLKEntity *tempEntity=(GLKEntity *)(m_displayObjList.GetNext(Pos));
		newRange=tempEntity->getRange();
		if ((newRange>m_Range) || (flag))
		{
			m_Range=newRange;
			flag=false;
		}
	}

	m_Scaling=m_Scaling*(m_Range/oldRange);
	refresh();	

	delete entity;
}

void GLK::DelDisplayObj2(GLKEntity *entity)
{
	GLKPOSITION Pos;	GLKObList tempList;
	tempList.RemoveAll();
	for(Pos=m_displayObjList.GetHeadPosition();Pos!=NULL;)
	{
		GLKEntity *tempEntity=(GLKEntity *)(m_displayObjList.GetNext(Pos));
		if (tempEntity!=entity) tempList.AddTail(tempEntity);
	}
	m_displayObjList.RemoveAll();	m_displayObjList.AddTail(&tempList);
}

void GLK::DelDisplayObj3(GLKEntity *entity)
{
	float newRange,oldRange=m_Range;
	m_Range=1.0f;

	GLKPOSITION Pos;	GLKObList tempList;
	tempList.RemoveAll();
	for(Pos=m_displayObjList.GetHeadPosition();Pos!=NULL;)
	{
		GLKEntity *tempEntity=(GLKEntity *)(m_displayObjList.GetNext(Pos));
		if (tempEntity!=entity) tempList.AddTail(tempEntity);
	}
	m_displayObjList.RemoveAll();	m_displayObjList.AddTail(&tempList);
	
	bool flag=true;
	for(Pos=m_displayObjList.GetHeadPosition();Pos!=NULL;)
	{
		GLKEntity *tempEntity=(GLKEntity *)(m_displayObjList.GetNext(Pos));
		newRange=tempEntity->getRange();
		if ((newRange>m_Range) || (flag))
		{
			m_Range=newRange;
			flag=false;
		}
	}

	m_Scaling=m_Scaling*(m_Range/oldRange);
	refresh();	
}

int GLK::DisplayObjCount()
{
	return m_displayObjList.GetCount();
}

GLKEntity* GLK::GetDisplayObjAt(int nIndex)
{
	GLKPOSITION Pos; int n=1;

	for(Pos=m_displayObjList.GetHeadPosition();Pos!=NULL;n++)
	{
		GLKEntity *tempEntity=(GLKEntity *)(m_displayObjList.GetNext(Pos));
		if (n==nIndex) return tempEntity;
	}
	return NULL;
}

void GLK::ClearDisplayObjList()
{
	m_displayObjList.RemoveAll();
	m_Range=1.0f;
}

void GLK::draw_polyline_2d(int pointNum, const float pts[], bool bFill)
{
	int i;
	double xx,yy,zz;

    glLoadIdentity();

    setCamera();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	glClearColor(m_ClearColorRed,m_ClearColorGreen,m_ClearColorBlue,1.0f);

	GLDisableLight();

	////////////////////////////////////////////////////////////////
	//	The following lines are drawing axis.
	if (m_axisDisplay) GLDrawAxis();

	////////////////////////////////////////////////////////////////
	//	The following lines are drawing Object.
	//		Default rendering 
	GLDrawDisplayObjList();

	////////////////////////////////////////////////////////////////
	//	The following lines are drawing GLList.
	GLDrawGLList();

	glGetDoublev(GL_MODELVIEW_MATRIX, modelMatrix);
	glGetDoublev(GL_PROJECTION_MATRIX, projMatrix);
	glGetIntegerv(GL_VIEWPORT, viewport);

	GLDisableLight();

	if (bFill)
		glBegin(GL_LINE_LOOP);
	else
		glBegin(GL_LINE_STRIP);
	glColor3f(m_red,m_green,m_blue);
	glLineWidth(m_lineWidth);
	for(i=0;i<pointNum;i++) {
		screen_to_wcl(pts[i*2],pts[i*2+1],xx,yy,zz);
		glVertex3d(xx,yy,zz);
	}
	glEnd();

	////////////////////////////////////////////////////////////////
	//	The following lines are drawing coordinate.
	GLDrawCoordinate();

    glutSwapBuffers();
}

void GLK::screen_to_wcl(double sx, double sy, double &cx, double &cy, double &cz)
{
	GLdouble objx, objy, objz;
	double y = m_SizeY - sy;
	gluUnProject(sx, y, 0.5, modelMatrix, projMatrix, viewport, &objx, &objy, &objz); 

	cx=objx;	cy=objy;	cz=objz;
}

void GLK::wcl_to_screen(double cx, double cy, double cz, double &sx, double &sy)
{
	GLdouble winx, winy, winz;
	gluProject(cx, cy, cz, modelMatrix, projMatrix, viewport, &winx, &winy, &winz); 
	
	sx=winx;
	sy=m_SizeY-winy;
}

void GLK::Initialization()
{
	initValue();
}

void GLK::Reshape(int w, int h)
{
	m_SizeX=w;	m_SizeY=h;

	setViewport();
}

void GLK::setViewport()
{
	int cx=m_SizeX;
	int cy=m_SizeY;
	float scale=m_Scaling;

	if ((m_Range*scale)<0.5) scale=0.5/m_Range;
	
	glViewport(0,0,cx,cy);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
    if (cx <= cy)
	{
	    glOrtho (-m_Range, m_Range, -m_Range*(GLfloat)cy/(GLfloat)cx, 
			m_Range*(GLfloat)cy/(GLfloat)cx, 
			-m_Range*scale, m_Range*scale);
		m_MappingScale=cx/(m_Range*2.0);
	}
    else 
	{
		glOrtho (-m_Range*(GLfloat)cx/(GLfloat)cy, 
			m_Range*(GLfloat)cx/(GLfloat)cy, -m_Range, m_Range, 
			-m_Range*scale, m_Range*scale);
		m_MappingScale=cy/(m_Range*2.0);
	}

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}




void GLK::refresh()
{
	glPushMatrix();

	setViewport();

	glClearColor(m_ClearColorRed,m_ClearColorGreen,m_ClearColorBlue,1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	setCamera();
	
	

//#ifdef CLIPPING
	if (bClip)
	{
		   /* z < 0 */
		if (ClipEnable[0])
		{	
			glClipPlane (GL_CLIP_PLANE0, eqn[0]);
			glEnable (GL_CLIP_PLANE0);
		}
		if (ClipEnable[1])
		{
			glClipPlane (GL_CLIP_PLANE1, eqn[1]);
			glEnable (GL_CLIP_PLANE1);		
		}
		if (ClipEnable[2])
		{
			glClipPlane (GL_CLIP_PLANE2, eqn[2]);
			glEnable (GL_CLIP_PLANE2);
		}


		//	glPolygonMode(GL_BACK, GL_LINE);
		doDisplay();
		//	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glDisable (GL_CLIP_PLANE0);
		glDisable (GL_CLIP_PLANE1);
		glDisable (GL_CLIP_PLANE2);
	}
	else
	{
		doDisplay();
	}



	glPopMatrix();

    glutSwapBuffers();
	
}

void GLK::setCamera()
{
	// Position / translation / scale
	glTranslatef(m_xTranslation,m_yTranslation,m_zTranslation);
	glRotatef(m_xRotation,1.0f,0.0f,0.0f);
	glRotatef(m_yRotation,0.0f,1.0f,0.0f);
	glScalef(m_Scaling,m_Scaling,m_Scaling);
}

void GLK::doDisplay()
{
	GLDisableLight();

	////////////////////////////////////////////////////////////////
	//	The following lines are drawing axis.
	if (m_axisDisplay) GLDrawAxis();

	////////////////////////////////////////////////////////////////
	//	The following lines are drawing Object.
	//		Default rendering 
	GLDrawDisplayObjList();

	////////////////////////////////////////////////////////////////
	//	The following lines are drawing GLList.
	GLDrawGLList();

	glGetDoublev(GL_MODELVIEW_MATRIX, modelMatrix);
	glGetDoublev(GL_PROJECTION_MATRIX, projMatrix);
	glGetIntegerv(GL_VIEWPORT, viewport);


	GLDisableLight();


	////////////////////////////////////////////////////////////////
	//	The following lines are drawing coordinate.
	GLDrawCoordinate();
}

void GLK::initValue()
{
	m_xRotation = 0.0f;
	m_yRotation = 0.0f;

	m_xTranslation = 0.0f;
	m_yTranslation = 0.0f;
	m_zTranslation = 0.0f;

	m_Scaling = 1.0f;
	m_Range = 1.0f;

	m_ClearColorRed   = 1.0;
	m_ClearColorGreen = 1.0;
	m_ClearColorBlue  = 1.0;


	m_red = 0.0;
	m_green = 1.0;
	m_blue = 0.0;
	
	m_lineWidth = 1;

	m_axisDisplay = true;
}

void GLK::GLEnableLight()
{
	// Lights, material properties
	glShadeModel(GL_SMOOTH);
	glEnable(GL_NORMALIZE);
	
	glEnable(GL_DEPTH_TEST);
    glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
//    glColorMaterial(GL_FRONT, GL_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL); 

	GLfloat	ambientProperties[]  = {0.7f, 0.7f, 0.7f, 1.0f};
	GLfloat	diffuseProperties[]  = {0.8f, 0.8f, 0.8f, 1.0f};
	GLfloat	specularProperties[] = {1.0f, 1.0f, 1.0f, 1.0f};

	glLightfv( GL_LIGHT0, GL_AMBIENT, ambientProperties);
	glLightfv( GL_LIGHT0, GL_DIFFUSE, diffuseProperties);
	glLightfv( GL_LIGHT0, GL_SPECULAR, specularProperties);
#ifdef CLIPPING
	glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, 0.0);
#else
	glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, 1.0);
#endif

	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);
}

void GLK::GLDisableLight()
{
	glDisable(GL_LIGHT0);
	glDisable(GL_LIGHTING);
}

void GLK::GLDrawCoordinate()
{
	if (m_bCoordDisp) {
		GLDisableLight();
		glLineWidth(1.0);
		glColor3f(1.0-m_ClearColorRed,1.0-m_ClearColorGreen,1.0-m_ClearColorBlue);

		char text[256];
		char *p;

		sprintf(text,"(%.2f, %.2f, %.2f)",m_currentCoord[0],m_currentCoord[1],m_currentCoord[2]);

		glLoadIdentity();
		if (m_SizeX>m_SizeY)
			glTranslatef(-0.99*((double)m_SizeX/(double)m_SizeY), -0.98, 0);
		else
			glTranslatef(-0.99, -0.98*((double)m_SizeY/(double)m_SizeX), 0);
		glScalef(0.0003,0.0003,0.00032);

		for (p = text; *p; p++)
			glutStrokeCharacter(GLUT_STROKE_ROMAN, *p);
		
		glLineWidth(1.0);
	}
}

void GLK::GLDrawAxis()
{
	double axisLength=0.2*m_Range/m_Scaling;
	glColor3f(1.0,0.0,0.0);		//	x-axis
	glBegin(GL_LINES);
	glVertex3f(0.0,0.0,0.0);
	glVertex3f(axisLength,0.0,0.0);
	glEnd();
	glColor3f(0.0,1.0,0.0);		//	y-axis
	glBegin(GL_LINES);
	glVertex3f(0.0,0.0,0.0);
	glVertex3f(0.0,axisLength,0.0);
	glEnd();
	glColor3f(0.0,0.0,1.0);		//	z-axis
	glBegin(GL_LINES);
	glVertex3f(0.0,0.0,0.0);
	glVertex3f(0.0,0.0,axisLength);
	glEnd();
}

void GLK::GLDrawDisplayObjList()
{
	glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(0.5,0.5);
	glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);
	glEnable(GL_POLYGON_OFFSET_LINE);
	glPolygonOffset(1.0,1.0);

	GLKPOSITION Pos;
	if (m_Shading) {
		GLEnableLight();
		for(Pos=m_displayObjList.GetHeadPosition();Pos!=NULL;)
		{
			GLKEntity *entity=(GLKEntity *)(m_displayObjList.GetNext(Pos));
			if (!entity) continue;
			if (!(entity->bShow)) continue;
			if (entity->bUpdate) entity->Update();			
			entity->drawShade();
		}
		GLDisableLight();
	}
	else {
		if (m_Mesh)	{
//			glEnable(GL_DEPTH_TEST);
			for(Pos=m_displayObjList.GetHeadPosition();Pos!=NULL;)
			{
				GLKEntity *entity=(GLKEntity *)(m_displayObjList.GetNext(Pos));
				if (!entity) continue;
				if (!(entity->bShow)) continue;
				if (entity->bUpdate) entity->Update();
				entity->drawPreMesh();
			}
		}
	}

	if (m_Mesh)
	for(Pos=m_displayObjList.GetHeadPosition();Pos!=NULL;)
	{
		GLKEntity *entity=(GLKEntity *)(m_displayObjList.GetNext(Pos));
		if (!entity) continue;
		if (!(entity->bShow)) continue;
		if (entity->bUpdate) entity->Update();
		entity->drawMesh();
	}

	if (m_Profile)
	for(Pos=m_displayObjList.GetHeadPosition();Pos!=NULL;)
	{
		GLKEntity *entity=(GLKEntity *)(m_displayObjList.GetNext(Pos));
		if (!entity) continue;
		if (!(entity->bShow)) continue;
		if (entity->bUpdate) entity->Update();
		entity->drawProfile();
	}

	if (m_Contouring)
	for(Pos=m_displayObjList.GetHeadPosition();Pos!=NULL;)
	{
			GLKEntity *entity=(GLKEntity *)(m_displayObjList.GetNext(Pos));
			if (!entity) continue;
			if (!(entity->bShow)) continue;
			if (entity->bUpdate) entity->Update();
		
			entity->drawContour();
	}

	for(Pos=m_displayObjList.GetHeadPosition();Pos!=NULL;)
	{
		GLKEntity *entity=(GLKEntity *)(m_displayObjList.GetNext(Pos));
		if (!entity) continue;
		if (!(entity->bShow)) continue;
		if (entity->bUpdate) entity->Update();
		if (entity==m_HighLightObj) 
		{
			entity->drawHighLight();
			continue;
		}
	}
}

void GLK::GLDrawGLList()
{
	GLKPOSITION Pos;
	for(Pos=m_glList.GetHeadPosition();Pos!=NULL;)
	{
		GLKGLList *glList=(GLKGLList *)(m_glList.GetNext(Pos));
		if (!glList) continue;
		glList->draw(this);
	}
}

void GLK::TransClipPlane(short i, double v)
{
	clipTrans[i] += v;
	eqn[i][3] = clipTrans[i];
}




void GLK::SetViewDirection(short nDirID)
{
	switch(nDirID)
	{
	case 0:{	//VD_FRONTVIEW
				m_xRotation=0.0;	m_yRotation=0.0;
				refresh();
		   }break;
	case 1:{	//VD_LEFTVIEW
				m_xRotation=0.0;	m_yRotation=90.0;
				refresh();
		   }break;
	case 2:{	//VD_RIGHTVIEW
				m_xRotation=0.0;	m_yRotation=-90.0;
				refresh();
		   }break;
	case 3:{	//VD_BACKVIEW
				m_xRotation=0.0;	m_yRotation=180.0;
				refresh();
		   }break;
	case 4:{	//VD_TOPVIEW	
				m_xRotation=90.0;	m_yRotation=0.0;
				refresh();
		   }break;
	case 5:{	//VD_BOTTOMVIEW
				m_xRotation=-90.0;	m_yRotation=0.0;
				refresh();
		   }break;
	case 6:{	//VD_ISOMETRICVIEW
				m_xRotation=27.0;	m_yRotation=-45.0;
				refresh();
		   }break;
	case 7:{	//VD_BACKISOMETRICVIEW
				m_xRotation=27.0;	m_yRotation=135.0;
				refresh();
		   }break;
	}
}

void GLK::zoom(double ratio)
{
	m_Scaling*=ratio;
	if (m_Scaling<0.00001) m_Scaling=0.00001f;
	refresh();
}

void GLK::zoom_all_in_view()
{
	float newRange,oldRange=m_Range;
	m_Range=1.0f;

	GLKPOSITION Pos;	
	bool flag=true;
	for(Pos=m_displayObjList.GetHeadPosition();Pos!=NULL;)
	{
		GLKEntity *tempEntity=(GLKEntity *)(m_displayObjList.GetNext(Pos));
		newRange=tempEntity->getRange();
		
		if ((newRange>m_Range) || (flag))
		{
			m_Range=newRange;
			flag=false;
		}
	}

	m_Scaling=1.0;
	m_xTranslation = 0.0f;
	m_yTranslation = 0.0f;
	m_zTranslation = 0.0f;
	refresh();
}

void GLK::GetViewVector(double &x, double &y, double &z)
{
	GLKGeometry geo;
	double cx,cy,cz,d;
	double xx[3],yy[3],zz[3];

	screen_to_wcl(100,100,cx,cy,cz);
	xx[0]=cx;	yy[0]=cy;	zz[0]=cz;
	screen_to_wcl(200,200,cx,cy,cz);
	xx[1]=cx;	yy[1]=cy;	zz[1]=cz;
	screen_to_wcl(200,100,cx,cy,cz);
	xx[2]=cx;	yy[2]=cy;	zz[2]=cz;
	geo.CalPlaneEquation(x,y,z,d,xx,yy,zz);
}

void GLK::GetUpVector(double &x, double &y, double &z)
{
	GLKGeometry geo;
	double n[3],p1[3],p2[3],mu;
	GetViewVector(n[0],n[1],n[2]);

	screen_to_wcl(0,0,p1[0],p1[1],p1[2]);
	geo.CalPlaneLineIntersection(p1,n,n[0],n[1],n[2],0.0,mu);
	p1[0]=p1[0]+n[0]*mu;	p1[1]=p1[1]+n[1]*mu;	p1[2]=p1[2]+n[2]*mu;

	screen_to_wcl(0,-10,p2[0],p2[1],p2[2]);
	geo.CalPlaneLineIntersection(p2,n,n[0],n[1],n[2],0.0,mu);
	p2[0]=p2[0]+n[0]*mu;	p2[1]=p2[1]+n[1]*mu;	p2[2]=p2[2]+n[2]*mu;

	p2[0]=p2[0]-p1[0];	p2[1]=p2[1]-p1[1];	p2[2]=p2[2]-p1[2];
	geo.Normalize(p2);
	x=p2[0];	y=p2[1];	z=p2[2];
}

void GLK::GetScale(float &scale) 
{
	scale=m_Scaling;
}

void GLK::SetScale(float scale) 
{
	m_Scaling=scale;
}


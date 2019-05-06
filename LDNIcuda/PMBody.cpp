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


#include <stdio.h>
#include <string.h>
#include <math.h>

#include "../common/GL/glew.h"

#include "PMBody.h"
#include <thrust/unique.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>



PMBody::PMBody(void)
{
	m_drawShadingListID=-1;
	m_drawMeshListID=-1;
	bUpdate = false;
}

PMBody::~PMBody(void)
{
	ClearAll();
	if (m_drawShadingListID!=-1) {glDeleteLists(m_drawShadingListID, 1); m_drawShadingListID=-1;}
	if (m_drawMeshListID!=-1) {glDeleteLists(m_drawMeshListID, 1); m_drawMeshListID=-1;}
}

void PMBody::DeleteGLList(bool bShadeOrMesh)
{
	//if (m_drawListID!=-1) {glDeleteLists(m_drawListID, 2);	m_drawListID=-1;}
	if (bShadeOrMesh && m_drawShadingListID!=-1) {glDeleteLists(m_drawShadingListID, 1); m_drawShadingListID=-1;} 
	if (!bShadeOrMesh && m_drawMeshListID!=-1) {glDeleteLists(m_drawMeshListID, 1); m_drawMeshListID=-1;} 
}

void PMBody::BuildGLList(bool bShadeOrMesh, bool bContourList)
{
	DeleteGLList(bShadeOrMesh);

	if (bContourList)
	{	
		m_drawContourID = glGenLists(1);
		_buildContourList();
	}
	else
	{
		if (bShadeOrMesh) {
			m_drawShadingListID = glGenLists(1);	
			_buildDrawShadeList();			
		}
		if (!bShadeOrMesh) {
			m_drawMeshListID = glGenLists(1);
			_buildDrawMeshList();			
		}
	}
	

	
}


void PMBody::Update()
{
	DeleteGLList(true);
	DeleteGLList(false);
	BuildGLList(true);
	BuildGLList(false);
	SetPMUpdate(false);
}
void PMBody::drawShade()
{
	if (meshList.IsEmpty()) {DeleteGLList(true); DeleteGLList(false); return;}
	glCallList(m_drawShadingListID);
}

void PMBody::drawMesh()
{
	if (meshList.IsEmpty()) {DeleteGLList(true); DeleteGLList(false); return;}
	glCallList(m_drawMeshListID);
}

void PMBody::drawProfile() {}
void PMBody::drawPreMesh() {}
void PMBody::drawHighLight() {}

void PMBody::drawContour()
{
	if (meshList.IsEmpty()) {DeleteGLList(true); DeleteGLList(false); return;}

	glCallList(m_drawContourID);
}
	
void PMBody::drawBox(float xx, float yy, float zz, float r)
{
	glBegin(GL_QUADS);

	glNormal3f(0.0f,0.0f,-1.0f);
	glVertex3f(xx-r,yy-r,zz-r);
	glVertex3f(xx-r,yy+r,zz-r);
	glVertex3f(xx+r,yy+r,zz-r);
	glVertex3f(xx+r,yy-r,zz-r);

	glNormal3f(0.0f,0.0f,1.0f);
	glVertex3f(xx-r,yy-r,zz+r);
	glVertex3f(xx+r,yy-r,zz+r);
	glVertex3f(xx+r,yy+r,zz+r);
	glVertex3f(xx-r,yy+r,zz+r);
		
	glNormal3f(-1.0f,0.0f,0.0f);
	glVertex3f(xx-r,yy-r,zz-r);
	glVertex3f(xx-r,yy-r,zz+r);
	glVertex3f(xx-r,yy+r,zz+r);
	glVertex3f(xx-r,yy+r,zz-r);
		
	glNormal3f(1.0f,0.0f,0.0f);
	glVertex3f(xx+r,yy-r,zz-r);
	glVertex3f(xx+r,yy+r,zz-r);
	glVertex3f(xx+r,yy+r,zz+r);
	glVertex3f(xx+r,yy-r,zz+r);
		
	glNormal3f(0.0f,-1.0f,0.0f);
	glVertex3f(xx-r,yy-r,zz-r);
	glVertex3f(xx+r,yy-r,zz-r);
	glVertex3f(xx+r,yy-r,zz+r);
	glVertex3f(xx-r,yy-r,zz+r);
		
	glNormal3f(0.0f,1.0f,0.0f);
	glVertex3f(xx-r,yy+r,zz-r);
	glVertex3f(xx-r,yy+r,zz+r);
	glVertex3f(xx+r,yy+r,zz+r);
	glVertex3f(xx+r,yy+r,zz-r);

	glEnd();
}
	
void PMBody::ClearAll()
{
	GLKPOSITION Pos;

	for(Pos=meshList.GetHeadPosition();Pos!=NULL;) {
		QuadTrglMesh *mesh=(QuadTrglMesh *)(meshList.GetNext(Pos));
		delete mesh;
	}
	meshList.RemoveAll();
}

void PMBody::FlipModel(short nDir)
{
	GLKPOSITION Pos;

	for(Pos=meshList.GetHeadPosition();Pos!=NULL;) {
		QuadTrglMesh *mesh=(QuadTrglMesh *)(meshList.GetNext(Pos));
		switch(nDir)
		{
		case 0: mesh->FlipModel(true,false,false);
				break;
		case 1: mesh->FlipModel(false,true,false);
				break;
		case 2: mesh->FlipModel(false,false,true);
				break;

		}
	}
}

void PMBody::ShiftToPosSystem()
{
	GLKPOSITION Pos; int i,nodeNum;
	float pos[3];
	double xmin = 1.0e12;
	double ymin = 1.0e12;
	double zmin = 1.0e12;
	for(Pos=meshList.GetHeadPosition();Pos!=NULL;) {
		QuadTrglMesh *mesh=(QuadTrglMesh *)(meshList.GetNext(Pos));
		nodeNum=mesh->GetNodeNumber();
		for(i=0;i<nodeNum;i++) {
			mesh->GetNodePos(i+1,pos);
			if(xmin > pos[0]) xmin = pos[0];
			if(ymin > pos[1]) ymin = pos[1];
			if(zmin > pos[2]) zmin = pos[2];
		}
		for(i=0;i<nodeNum;i++) {
			mesh->GetNodePos(i+1,pos);
			pos[0] -= xmin;	pos[1] -= ymin;	pos[2] -= zmin;
			mesh->SetNodePos(i+1,pos);
		}
	}
}

void PMBody::ShiftToOrigin()
{
	GLKPOSITION Pos; int i,nodeNum;
	double centroid[3] = {0.0, 0.0, 0.0};
	float pos[3];

	for(Pos=meshList.GetHeadPosition();Pos!=NULL;) {
		QuadTrglMesh *mesh=(QuadTrglMesh *)(meshList.GetNext(Pos));
		nodeNum=mesh->GetNodeNumber();
		centroid[0] = 0.0;	centroid[1] = 0.0;	centroid[2] = 0.0;
		for(i=0;i<nodeNum;i++) {
			mesh->GetNodePos(i+1,pos);
			centroid[0] += pos[0];	centroid[1] += pos[1];	centroid[2] += pos[2];
		}
		centroid[0] /= (double)nodeNum;	centroid[1] /= (double)nodeNum;	centroid[2] /= (double)nodeNum;
		for(i=0;i<nodeNum;i++) {
		mesh->GetNodePos(i+1,pos);
		pos[0] -= centroid[0];	pos[1] -= centroid[1];	pos[2] -= centroid[2];
		mesh->SetNodePos(i+1,pos);
		}
	}
}

	
void PMBody::CompBoundingBox(float boundingBox[])
{
	GLKPOSITION PosMesh;	
	float pos[3];	int i,nodeNum;

	boundingBox[0]=boundingBox[2]=boundingBox[4]=1.0e+16;	
	boundingBox[1]=boundingBox[3]=boundingBox[5]=-1.0e+16;	

	for(PosMesh=meshList.GetHeadPosition();PosMesh!=NULL;) {
		QuadTrglMesh *mesh=(QuadTrglMesh *)(meshList.GetNext(PosMesh));
		nodeNum=mesh->GetNodeNumber();
		for(i=0;i<nodeNum;i++) {
			mesh->GetNodePos(i+1,pos);
			if (pos[0]<boundingBox[0]) boundingBox[0]=pos[0];
			if (pos[0]>boundingBox[1]) boundingBox[1]=pos[0];
			if (pos[1]<boundingBox[2]) boundingBox[2]=pos[1];
			if (pos[1]>boundingBox[3]) boundingBox[3]=pos[1];
			if (pos[2]<boundingBox[4]) boundingBox[4]=pos[2];
			if (pos[2]>boundingBox[5]) boundingBox[5]=pos[2];
		}
	}
}

void PMBody::Transformation(float dx, float dy, float dz)
{
	float pos[3];	int i,nodeNum,j;
	GLKPOSITION Pos; j=0;

	
	for(Pos=meshList.GetHeadPosition();Pos!=NULL;j++) {
		QuadTrglMesh *mesh=(QuadTrglMesh *)(meshList.GetNext(Pos));
		
		//if ((mesh->GetMeshId() == 1 && j == meshList.GetCount()-1) || mesh->GetMeshUpdateStatus())
		if (mesh->GetMeshId() == meshList.GetCount())
		{
			nodeNum=mesh->GetNodeNumber();
			for(i=0;i<nodeNum;i++) {
				mesh->GetNodePos(i+1,pos);
				pos[0]+=dx;	pos[1]+=dy;	pos[2]+=dz;
				mesh->SetNodePos(i+1,pos);
			}
			SetPMUpdate(true);
			computeRange();
		}
		
	}
	//_buildDrawShadeList();
	
}

void PMBody::Scaling(float sx, float sy, float sz)
{
	float pos[3];	int i,nodeNum,j;
	GLKPOSITION Pos;
	j=0;
	for(Pos=meshList.GetHeadPosition();Pos!=NULL;j++) {
		QuadTrglMesh *mesh=(QuadTrglMesh *)(meshList.GetNext(Pos));
		
		if (mesh->GetMeshId() == meshList.GetCount())
		{
			
			nodeNum=mesh->GetNodeNumber();
			for(i=0;i<nodeNum;i++) {
				mesh->GetNodePos(i+1,pos);
				pos[0]*=sx;	pos[1]*=sy;	pos[2]*=sz;
				mesh->SetNodePos(i+1,pos);
			}
			SetPMUpdate(true);		//_buildDrawShadeList();			
			computeRange();
		}
	}
	
}

void PMBody::_buildDrawShadeList()
{
	GLKPOSITION Pos;
	float pos[3],nv[3],rgb[3];		int i,meshIndex,faceNum;
	unsigned int ver[4];
	float *vertexArray,*normalArray;

	glNewList(m_drawShadingListID, GL_COMPILE);
	glEnable(GL_NORMALIZE);
	glEnable(GL_LIGHTING);
	glBegin(GL_TRIANGLES);
	meshIndex=0;
	for(Pos=meshList.GetHeadPosition();Pos!=NULL;meshIndex++) {
		QuadTrglMesh *mesh=(QuadTrglMesh *)(meshList.GetNext(Pos));

		
		faceNum=mesh->GetFaceNumber();

		_changeValueToColor(mesh->GetMeshId(),rgb[0],rgb[1],rgb[2]);

		glColor3fv(rgb);

		for(i=0;i<faceNum;i++) {
			mesh->GetFaceNodes(i+1,ver[0],ver[1],ver[2],ver[3]);
			
			
			mesh->CompNormal(i+1,nv);

			if (mesh->IsQuadFace(i+1)) glColor3f(1,1,0);

			glNormal3fv(nv);
			mesh->GetNodePos(ver[0],pos);
			glVertex3fv(pos);
			mesh->GetNodePos(ver[1],pos);
			glVertex3fv(pos);
			mesh->GetNodePos(ver[2],pos);
			glVertex3fv(pos);

			if (!(mesh->IsQuadFace(i+1))) continue;

			glNormal3fv(nv);
			mesh->GetNodePos(ver[0],pos);
			glVertex3fv(pos);
			mesh->GetNodePos(ver[2],pos);
			glVertex3fv(pos);
			mesh->GetNodePos(ver[3],pos);
			glVertex3fv(pos);

			glColor3fv(rgb);
		}

	}
	glEnd();
	glEndList();
}


void PMBody::_buildContourList()
{
	if (meshList.GetCount()==0) return;
	glNewList(m_drawContourID, GL_COMPILE);
	glDisable(GL_LIGHTING);
	glLineWidth(2.0);

	glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);
	
	glEnable(GL_POLYGON_OFFSET_LINE);
	glPolygonOffset(1.0,1.0);


	GLKPOSITION Pos;
	int i, num, j;
	float pos1[3], pos2[3], rgb[3];
	float *s_ptr, *e_ptr;
	
	j=1;
	glBegin(GL_LINES);	
	for(Pos=meshList.GetHeadPosition();Pos!=NULL;j++) {

		ContourMesh *cmesh=(ContourMesh *)(meshList.GetNext(Pos));
		_changeValueToColor(j,rgb[0],rgb[1],rgb[2]);
		glColor3f(rgb[0],rgb[1],rgb[2]);
		if (cmesh->GetContourNum()>0)
		{			
			GLKPOSITION Pos2;
			double xx,yy,zz;
			for (Pos=cmesh->GetVSAMeshList().GetHeadPosition();Pos!=NULL;)
			{
				
				VSAMesh *mesh=(VSAMesh *)(cmesh->GetVSAMeshList().GetNext(Pos));
				for (Pos2=mesh->GetVSAEdgeList().GetHeadPosition(); Pos2!=NULL;)
				{
					VSAEdge *edge=(VSAEdge *)(mesh->GetVSAEdgeList().GetNext(Pos2));
					edge->GetStartPoint()->GetCoord3D(xx, yy, zz);
					glVertex3d(xx,yy,zz);

			

					edge->GetEndPoint()->GetCoord3D(xx, yy, zz);
					glVertex3d(xx,yy,zz);

					
				}


			}
			
		}
		else if (cmesh->GetContourNumPtr()!= NULL)
		{			
			num = cmesh->GetTotalStickNum();
			s_ptr = cmesh->GetStartNodeArrayPtr();
			e_ptr = cmesh->GetEndNodeArrayPtr();
			
			for(i=0; i < num; i++)
			{
				pos1[0] = s_ptr[i*3];
				pos1[1] = s_ptr[i*3+1];
				pos1[2] = s_ptr[i*3+2];

				
				glVertex3fv(pos1);

				pos2[0] = e_ptr[i*3];
				pos2[1] = e_ptr[i*3+1];
				pos2[2] = e_ptr[i*3+2];

				
				glVertex3fv(pos2);
			}
			
		}

		
	}

	glEnd();

	
	glEnable(GL_TEXTURE_2D);
	
	
	for(Pos=meshList.GetHeadPosition();Pos!=NULL;) {

		ContourMesh *cmesh=(ContourMesh *)(meshList.GetNext(Pos));
		
		
		if (cmesh->m_drawImage)
		{
			pos1[0] = cmesh->GetThickness();
			for (i=0; i < cmesh->GetResolution(1); i++) 
			{
				glBindTexture(GL_TEXTURE_2D, cmesh->tex[i]);

				glBegin(GL_QUADS);

				glTexCoord2f(1.0f,1.0f); glVertex3f(0.0f, i*pos1[0], 1.0f);
				glTexCoord2f(1.0f,0.0f); glVertex3f( 1.0f,i*pos1[0], 1.0f);
				glTexCoord2f(0.0f,0.0f); glVertex3f( 1.0f,i*pos1[0],0.0f);
				glTexCoord2f(0.0f,1.0f); glVertex3f(0.0f, i*pos1[0],0.0f);


				glEnd();
				glBindTexture(GL_TEXTURE_2D,0);
			}
			
		}


	}

	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);


	glEndList();

}


void PMBody::_buildDrawMeshList()
{
	if (meshList.GetCount()==0) return;

	glNewList(m_drawMeshListID, GL_COMPILE);
	glDisable(GL_LIGHTING);
	glLineWidth(1.0);

	glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);
	glColor3f(0.0,0.0,0.0);
	glEnable(GL_POLYGON_OFFSET_LINE);
	glPolygonOffset(1.0,1.0);

	GLKPOSITION Pos;
	float pos[3];	int i,j,edgeNum,faceNum;
	unsigned int ver[4];

	glBegin(GL_LINES);
	for(Pos=meshList.GetHeadPosition();Pos!=NULL;) {
		QuadTrglMesh *mesh=(QuadTrglMesh *)(meshList.GetNext(Pos));
		faceNum=mesh->GetFaceNumber();

		for(i=0;i<faceNum;i++) {
			mesh->GetFaceNodes(i+1,ver[0],ver[1],ver[2],ver[3]);
			if (mesh->IsQuadFace(i+1)) edgeNum=4; else edgeNum=3;
			for(j=0;j<edgeNum;j++) {
				if (ver[j]>ver[(j+1)%edgeNum]) continue;	// to avoid the duplicate of drawing
				mesh->GetNodePos(ver[j],pos);
				glVertex3fv(pos);
				mesh->GetNodePos(ver[(j+1)%edgeNum],pos);
				glVertex3fv(pos);
			}
		}
	}
	glEnd();

	glEndList();
}


void PMBody::_changeValueToColor(int nType, float & nRed, float & nGreen, float & nBlue)
{
	float color[][3]={
		{255,255,255},
		{242,190,13},
		{255,0,128},
		{0,255,255},
		{128,255,0},
		{128,128,64},
		{255,0,0},{0,255,0},{0,0,255},
		{128,128,192},
		{255,255,128},
		{255,128,0},
		{255,128,255},
		{255,214,202},
		{128,128,192},
		{255,165,0}, //orange
		{255,128,192},
		{128,128,64},
		{0,255,255},
		{238,130,238},//violet
		{220,220,220},//gainsboro
		{188, 143, 143}, // rosy brown 
		{46, 139, 87},//sea green
		{210, 105, 30 },//chocolate
		{100, 149, 237}//cornflower blue 
	};

	nRed=color[nType%23][0]/255.0f;
	nGreen=color[nType%23][1]/255.0f;
	nBlue=color[nType%23][2]/255.0f;
}
	
void PMBody::computeRange()
{
	double range=0.0,ll;	float pos[3];	int i,nodeNum;
	GLKPOSITION Pos;

	for(Pos=meshList.GetHeadPosition();Pos!=NULL;) {
		QuadTrglMesh *mesh=(QuadTrglMesh *)(meshList.GetNext(Pos));
		nodeNum=mesh->GetNodeNumber();
		for(i=0;i<nodeNum;i++) {
			mesh->GetNodePos(i+1,pos);
			ll=pos[0]*pos[0]+pos[1]*pos[1]+pos[2]*pos[2];

			if (ll>range) range=ll;
		}
	}

	m_range=(float)(sqrt(range));
}

QuadTrglMesh::QuadTrglMesh(void)
{
	m_nodeNum = m_faceNum = 0;
	bUpdate = false;
	meshID = 0;
}

QuadTrglMesh::~QuadTrglMesh(void)
{
	ClearAll();
}

void QuadTrglMesh::ClearAll()
{
	if (m_nodeNum!=0) free(m_nodeTable);
	if (m_faceNum!=0) free(m_faceTable);
	bUpdate = false;
}
	
void QuadTrglMesh::MallocMemory(int nodeNum, int faceNum)
{
	ClearAll();

	m_nodeNum=nodeNum;	
	if (nodeNum>0) m_nodeTable=(float*)malloc(m_nodeNum*3*sizeof(float));
	
	m_faceNum=faceNum;	
	if (faceNum>0) m_faceTable=(unsigned int*)malloc(m_faceNum*4*sizeof(unsigned int));
}

void QuadTrglMesh::SetNodePos(int index/*starting from 1*/, float pos[])
{
	m_nodeTable[(index-1)*3]=pos[0];
	m_nodeTable[(index-1)*3+1]=pos[1];
	m_nodeTable[(index-1)*3+2]=pos[2];
}

void QuadTrglMesh::SetFaceNodes(int index/*starting from 1*/, 
								unsigned int verIndex1, unsigned int verIndex2, 
								unsigned int verIndex3, unsigned int verIndex4)
{
	m_faceTable[(index-1)*4]=verIndex1;
	m_faceTable[(index-1)*4+1]=verIndex2;
	m_faceTable[(index-1)*4+2]=verIndex3;
	m_faceTable[(index-1)*4+3]=verIndex4;
}

int QuadTrglMesh::GetFaceNumber()
{
	return m_faceNum;
}

int QuadTrglMesh::GetNodeNumber()
{
	return m_nodeNum;
}

bool QuadTrglMesh::IsQuadFace(int index/*starting from 1*/)
{
	return (m_faceTable[(index-1)*4+3]!=0);
}

void QuadTrglMesh::GetFaceNodes(int index/*starting from 1*/,
								unsigned int &verIndex1, unsigned int &verIndex2, 
								unsigned int &verIndex3, unsigned int &verIndex4)
{
	verIndex1=m_faceTable[(index-1)*4];
	verIndex2=m_faceTable[(index-1)*4+1];
	verIndex3=m_faceTable[(index-1)*4+2];
	verIndex4=m_faceTable[(index-1)*4+3];
}

void QuadTrglMesh::GetNodePos(int index/*starting from 1*/, float pos[])
{
	pos[0]=m_nodeTable[(index-1)*3];
	pos[1]=m_nodeTable[(index-1)*3+1];
	pos[2]=m_nodeTable[(index-1)*3+2];
}

void QuadTrglMesh::CompBoundingBox(float boundingBox[])
{
	float pos[3];	int i,nodeNum;

	boundingBox[0]=boundingBox[2]=boundingBox[4]=1.0e+16;	
	boundingBox[1]=boundingBox[3]=boundingBox[5]=-1.0e+16;	

	nodeNum=GetNodeNumber();
	for(i=0;i<nodeNum;i++) {
		GetNodePos(i+1,pos);
		if (pos[0]<boundingBox[0]) boundingBox[0]=pos[0];
		if (pos[0]>boundingBox[1]) boundingBox[1]=pos[0];
		if (pos[1]<boundingBox[2]) boundingBox[2]=pos[1];
		if (pos[1]>boundingBox[3]) boundingBox[3]=pos[1];
		if (pos[2]<boundingBox[4]) boundingBox[4]=pos[2];
		if (pos[2]>boundingBox[5]) boundingBox[5]=pos[2];
	}
}

void QuadTrglMesh::CompNormal(int faceIndex/*starting from 1*/, float nv[])
{
	GLKGeometry geo;
	float p[3][3];	double nnv[3],aa,bb,cc,dd;
	int i,edgeNum;	unsigned int ver[4];

	if (IsQuadFace(faceIndex)) edgeNum=4; else edgeNum=3;
	GetFaceNodes(faceIndex,ver[0],ver[1],ver[2],ver[3]);
	nnv[0]=nnv[1]=nnv[2]=0.0;

	for(i=0;i<edgeNum;i++) {
		GetNodePos(ver[i],p[0]);
		GetNodePos(ver[(i+1)%edgeNum],p[1]);
		GetNodePos(ver[(i+2)%edgeNum],p[2]);
		geo.CalPlaneEquation(p[0],p[1],p[2],aa,bb,cc,dd);
		nnv[0]+=aa;		nnv[1]+=bb;		nnv[2]+=cc;
	}
	dd=nnv[0]*nnv[0]+nnv[1]*nnv[1]+nnv[2]*nnv[2];	if (dd<1.0e-10) dd=1.0;
	dd=sqrt(dd);
	nv[0]=(float)(nnv[0]/dd);
	nv[1]=(float)(nnv[1]/dd);
	nv[2]=(float)(nnv[2]/dd);
}

bool QuadTrglMesh::OutputOBJFile(char *filename)
{
	FILE *fp;

	fp = fopen(filename, "w");
    if(!fp)
	{
		printf("===============================================\n");
	    printf("Can not open the data file - OBJ File Export!\n");
		printf("===============================================\n");
	    return false;
	}

	fprintf(fp,"# The units used in this file are meters.\n");
	for(int i=0;i<m_nodeNum;i++) {
		fprintf(fp,"v %.12f %.12f %.12f\n", 
			m_nodeTable[i*3], m_nodeTable[i*3+1], m_nodeTable[i*3+2]);
	}

	fprintf(fp,"\n");

	for(int i=0;i<m_faceNum;i++) {
		if (IsQuadFace(i+1)) {
			fprintf(fp,"f %u %u %u %u\n", 
				m_faceTable[i*4],m_faceTable[i*4+1],m_faceTable[i*4+2],m_faceTable[i*4+3]);
		}
		else {
			fprintf(fp,"f %u %u %u\n", 
				m_faceTable[i*4],m_faceTable[i*4+1],m_faceTable[i*4+2]);
		}
	}

	fclose(fp);

	return true;
}

bool QuadTrglMesh::InputMEBFile(char *filename)
{
	FILE *fp;	int nodeNum, faceNum;

	fp = fopen(filename, "rb");
    if(!fp) {
		printf("===============================================\n");
	    printf("Can not open the data file - MEB File Export!\n");
		printf("===============================================\n");
	    return false;
	}

	fread(&nodeNum,sizeof(int),1,fp);
	fread(&faceNum,sizeof(int),1,fp);
	if (nodeNum<=0 || faceNum<=0) {
		printf("===============================================\n");
		printf("MEB File Import ERROR:  nodeNum=%u  faceNum=%u!\n",nodeNum,faceNum);
		printf("===============================================\n");
		fclose(fp);		return false;
	}

	//---------------------------------------------------------------
	MallocMemory(nodeNum,faceNum);
	fread(m_nodeTable,sizeof(float),m_nodeNum*3,fp);
	fread(m_faceTable,sizeof(unsigned int),m_faceNum*4,fp);

	fclose(fp);

	printf("-----------------------------------------------------\n");
	printf("Face number: %d\n",m_faceNum);
	printf("Node number: %d\n",m_nodeNum);

	return true;
}

bool QuadTrglMesh::OutputMEBFile(char *filename)
{
	FILE *fp;

	fp = fopen(filename, "wb");
    if(!fp) {
		printf("===============================================\n");
	    printf("Can not open the data file - MEB File Export!\n");
		printf("===============================================\n");
	    return false;
	}

	fwrite(&m_nodeNum,sizeof(int),1,fp);
	fwrite(&m_faceNum,sizeof(int),1,fp);

	fwrite(m_nodeTable,sizeof(float),m_nodeNum*3,fp);
	fwrite(m_faceTable,sizeof(unsigned int),m_faceNum*4,fp);

	fclose(fp);

	return true;
}


bool QuadTrglMesh::InputSTLFileFromMapping(char *filedata)
{
	char *getline;	
	char getline2[256];	
	char buf[100];
	char fields[4][255];
	unsigned int ver[4];
	float pos[3];
	int i=0,j=0,nodeNum=0,faceNum=0;
	int nodeIndex=1,faceIndex=1;

    

	getline = strtok (filedata,"\n");
	
	while (getline != NULL)
	{
		sscanf(getline,"%s",buf);
		
		if ((strlen(buf)==2) && (buf[0]=='v')) {
			sscanf(getline,"%s %d",buf,&nodeNum); 
			faceNum = nodeNum/3;
		}
		
		if (nodeNum>0 && j==0) {MallocMemory(nodeNum,faceNum); j++;}
		if (j>0){
			if ((strlen(buf)==6) && (buf[0]=='v')) //facet
			{
				sscanf(getline, "%s %f %f %f \n", buf, &(pos[0]), &(pos[1]), &(pos[2]));			
				SetNodePos(nodeIndex,pos);	nodeIndex++;
			}
		}
		getline = strtok (NULL,"\n");
	}

	for (int i = 0; i < faceNum; i++)
		SetFaceNodes(i+1,3*i+1,3*i+2,3*i+3,0);

	
	printf("-----------------------------------------------------\n");
	printf("Face number: %d\n",m_faceNum);
	printf("Node number: %d\n",m_nodeNum);
	return true;
}

bool QuadTrglMesh::InputOBJFileFromMapping(char *filedata)
{
	char *getline;	
	char getline2[256];	
	char buf[100];
	char fields[4][255];
	unsigned int ver[4];
	float pos[3];
	int i=0,j=0,nodeNum=0,faceNum=0;
	int nodeIndex=1,faceIndex=1;

    int count = 0;
	
	
	getline = strtok (filedata,"\n");

	while (getline != NULL)
	{
		sscanf(getline,"%s",buf);
		//count++;
		if ((strlen(buf)==2) && (buf[0]=='v')) sscanf(getline,"%s %d",buf,&nodeNum);
		if ((strlen(buf)==2) && (buf[0]=='f')) sscanf(getline,"%s %d",buf,&faceNum);

		
		if (nodeNum>0 && faceNum>0 && j==0) {MallocMemory(nodeNum,faceNum); j++;}
		strcpy(getline2,getline);
		getline = strtok (getline+(strlen(getline2))+1, "\n");
		if (j>0)
		{
			
			if ((strlen(buf)==1) && (buf[0]=='v') ) {
				sscanf(getline2, "%s %f %f %f \n", buf, &(pos[0]), &(pos[1]), &(pos[2]));
				SetNodePos(nodeIndex,pos);	nodeIndex++;
			}	
			if ((strlen(buf)==1) && (buf[0]=='f')) {
			char seps[]=" \n";
			char seps2[]="/";
			char *token;
			char linebuf2[255];

			strcpy(linebuf2,getline2);

			int num=0;
			token = strtok( getline2, seps );
			while(token!=NULL) {token=strtok(NULL,seps); num++;}
			num=num-1;

			if (num<3) continue;

			if (num>4) {
				
			}
			else {
				token=strtok(linebuf2,seps);	
				for(i=0;i<num;i++) {token=strtok(NULL,seps); strcpy(fields[i],token);}
				for(i=0;i<num;i++) {token = strtok( fields[i], seps2 ); ver[i]=(unsigned int)(atoi(token));}
				if (num==3) ver[3]=0;

				SetFaceNodes(faceIndex,ver[0],ver[1],ver[2],ver[3]);
               // fprintf(fp,"f %d %d %d \n",ver[0], ver[1], ver[2]);

				faceIndex++;
				}
			}	
		}

		
	}
    if (nodeNum==0 || faceNum==0) return false;
	//fclose(fp);
	printf("-----------------------------------------------------\n");
	printf("Face number: %d %d\n",m_faceNum, faceIndex);
	printf("Node number: %d %d\n",m_nodeNum, nodeIndex);
	
	return true;
}

bool QuadTrglMesh::InputSTLFile(char *filename)
{
	FILE *fp;
	unsigned int ver[4];
	char linebuf[256],buf[100];
	float pos[3];
	int i,nodeNum,faceNum;
	int nodeIndex=1,faceIndex=1;

	fp = fopen(filename, "r");
    if(!fp) {
	    printf("===============================================\n");
	    printf("Can not open the data file - STL File Import!\n");
	    printf("===============================================\n");
	    return false;
	}

	nodeNum=faceNum=0;
	while(!feof(fp)) {

		sprintf(buf,"");
		sprintf(linebuf,"");
		fgets(linebuf, 255, fp);
		sscanf(linebuf,"%s",buf);

		if ((strlen(buf)==6) && (buf[0]=='v')) 	nodeNum++;
	}
	fclose(fp);
	if (nodeNum==0) return false;

	faceNum = nodeNum/3;
	
	MallocMemory(nodeNum,faceNum);
	

	fp = fopen(filename, "r");	
	
	while(!feof(fp)) {

		sprintf(buf,"");
		sprintf(linebuf,"");
		fgets(linebuf, 255, fp);
		sscanf(linebuf,"%s",buf);

		if ((strlen(buf)==6) && (buf[0]=='v')) //vertex
		{
			nodeNum++;
			sscanf(linebuf, "%s %f %f %f \n", buf, &(pos[0]), &(pos[1]), &(pos[2]));
			SetNodePos(nodeIndex,pos);	nodeIndex++;
			
		}
	}
	fclose(fp);
	for (int i = 0; i < faceNum; i++)
		SetFaceNodes(i+1,3*i+1,3*i+2,3*i+3,0);

	
	printf("-----------------------------------------------------\n");
	printf("Face number: %d\n",m_faceNum);
	printf("Node number: %d\n",m_nodeNum);
	return true;
}

bool QuadTrglMesh::InputOBJFile(char *filename)
{
	FILE *fp;
	char fields[4][255];
	unsigned int ver[4];
	char linebuf[256],buf[100];
	float pos[3];
	int i,nodeNum,faceNum;

	fp = fopen(filename, "r");
    if(!fp) {
	    printf("===============================================\n");
	    printf("Can not open the data file - OBJ File Import!\n");
	    printf("===============================================\n");
	    return false;
	}

	//---------------------------------------------------------------
	//	Analysis of OBJ file
	nodeNum=faceNum=0;
	while(!feof(fp)) {
		sprintf(buf,"");
		sprintf(linebuf,"");
		fgets(linebuf, 255, fp);
		sscanf(linebuf,"%s",buf);
	
		if ( (strlen(buf)==1) && (buf[0]=='v') ) nodeNum++;
		if ( (strlen(buf)==1) && (buf[0]=='f') ) {
			char seps[]=" \n";
			char seps2[]="/";
			char *token;
			char linebuf2[255];

			strcpy(linebuf2,linebuf);

			int num=0;
			token = strtok( linebuf, seps );
			while(token!=NULL) {token=strtok(NULL,seps); num++;}
			num=num-1;

			if (num==3 || num==4) {
				faceNum++;
			}
			else if (num>4) {
				fclose(fp);
				printf("Warning: face with more than 4 sides is found, which cannot be supported by this program!\n");
				return false;	// cannot support mesh with more than 4 sides
			}
		}
	}
	fclose(fp);
	if (nodeNum==0 || faceNum==0) return false;

	//---------------------------------------------------------------
	//	Import of OBJ file
	MallocMemory(nodeNum,faceNum);
	//---------------------------------------------------------------
	fp = fopen(filename, "r");	
	int nodeIndex=1,faceIndex=1;
	while(!feof(fp)) {
		sprintf(buf,"");
		sprintf(linebuf,"");
		fgets(linebuf, 255, fp);
		sscanf(linebuf,"%s",buf);
	
		if ( (strlen(buf)==1) && (buf[0]=='v') ) {
			sscanf(linebuf, "%s %f %f %f \n", buf, &(pos[0]), &(pos[1]), &(pos[2]));
			SetNodePos(nodeIndex,pos);	nodeIndex++;
		}	
		if ( (strlen(buf)==1) && (buf[0]=='f') ) {
			char seps[]=" \n";
			char seps2[]="/";
			char *token;
			char linebuf2[255];

			strcpy(linebuf2,linebuf);

			int num=0;
			token = strtok( linebuf, seps );
			while(token!=NULL) {token=strtok(NULL,seps); num++;}
			num=num-1;

			if (num<3) continue;

			if (num>4) {
				
			}
			else {
				token=strtok(linebuf2,seps);	
				for(i=0;i<num;i++) {token=strtok(NULL,seps); strcpy(fields[i],token);}
				for(i=0;i<num;i++) {token = strtok( fields[i], seps2 ); ver[i]=(unsigned int)(atoi(token));}
				if (num==3) ver[3]=0;

				SetFaceNodes(faceIndex,ver[0],ver[1],ver[2],ver[3]);
				faceIndex++;
			}
		}	
	}
	fclose(fp);

	printf("-----------------------------------------------------\n");
	printf("Face number: %d\n",m_faceNum);
	printf("Node number: %d\n",m_nodeNum);

	return true;
}

void QuadTrglMesh::Transformation(float dx, float dy, float dz)
{
	float pos[3];	int i,nodeNum;
	

		nodeNum=GetNodeNumber();
		for(i=0;i<nodeNum;i++) {
			GetNodePos(i+1,pos);
			pos[0]+=dx;	pos[1]+=dy;	pos[2]+=dz;
			SetNodePos(i+1,pos);
		}
	
	
}

void QuadTrglMesh::Scaling(float sx, float sy, float sz)
{
	float pos[3];	int i,nodeNum;
	
	nodeNum=GetNodeNumber();
	for(i=0;i<nodeNum;i++) {
		GetNodePos(i+1,pos);
		pos[0]*=sx;	pos[1]*=sy;	pos[2]*=sz;
		SetNodePos(i+1,pos);
	}
	
	
}

void QuadTrglMesh::ShiftToPosSystem()
{
	double cmin[3];
	int i,nodeNum;
	float pos[3];

	nodeNum=GetNodeNumber();
	cmin[0] = 1.0e12;	cmin[1] = 1.0e12;	cmin[2] = 1.0e12;
	for(i=0;i<nodeNum;i++) {
		GetNodePos(i+1,pos);
		if(cmin[0] > pos[0]) cmin[0] = pos[0];
		if(cmin[1] > pos[1]) cmin[1] = pos[1];
		if(cmin[2] > pos[2]) cmin[2] = pos[2];
	}
	
	for(i=0;i<nodeNum;i++) {
		GetNodePos(i+1,pos);
		pos[0] -= cmin[0];	pos[1] -= cmin[1];	pos[2] -= cmin[2];
		SetNodePos(i+1,pos);
	}
	
}

void QuadTrglMesh::ShiftToOrigin()
{
	double centroid[3];
	int i,nodeNum;
	float pos[3];

	nodeNum=GetNodeNumber();
	centroid[0] = 0.0;	centroid[1] = 0.0;	centroid[2] = 0.0;
	for(i=0;i<nodeNum;i++) {
		GetNodePos(i+1,pos);
		centroid[0] += pos[0];	centroid[1] += pos[1];	centroid[2] += pos[2];
	}
	centroid[0] /= (double)nodeNum;	centroid[1] /= (double)nodeNum;	centroid[2] /= (double)nodeNum;
	for(i=0;i<nodeNum;i++) {
		GetNodePos(i+1,pos);
		pos[0] -= centroid[0];	pos[1] -= centroid[1];	pos[2] -= centroid[2];
		SetNodePos(i+1,pos);
	}
	
}

void QuadTrglMesh::FlipModel(bool nDir_X, bool nDir_Y, bool nDir_Z)
{
	float bndBox[6],cx,cy,cz;
	short nFlipTime;
	float pos[3];	int i,nodeNum,faceNum;
	unsigned int ver[4];

	CompBoundingBox(bndBox);
	cx=(bndBox[0]+bndBox[1])*0.5;
	cy=(bndBox[2]+bndBox[3])*0.5;
	cz=(bndBox[4]+bndBox[5])*0.5;

	nFlipTime=0;
	if (nDir_X) nFlipTime++;
	if (nDir_Y) nFlipTime++;
	if (nDir_Z) nFlipTime++;

	nodeNum=GetNodeNumber();
	for(i=0;i<nodeNum;i++) {
		GetNodePos(i+1,pos);
		if (nDir_X) pos[0]=cx+(cx-pos[0]);
		if (nDir_Y) pos[1]=cy+(cy-pos[1]);
		if (nDir_Z) pos[2]=cz+(cz-pos[2]);
		SetNodePos(i+1,pos);
	}
	faceNum=GetFaceNumber();
	for(i=0;i<faceNum;i++) {
		if (nFlipTime%2==1) {
			GetFaceNodes(i+1,ver[0],ver[1],ver[2],ver[3]);
			SetFaceNodes(i+1,ver[2],ver[1],ver[0],ver[3]);
		}
	}

	

}



ContourMesh::ContourMesh(void)
{
	m_ContourNum = NULL;
	bUpdate = false;
	meshID = 0;
	iRes[0] = 0;
	iRes[1] = 0;
	iRes[2] = 0;
	VSAMeshNum = 0;
	m_range = 1.0;
	m_drawImage = false;
}

ContourMesh::~ContourMesh(void)
{
	ClearAll();
}

void ContourMesh::ClearAll()
{
	if (m_ContourNum!=NULL) 
	{
		free(m_StnodeTable);
		free(m_EdnodeTable);
		free(m_ContourNum);
	}
	

	GLKPOSITION Pos;

	for(Pos=VSAMeshList.GetHeadPosition();Pos!=NULL;)
	{
		VSAMesh *temp=
			(VSAMesh *)(VSAMeshList.GetNext(Pos));
		delete temp;
	}
	VSAMeshList.RemoveAll();
	bUpdate = false;


	if (m_drawImage)
	{
		glDeleteTextures(iRes[1],tex);
	}
	
}



void ContourMesh::MallocMemory(unsigned int* ContourNum, int imageSize[], int stickNum)
{
	int i;

	m_ContourNum = (int*)malloc(imageSize[1]*sizeof(int));
	for(i=0; i < imageSize[1]; i++)
	{
		m_ContourNum[i] = ContourNum[i+1]-ContourNum[i];
	}

	iRes[0] = imageSize[0];	iRes[1] = imageSize[1];	iRes[2] = imageSize[2];
	m_StnodeTable = (float*)malloc(stickNum*3*sizeof(float)); 
	m_EdnodeTable = (float*)malloc(stickNum*3*sizeof(float)); 
	totalstickNum = stickNum;

	
}

void ContourMesh::PerformVSA3D()
{
	GLKPOSITION Pos;
	VSAMesh *mesh;
	int VSAiter = 20;
	int simpratio = 10;
	double desireddistortionerror = 1.0*sampleWidth*sampleWidth;

	for(Pos=VSAMeshList.GetHeadPosition(); Pos!=NULL; )
	{		
		mesh = (VSAMesh *)(VSAMeshList.GetNext(Pos));
		PerformVSA2D(mesh, VSAiter, desireddistortionerror, simpratio);			
	}


}

double ContourMesh::PerformVSA2D(VSAMesh *vmesh, int iter, double paradistterror, int simpratio)
{	
	double tempdistterror;
	double maxdistterror = 0.0;
	GLKGeometry geo;
	VSAEdge *curredge;
	VSANode *firstnode;
	VSANode *startnode;
	VSANode *endnode;
	VSANode *currnode;
	double startnodecoord[3];
	double endnodecoord[3];
	double edgevec[3];
	double resultnormal[3];
	double tempnormal[3];
	double dir[3] = {0.0, 1.0, 0.0};

	VSAMesh *simplifiedpatch = new VSAMesh();

	VSA vsaoper;
	
	vsaoper.InitializeVSA(vmesh, vmesh->GetVSAEdgeList().GetCount()/simpratio, 2);  //Warning: this may be changed when you want to cancle VSA segmentation
	
	vsaoper.PerformVSA2D(iter, paradistterror, false); 
	
	vsaoper.BinaryImageInOutCorrection(imageOrigin, sampleWidth);

	return maxdistterror;
}

//void ContourMesh::ConvertContourToVSAMesh(float* st_stick, float* ed_stick, int* stickID, int stickNum)
//{
//	int i, j, num, st;
//	int VSAiter = 20;
//	double distortRatioError;
//	double simpRatio = 10;
//	int id, patchNum = 0, patchIndex = 0;
//	
//
//	
//	int count = 0;
//	st = 0;
//	int prev;
//	
//	int *patchID = new int[stickNum]; // define which patch the stick is belonged to
//	unsigned int *patchStack = new unsigned int[stickNum]; //store each contour
//	unsigned int *ContourNum = new unsigned int[iRes[1]]; // the number of stick for each layer
//
//	// initialize
//	for(j=0;j < stickNum; j++)
//	{
//		patchID[j] = stickNum+1;
//		patchStack[j] = 0;
//	}
//	for(j=0;j < iRes[1]; j++)
//	{
//		ContourNum[j] = 0;
//	}
//			
//
//	// for each slice it may exist multiple contours
//	// so that use patchNum to count how many contour per slice			                
//	for(i=0; i<iRes[1]; i++)
//	{
//		patchNum = 0;
//		if (m_ContourNum[i]>0) //assigned in MallocMemory function before
//		{
//			do 
//			{
//				count = 0;
//				for(j=st; j < st+m_ContourNum[i]; j++)
//				{
//					id = stickID[j];
//					if (patchID[j] < patchID[id])
//					{
//						patchID[id] = patchID[j];
//						count++;
//					}
//					else if (patchID[j] > patchID[id])
//					{
//						patchID[j] = patchID[id];
//						count++;
//					}
//					else if (patchID[j] >= stickNum+1)
//					{
//						patchNum++;
//						count++;
//						patchID[id] = patchNum;
//						patchID[j] = patchNum; 
//					}
//				}
//			} while (count>0);
//
//			thrust::host_vector<short> A(m_ContourNum[i]);
//			thrust::copy(patchID+st, patchID + st + m_ContourNum[i], A.begin()); //copy a segment of patchID array to A
//			thrust::sort(A.begin(),A.end()); // sort A
//			thrust::host_vector<short>::iterator iter = thrust::unique(A.begin(), A.end()); // Move unique elements to the front of a range 
//
//			prev = A[0];
//			patchStack[patchIndex] = A[0];
//			patchIndex++;
//			int a;
//			for(a=1; a < m_ContourNum[i]; a++)
//			{
//				if (A[a]<= prev) break;
//				patchStack[patchIndex] = A[a]; //store all the patches in each slice
//				patchIndex++;
//				prev = A[a];
//			}
//						
//			ContourNum[i] = a; //store how many contour for each slice
//			st = st + m_ContourNum[i];
//		}
//	}
//			
//
//	printf("num of contour %d\n", patchIndex);
//	VSAMeshNum = patchIndex;
//
//	float xx, yy, zz;
//	st = 0;
//	int index = 0, k = 0;
//	num = 0;
//	VSAMesh *vmesh;
//	VSAEdge *vedge;
//	VSANode *vnode;
//	count = 0;
//
//	for(i=0; i<iRes[1]; i++)
//	{
//		if (m_ContourNum[i] > 0)
//		{
//			
//			for(k=num ; k < num + ContourNum[i]; k++) // for each patch in one slice
//			{
//				vmesh = new VSAMesh();
//				vmesh->SetIndexNo(count);
//
//				for(j=0; j < m_ContourNum[i]; j++)
//				{
//					if (patchID[st+j]==patchStack[k])
//					{
//						vedge = new VSAEdge();
//						vedge->SetIndexNo(st+j);
//						vedge->SetPrevIndexNo(stickID[st+j]);
//
//						if (st+j > stickNum || stickID[st+j]> stickNum) printf("warning!!!!\n");
//
//						vnode = new VSANode();
//										
//						xx = st_stick[2*st+2*j];
//						zz = st_stick[2*st+2*j+1];
//						yy = (i+1)*thickness;
//
//						vnode->SetCoord3D(xx,yy,zz);
//						vedge->SetStartPoint(vnode);
//
//
//						vnode = new VSANode();
//						xx = ed_stick[2*st+2*j];
//						zz = ed_stick[2*st+2*j+1];
//						vnode->SetCoord3D(xx,yy,zz);
//						vedge->SetEndPoint(vnode);
//						
//						vmesh->GetVSAEdgeList().AddTail(vedge);
//
//					}
//					//index++;
//				}
//				VSAMeshList.AddTail(vmesh);
//				count++;
//			}
//			
//			num += ContourNum[i];
//		}
//		st += m_ContourNum[i];
//	}
//
//
//	free(patchStack);
//	free(patchID);
//	free(ContourNum);
//		
//}

void ContourMesh::BuildContourTopology(float* st_stick, float* ed_stick, int* stickID, int stickNum, int* stickDir)
{
	int i, j, num, st;
	int id, patchNum = 0, patchIndex = 0;

	int count = 0;
	st = 0;
	int prev;

	int *patchID = new int[stickNum]; // define which patch the stick is belonged to
	unsigned int *patchStack = new unsigned int[stickNum]; //store each contour
	ContourNum = new unsigned int[iRes[1]]; // the number of stick for each layer
	for(j=0;j < stickNum; j++)
	{
		patchID[j] = stickNum+1;
		patchStack[j] = 0;
	}
	for(j=0;j < iRes[1]; j++)
	{
		ContourNum[j] = 0;
	}

	// -------------------------------- Grouping --------------------------------- //
	printf("Group sticks into each contour....\n");
	printf(" origin and gwidth %f %f %f\n", imageOrigin[0],imageOrigin[1], sampleWidth);
	// for each slice it may exist multiple contours
	// so that use patchNum to count how many contour per slice			
	for(i=0; i<iRes[1]; i++)
	{
		patchNum = 0;
		if (m_ContourNum[i]>0)//assigned in MallocMemory function before
		{
			do 
			{
				count = 0;
				for(j=st; j < st+m_ContourNum[i]; j++) 
				{
					id = stickID[j];
					if (patchID[j] < patchID[id])
					{
						patchID[id] = patchID[j];
						count++;
					}
					else if (patchID[j] > patchID[id])
					{
						patchID[j] = patchID[id];
						count++;
					}
					else if (patchID[j] >= stickNum+1)
					{
						patchNum++;
						count++;
						patchID[id] = patchNum;
						patchID[j] = patchNum; 
					}
				}
			} while (count>0);

			thrust::host_vector<short> A(m_ContourNum[i]);
			thrust::copy(patchID+st, patchID + st + m_ContourNum[i], A.begin()); //copy a segment of patchID array to A
			thrust::sort(A.begin(),A.end()); // sort A
			thrust::host_vector<short>::iterator iter = thrust::unique(A.begin(), A.end()); // Move unique elements to the front of a range 

			prev = A[0];
			patchStack[patchIndex] = A[0];
			patchIndex++;
			int a;
			for(a=1; a < m_ContourNum[i]; a++)
			{
				if (A[a]<= prev) break;
				patchStack[patchIndex] = A[a]; //store all the patches in each slice
				patchIndex++;
				prev = A[a];
			}

			ContourNum[i] = a;  //store how many contour for each slice
			st = st + m_ContourNum[i];
		}
	}
	printf("Finished grouping.\n");
	// --------------------------------------------------------- //
	printf("num of contour %d\n", patchIndex);
	VSAMeshNum = patchIndex;

	double xx, yy, zz;
	st = 0;
	int index = 0, k = 0, first;
	num = 0;
	VSAMesh *vmesh;
	VSAEdge *vedge;
	VSANode *lastnode;
	VSANode *nextnode;
	VSANode *firstnode;
	VSANode *vnode;
	count = 0;
	int localcount = 0;
	double aa, bb, cc;
	//printf("Building topology for each contour....\n");
	for(i=0; i<iRes[1]; i++)
	{
		if (m_ContourNum[i] > 0)
		{
			localcount = 0;
			for(k=num ; k < num + ContourNum[i]; k++) // for each patch in one slice
			{
				vmesh = new VSAMesh();
				vmesh->SetIndexNo(count);

				//first = 0;
				for(j=0; j < m_ContourNum[i]; j++)
				{
					if (patchID[st+j]==patchStack[k])
					{
						lastnode = new VSANode();

						xx = st_stick[2*st+2*j];
						zz = st_stick[2*st+2*j+1];
						yy = (i+1)*thickness;

						

						lastnode->SetCoord3D(xx,yy,zz);

						first = st+j;
						
					
						lastnode->SetStickDir(stickDir[2*(st+j)], stickDir[2*(st+j)+1]);
						firstnode = lastnode;
						break;
						
					}
				}
				//first = st+j;
				index = st+j;
				localcount = 0;
				while (true)
				{
					nextnode = new VSANode();

					xx = st_stick[stickID[index]*2];
					zz = st_stick[stickID[index]*2+1];
					yy = (i+1)*thickness;

				
					nextnode->SetStickDir(stickDir[2*stickID[index]], stickDir[2*stickID[index]+1]);
					
					nextnode->SetCoord3D(xx,yy,zz);

					
					vedge = new VSAEdge();

					vedge->SetEndPoint(lastnode);
					lastnode->AddEdge(vedge);
					vmesh->GetVSAEdgeList().AddTail(vedge);
					vmesh->GetVSANodeList().AddTail(lastnode);

					//vmesh->GetVSAEdgeList().AddHead(vedge);
					//vmesh->GetVSANodeList().AddHead(lastnode);

					lastnode->SetIndexNo(localcount);
					vedge->SetIndexNo(localcount);

					localcount++;

					lastnode = nextnode;

					index = stickID[index];

			
					//if (stickID[index] != first)
					if (index != first)
					{
						vedge->SetStartPoint(nextnode);
						nextnode->AddEdge(vedge);
					}
					else
					{
						
						vedge->SetStartPoint(firstnode);
						firstnode->GetVSAEdgeList().AddTail(vedge);

						delete nextnode;
						break;
					}

				}
				VSAMeshList.AddTail(vmesh);
				//printf("%d - no of node:%d\n",i,vmesh->GetVSANodeList().GetCount());
				count++;
			}

			num += ContourNum[i];
		}
		st += m_ContourNum[i];
	}


	GLKPOSITION Pos;
	GLKPOSITION Pos2;
	//double aa,bb,cc;

	for(Pos=VSAMeshList.GetHeadPosition(); Pos!=NULL; )
	{
		vmesh = (VSAMesh *)(VSAMeshList.GetNext(Pos));
		
		for(Pos2=vmesh->GetVSAEdgeList().GetHeadPosition(); Pos2!=NULL; )
		{
			vedge = (VSAEdge *)(vmesh->GetVSAEdgeList().GetNext(Pos2));

			vedge->GetStartPoint()->GetCoord3D(xx,yy,zz);
			vedge->GetEndPoint()->GetCoord3D(aa,bb,cc);

			vnode = vedge->GetStartPoint();

			if (vnode->GetStickDirX()<0)
			{
				if (cc > zz)
					vnode->SetStick((-(vnode->GetStickDirX())+1), vnode->GetStickDirZ(), -(vnode->GetStickDirX()), vnode->GetStickDirZ() );
				else
					vnode->SetStick(-(vnode->GetStickDirX()), vnode->GetStickDirZ(), -(vnode->GetStickDirX())+1, vnode->GetStickDirZ() );
			}
			else
			{
				if (aa > xx)
					vnode->SetStick(vnode->GetStickDirX(), vnode->GetStickDirZ(), vnode->GetStickDirX(), vnode->GetStickDirZ()+1 );
				else
					vnode->SetStick(vnode->GetStickDirX(), vnode->GetStickDirZ()+1, vnode->GetStickDirX(), vnode->GetStickDirZ() );
			}

		}
		


	}


	printf("Finished build-up.\n");
	free(patchStack);
	free(patchID);
	//free(ContourNum);

}

void ContourMesh::WriteBMP(const char *filename, GLubyte* data, int m_SizeX, int m_SizeY)
{
	int w = m_SizeX;
	int h = m_SizeY;

	int filesize = 54 + 3*w*h;

	FILE *fp;
	fp = fopen(filename,"wb");
	if(!fp) {printf("Cannot open the file!! \n"); return ;}

	unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
	//unsigned char bmpfileheader[14] = {'B','M', 0,0,0, 0,0, 0,0, 54,0,0,0};
	unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
	unsigned char bmppad[3] = {0,0,0};
	//unsigned char bmppad[4] = {0,0,0,0};

	bmpfileheader[ 2] = (unsigned char)(filesize    );
	bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
	bmpfileheader[ 4] = (unsigned char)(filesize>>16);
	//bmpfileheader[ 5] = (unsigned char)(filesize>>24);

	bmpinfoheader[ 4] = (unsigned char)(       w    );
	bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
	bmpinfoheader[ 6] = (unsigned char)(       w>>16);
	//bmpinfoheader[ 7] = (unsigned char)(       w>>24);
	bmpinfoheader[ 8] = (unsigned char)(       h    );
	bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
	bmpinfoheader[10] = (unsigned char)(       h>>16);
	//bmpinfoheader[11] = (unsigned char)(       h>>24);

	fwrite(bmpfileheader,1,14,fp);
	fwrite(bmpinfoheader,1,40,fp);

	for(int i=0; i<h; i++)
	{
		fwrite(data+(w*(i)*3),3,w,fp);
		fwrite(bmppad,1,(4-(w*3)%4)%4,fp);

	}

	fclose(fp);
}

void ContourMesh::ArrayToImage(bool *InNodes1, bool *InNodes2, int imageSize[], int base, bool bSave, bool bDisplay)
{
	int i, j, k;

	
	int rec[2], v, m;
	v = ((imageSize[0] > imageSize[2])? imageSize[0]:imageSize[2]);
	m = v/128;

	//if (v < 256) v = 256;
	//else if (v)

	v = 128*(m+1);
	//if (v%2) v++;
	
	rec[0] = v;
	rec[1] = v;
	bool bflag = false;
	unsigned int index, index_s;
	
	if (bDisplay)
	{

	
		unsigned char* Pixels = new unsigned char[rec[0]*rec[1]*3];
		//unsigned char *Pixels = (unsigned char*)malloc(rec[0]*rec[1]*3*sizeof(unsigned char));
		for(i=0; i < rec[0]*rec[1]; i++)
		{
			Pixels[3*i] = 0;
			Pixels[3*i+1] = 0;
			Pixels[3*i+2] = 0;
		}
		printf("Loading Contour to Images......%d %d %d %d\n", rec[0],rec[1],imageSize[0],imageSize[2]);

		

		tex = (GLuint *)malloc((imageSize[1]+base)*sizeof(GLuint));
		glGenTextures( imageSize[1]+base, tex );

		glEnable(GL_TEXTURE_2D);

		
		for (i=0; i < (imageSize[1]+base-1); i++)
		{
			//printf("test \n");
			glBindTexture( GL_TEXTURE_2D, tex[i] );
			glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP);
			glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP);
			glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
			//printf("test 0\n");

			if (i>= base)
			{
				//printf("test 1\n");
				bflag = false;
				for(j=0; j < imageSize[0]; j++)
				{

					for(k=0; k < imageSize[2]; k++)
					{
						index = k*imageSize[0]*imageSize[1] + (i-base)*imageSize[0] + j;
						index_s = k*imageSize[0]*(imageSize[1]-1) + (i-base)*imageSize[0] + j;

						if (index > imageSize[0]*imageSize[1]*imageSize[2])
							printf("size error!!!!!!!!\n");

						

						if (i-base<imageSize[1]-1)
							bflag = InNodes2[index_s];
						else
							bflag = false;

						

						if (InNodes1[index] || bflag)
						//if (InNodes2[index_s])
						{
							Pixels[3*(j*rec[0]+k)] = 255;
							Pixels[3*(j*rec[0]+k)+1] = 255;
							Pixels[3*(j*rec[0]+k)+2] = 255;
							//if (i-base == 0)
							//	printf("%d %d %d %d \n", j, k, InNodes1[index],  InNodes2[index_s]);
						}
						else
						{
							Pixels[3*(j*rec[0]+k)] = 0;
							Pixels[3*(j*rec[0]+k)+1] = 0;
							Pixels[3*(j*rec[0]+k)+2] = 0;
						}
					}
				}
				//printf("test 2\n");
			}
			
			//printf("test 5\n");
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rec[0], rec[1], 0, GL_RGB, GL_UNSIGNED_BYTE, Pixels);
			glBindTexture(GL_TEXTURE_2D, 0);
			//printf("test 3\n");
			memset(Pixels, 0, rec[0]*rec[1]*3*sizeof(unsigned char));
		}

		delete [] Pixels;
		m_drawImage = true;
	}

	if (bSave)
	{
		unsigned char* data = new unsigned char[imageSize[0]*imageSize[2]*3];

		char filename[256];
		char value[10];
		
		
		for (i=0; i < (imageSize[1]+base-1); i++)
		{
			bflag = false;
			if (i<base)
			{
				filename[0] = '\0';
				strcat(filename,"Image\\");
				sprintf(value,"%d",i) ;
				strcat(filename,value);
				strcat(filename,".bmp");
				
				memset(data, 255, imageSize[0]*imageSize[2]*3*sizeof(unsigned char));

				WriteBMP(filename, data, imageSize[0], imageSize[2]);
				
			}
			else
			{
				filename[0] = '\0';
				strcat(filename,"Image\\");
				for(j=0; j < imageSize[0]; j++)
				{

					for(k=0; k < imageSize[2]; k++)
					{
						index = k*imageSize[0]*imageSize[1] + (i-base)*imageSize[0] + j;
						index_s = k*imageSize[0]*(imageSize[1]-1) + (i-base)*imageSize[0] + j;
						
						if (i-base<imageSize[1]-1)
							bflag = InNodes2[index_s];
						else
							bflag = false;

						if (InNodes1[index] || bflag)
						{
							data[3*(k*imageSize[0]+j)] = 255;
							data[3*(k*imageSize[0]+j)+1] = 255;
							data[3*(k*imageSize[0]+j)+2] = 255;
						}
						else
						{
							data[3*(k*imageSize[0]+j)] = 0;
							data[3*(k*imageSize[0]+j)+1] = 0;
							data[3*(k*imageSize[0]+j)+2] = 0;
						}
					}
				}
				sprintf(value,"%d",i) ;
				strcat(filename,value);
				strcat(filename,".bmp");

				WriteBMP(filename, data, imageSize[0], imageSize[2]);
			}
			
		}

		

		delete [] data;
	}

	


	
	printf("Finish loading.\n");


}


void ContourMesh::ArrayToImage(bool *nodes, int imageSize[])
{
	
	//RGB*** Pixels = new RGB**[imageSize[1]];
	int i, j, k;
	int rec[2], v;
	v = (imageSize[0] > imageSize[2]? imageSize[0]:imageSize[2]);

	if (v%2) v++;
	rec[0] = v;
	rec[1] = v;

	
	//unsigned char* Pixels = new unsigned char[imageSize[0]*imageSize[2]*3];
	unsigned char* Pixels = new unsigned char[rec[0]*rec[1]*3];
	//for(i=0; i < imageSize[0]*imageSize[2]; i++)
	for(i=0; i < rec[0]*rec[1]; i++)
	{
		Pixels[3*i] = 0;
		Pixels[3*i+1] = 0;
		Pixels[3*i+2] = 0;
	}

	printf("Loading Contour to Images......\n");

	unsigned int index;
	
	tex = (GLuint *)malloc(imageSize[1]*sizeof(GLuint));
	glGenTextures( imageSize[1], tex );
	
	glEnable(GL_TEXTURE_2D);
	
	for (i=0; i < imageSize[1]; i++)
	{
		
		glBindTexture( GL_TEXTURE_2D, tex[i] );
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP);
		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);


		for(j=0; j < imageSize[0]; j++)
		//for(j=0; j < rec[0]; j++)
		{
			
			for(k=0; k < imageSize[2]; k++)
			//for(k=0; k < rec[1]; k++)
			{
				index = k*imageSize[0]*imageSize[1] + i*imageSize[0] + j;
				//if (j >= imageSize[0] || k >=imageSize[2]) continue;
				

				if (nodes[index])
				{
					/*Pixels[3*(j*imageSize[2]+k)] = 255;
					Pixels[3*(j*imageSize[2]+k)+1] = 255;
					Pixels[3*(j*imageSize[2]+k)+2] = 255;*/

					Pixels[3*(j*rec[0]+k)] = 255;
					Pixels[3*(j*rec[0]+k)+1] = 255;
					Pixels[3*(j*rec[0]+k)+2] = 255;
				}
				else
				{
					/*Pixels[3*(j*imageSize[2]+k)] = 0;
					Pixels[3*(j*imageSize[2]+k)+1] = 0;
					Pixels[3*(j*imageSize[2]+k)+2] = 0;*/
					Pixels[3*(j*rec[0]+k)] = 0;
					Pixels[3*(j*rec[0]+k)+1] = 0;
					Pixels[3*(j*rec[0]+k)+2] = 0;
				}
			}
		}
		//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imageSize[0], imageSize[2], 0, GL_RGB, GL_UNSIGNED_BYTE, Pixels);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rec[0], rec[1], 0, GL_RGB, GL_UNSIGNED_BYTE, Pixels);
		glBindTexture(GL_TEXTURE_2D, 0);
	}



	delete [] Pixels;
	

	m_drawImage = true;
	printf("Finish loading.\n");
}

void ContourMesh::ArrayToContour(float* st_stick, float* ed_stick, double imgOri[], float imgWidth)
{
	int i, j, st, num;
	float xx, yy, zz;
	int index = 0;
	
	st = 0;
	
	for(i=0; i<iRes[1]; i++)
	{
		num = m_ContourNum[i];

		if (num > 0)
		{
			for(j=0; j < num; j++)
			{
				xx = st_stick[2*st+2*j];
				zz = st_stick[2*st+2*j+1];
				yy = (i+1)*thickness;
				
				
				//m_StnodeTable[3*index] = imgOri[0] + imgWidth*xx;
				m_StnodeTable[3*index] = xx;
				m_StnodeTable[3*index+1] = yy;
				m_StnodeTable[3*index+2] = zz;
				//m_StnodeTable[3*index+2] = imgOri[1] + imgWidth*zz;
				
				


				xx = ed_stick[2*st+2*j];
				zz = ed_stick[2*st+2*j+1];


				//m_EdnodeTable[3*index] = imgOri[0] + imgWidth*xx;
				m_EdnodeTable[3*index] = xx;
				m_EdnodeTable[3*index+1] = yy;
				m_EdnodeTable[3*index+2] = zz;
				//m_EdnodeTable[3*index+2] = imgOri[1] + imgWidth*zz;

				

				index++;
				
			}
		}

		st += num;
	}
	m_drawImage = false;

}

void ContourMesh::ArrayToContour(float* st_stick, float* ed_stick, unsigned int* id_stick)
{
	int i, j, st, num;
	float xx, yy, zz;
	int index = 0;

	for(i=0; i<iRes[1]; i++)
	{
		st = id_stick[i];
		num = m_ContourNum[i];

		if (num > 0)
		{
			for(j=0; j < num; j++)
			{
				xx = st_stick[2*st+2*j];
				zz = st_stick[2*st+2*j+1];
				yy = (i+1)*thickness;

				m_StnodeTable[3*index] = xx;
				m_StnodeTable[3*index+1] = yy;
				m_StnodeTable[3*index+2] = zz;


				

				xx = ed_stick[2*st+2*j];
				zz = ed_stick[2*st+2*j+1];
				

				m_EdnodeTable[3*index] = xx;
				m_EdnodeTable[3*index+1] = yy;
				m_EdnodeTable[3*index+2] = zz;

				

				index++;
			}
		}
	}

	m_drawImage = false;

}

//----SGM code by KaChun----//
void ContourMesh::Output_SGM_FILE(ContourMesh *supt_mesh)
{

	GLKObList templistforcontours;
	GLKObList templistfortrglnodes;
	templistforcontours.RemoveAll();
	templistfortrglnodes.RemoveAll();
	RPFilesInterface *rpfileintfc = new RPFilesInterface();
	rpfileintfc->thickness = this->thickness;
	rpfileintfc->lyrNum = this->iRes[1];
	rpfileintfc->layerarray = new Layer[rpfileintfc->lyrNum];
	int layercount = 0;
	int localctcount = 0;
	double firstcoordofeachct[3];
	double currentcoord[3];

	VSAMesh *vmesh;
	VSANode *nextnode;
	VSANode *vnode;

	GLKPOSITION Pos_c_m ;
	GLKPOSITION Pos1_c_n;
	GLKPOSITION Pos_s_m ;
	GLKPOSITION Pos1_s_n;

	int tmpctcount = 0;
	

	Pos_c_m = this->VSAMeshList.GetHeadPosition();
	Pos_s_m = supt_mesh->VSAMeshList.GetHeadPosition();
	templistforcontours.RemoveAll();
	
	

	for(int i=0;i<this->iRes[1];i++)// skipped the first of supt and last of c_mesh
	{		
		templistforcontours.RemoveAll();

		//process part contours
		tmpctcount = 0;
		for(int j =0;j<this->ContourNum[i];j++)
		{
			VSAMesh *tempqbody = (VSAMesh *)(this->VSAMeshList.GetNext(Pos_c_m));
			Pos1_c_n = tempqbody->GetVSANodeList().GetHeadPosition();
			VSANode *startnode = (VSANode *)(tempqbody->GetVSANodeList().GetNext(Pos1_c_n));
			
			startnode->GetCoord3D(firstcoordofeachct[0], firstcoordofeachct[1], firstcoordofeachct[2]);
			if(j == 0) // initialize one time only
			{
				rpfileintfc->layerarray[i].height = firstcoordofeachct[1];
				rpfileintfc->layerarray[i].thickness = rpfileintfc->thickness;
			}

			nextnode = NULL;
			Contour *newct = new Contour();
			newct->partorsppt = true;
			templistfortrglnodes.RemoveAll();
			
			templistfortrglnodes.AddTail(startnode);
			for(int i=1;i<tempqbody->GetVSANodeList().GetCount();i++)
			{
				nextnode = (VSANode *)(tempqbody->GetVSANodeList().GetNext(Pos1_c_n));
				nextnode->GetCoord3D(currentcoord[0], currentcoord[1], currentcoord[2]);
				templistfortrglnodes.AddTail(nextnode);
			}

			newct->pntNum = templistfortrglnodes.GetCount();
			newct->xp = new double[newct->pntNum];
			newct->yp = new double[newct->pntNum];
			VSANode *tempp;
			double pcoord[3];
			int localcount = 0;
			for(GLKPOSITION Pos3=templistfortrglnodes.GetHeadPosition(); Pos3!=NULL;)
			{
				tempp = (VSANode *)(templistfortrglnodes.GetNext(Pos3));
				tempp->GetCoord3D(pcoord[0], pcoord[1], pcoord[2]);

				newct->xp[localcount] = pcoord[0];
				newct->yp[localcount] = pcoord[2];

				localcount++;
			}
			templistforcontours.AddTail(newct);
			tmpctcount++;
			//printf("ct:%d nodes:%d\n",tmpctcount,templistfortrglnodes.GetCount());
		}

		//printf("part ctcount:%d \n",tmpctcount);

		if (i != (this->iRes[1]-1))// there is only part layer in the last layer
		{
			tmpctcount = 0;
			for(int j =0;j<supt_mesh->ContourNum[i];j++)
			{
				VSAMesh *tempqbody = (VSAMesh *)(supt_mesh->VSAMeshList.GetNext(Pos_s_m));
				Pos1_s_n = tempqbody->GetVSANodeList().GetHeadPosition();
				VSANode *startnode = (VSANode *)(tempqbody->GetVSANodeList().GetNext(Pos1_s_n));
				
				startnode->GetCoord3D(firstcoordofeachct[0], firstcoordofeachct[1], firstcoordofeachct[2]);

				nextnode = NULL;
				Contour *newct = new Contour();
				newct->partorsppt = false;
				templistfortrglnodes.RemoveAll();
				
				templistfortrglnodes.AddTail(startnode);
				for(int i=1;i<tempqbody->GetVSANodeList().GetCount();i++)
				{
					nextnode = (VSANode *)(tempqbody->GetVSANodeList().GetNext(Pos1_s_n));
					nextnode->GetCoord3D(currentcoord[0], currentcoord[1], currentcoord[2]);
					templistfortrglnodes.AddTail(nextnode);
				}

				newct->pntNum = templistfortrglnodes.GetCount();
				newct->xp = new double[newct->pntNum];
				newct->yp = new double[newct->pntNum];
				VSANode *tempp;
				double pcoord[3];
				int localcount = 0;
				for(GLKPOSITION Pos3=templistfortrglnodes.GetHeadPosition(); Pos3!=NULL;)
				{
					tempp = (VSANode *)(templistfortrglnodes.GetNext(Pos3));
					tempp->GetCoord3D(pcoord[0], pcoord[1], pcoord[2]);

					newct->xp[localcount] = pcoord[0];
					newct->yp[localcount] = pcoord[2];

					localcount++;
				}
				templistforcontours.AddTail(newct);
				tmpctcount++;
				//printf("ct:%d nodes:%d\n",tmpctcount,templistfortrglnodes.GetCount());
			}

			//printf("supt ctcount:%d \n",tmpctcount);

		}


		rpfileintfc->layerarray[i].contourNum = templistforcontours.GetCount();
		//printf("layer:%d contour:%d height:%f\n",i,templistforcontours.GetCount(),rpfileintfc->layerarray[i].height);
		rpfileintfc->layerarray[i].contourarray =(Contour **) new long[templistforcontours.GetCount()];

		localctcount = 0;
		for(GLKPOSITION Pos3=templistforcontours.GetHeadPosition(); Pos3!=NULL;)
		{
			Contour *tempct = (Contour *)(templistforcontours.GetNext(Pos3));
			rpfileintfc->layerarray[i].contourarray[localctcount] = tempct;
			localctcount++;
		}
		templistforcontours.RemoveAll();
	}


	for(int i=0; i<this->iRes[1]; i++)
	{
		rpfileintfc->layerarray[i].BuildTopologyOfContours();
	}
	

	
	rpfileintfc->OutputInsightSGMFile(this,"OutPutSgmFile");

	delete rpfileintfc;

}

bool RPFilesInterface::OutputInsightSGMFile(ContourMesh* c_mesh,const char *filename)
{

	char filepath[100],name[100],fileext[10],str[100];

	char filelocation[256] = "";


	strcpy(filepath,"rp files\\");
	strcpy(name,filename);
	strcpy(fileext,".sgm");

	strcat(filelocation,filepath);
	strcat(filelocation,name);
	strcat(filelocation,fileext);

	FILE *sgm_source;
	FILE *sgm_output;

	sgm_source = fopen("rp files\\sgmsource.sgm", "r");
	sgm_output = fopen(filelocation, "w");
    if(!sgm_source)
	{
		printf("===============================================\n");
	    printf("Can not open sgmsource.sgm\n");
		printf("===============================================\n");
	    return false;
	}
	if(!sgm_output)
	{
		printf("===============================================\n");
	    printf("Can not open sgm_output.sgm\n");
		printf("===============================================\n");
	    return false;
	}

	while(true)//get heading of specific machine
	{

		fgets (str , 100 , sgm_source) ;

		if(strstr (str,"STEP")!= NULL)
			fprintf(sgm_output,"STEP %f\n",this->thickness);
		else if(strstr (str,"HEIGHT")!= NULL)
			fprintf(sgm_output,"HEIGHT %f\n",this->thickness*this->lyrNum);
		else if(strstr (str,"FILE")!= NULL)
		{
			fprintf(sgm_output,str);
			break;
		}
		else
		{
			fprintf(sgm_output,str);
		}
	}

	for(int i=0; i<c_mesh->iRes[1]; i++)
	{
		fprintf(sgm_output,"Z %lf %lf F\n",this->layerarray[i].height,this->layerarray[i].thickness);
		if(i==0)
			fprintf(sgm_output,"O 0\n");
		for(int j=0; j<this->layerarray[i].contourNum; j++)
		{
			if(this->layerarray[i].contourarray[j]->printexplitcitly)
				this->layerarray[i].contourarray[j]->PrintContoursInSGMFormat(sgm_output);
		}
	}

	fprintf(sgm_output,"END\n");


	fclose(sgm_source);
	fclose(sgm_output);
	return true;
}

void Contour::PrintContoursInSGMFormat(FILE *sgm_output)
{
	if(this->partorsppt)
	{
		fprintf(sgm_output,"4\n");
		fprintf(sgm_output,"4\n");
	}
	else
	{
		fprintf(sgm_output,"8\n");
		fprintf(sgm_output,"8\n");
	}
	fprintf(sgm_output,"%d\n",this->pntNum);
	for(int i=0; i<this->pntNum; i++)
	{
		fprintf(sgm_output,"%f %f\n",this->xp[i],this->yp[i]);
	}
	if(this->partorsppt)
	{
		if(this->area>0.0)
			fprintf(sgm_output,"CCIMNN0 %f %f %f %f %f %f\n",this->xmin,this->xmax,this->ymin,this->ymax,this->perimeter,this->area);
		else
			fprintf(sgm_output,"CCBMNN0 %f %f %f %f %f %f\n",this->xmin,this->xmax,this->ymin,this->ymax,this->perimeter,this->area);
	}
	else
	{
		if(this->area>0.0)
			fprintf(sgm_output,"CCISTN0 %f %f %f %f %f %f\n",this->xmin,this->xmax,this->ymin,this->ymax,this->perimeter,this->area);
		else
			fprintf(sgm_output,"CCBSTN0 %f %f %f %f %f %f\n",this->xmin,this->xmax,this->ymin,this->ymax,this->perimeter,this->area);
	}
	
	for(GLKPOSITION Pos = this->includingcontours.GetHeadPosition(); Pos!=NULL;)
	{
		Contour *temcontour = (Contour *)(this->includingcontours.GetNext(Pos));
		temcontour->PrintContoursInSGMFormat(sgm_output);
	}
	fprintf(sgm_output,"*\n");
}


void Layer::BuildTopologyOfContours()
{
	GLKGeometry geo;

	int maxareaind;
	Contour *tempcontour;

	//prepare for the xmax, xmin, ymax, ymin, area and perimeter
	for(int i=0; i<this->contourNum; i++)
	{
		this->contourarray[i]->xmax = -1.0e12;
		this->contourarray[i]->xmin = 1.0e12;
		this->contourarray[i]->ymax = -1.0e12;
		this->contourarray[i]->ymin = 1.0e12;
		for(int j=0; j<this->contourarray[i]->pntNum; j++)
		{
			if(this->contourarray[i]->xp[j] > this->contourarray[i]->xmax)
				this->contourarray[i]->xmax = this->contourarray[i]->xp[j];
			if(this->contourarray[i]->xp[j] < this->contourarray[i]->xmin)
				this->contourarray[i]->xmin = this->contourarray[i]->xp[j];
			if(this->contourarray[i]->yp[j] > this->contourarray[i]->ymax)
				this->contourarray[i]->ymax = this->contourarray[i]->yp[j];
			if(this->contourarray[i]->yp[j] < this->contourarray[i]->ymin)
				this->contourarray[i]->ymin = this->contourarray[i]->yp[j];
		}
		this->contourarray[i]->area = geo.SpatialPolygonArea2D(this->contourarray[i]->xp, this->contourarray[i]->yp, this->contourarray[i]->pntNum);
		this->contourarray[i]->perimeter = geo.SpatialPolygonPerimeter2D(this->contourarray[i]->xp, this->contourarray[i]->yp, this->contourarray[i]->pntNum);
	}

	//sorting contours according to the abs value of their area
	for(int i=0; i<this->contourNum; i++)
	{
		maxareaind = i;
		for(int k=i+1; k<this->contourNum; k++)
		{
			if(abs(this->contourarray[maxareaind]->area) < abs(this->contourarray[k]->area))
				maxareaind = k;
		}
		tempcontour = this->contourarray[maxareaind];
		this->contourarray[maxareaind] = this->contourarray[i];
		this->contourarray[i] = tempcontour;
	}

	for(int i=0; i<this->contourNum; i++)
	{
		this->contourarray[i]->includingcontours.RemoveAll();
		this->contourarray[i]->printexplitcitly = true;
	}

	if(contourNum!=0 && this->contourarray[0]->area<-1.0e-8)
	{
		printf("unbounded layer !!!\n");
		printf("%f\n",this->contourarray[0]->area);
		return;
	}

	for(int i=1; i<this->contourNum; i++)
	{
		for(int j=i-1; j>-1; j--)
		{
			if( this->contourarray[i]->area*this->contourarray[j]->area<0.0 && ( this->contourarray[i]->xmax<this->contourarray[j]->xmax && 
				this->contourarray[i]->xmin>this->contourarray[j]->xmin && this->contourarray[i]->ymax<this->contourarray[j]->ymax && 
				this->contourarray[i]->ymin>this->contourarray[j]->ymin) )
			{
				this->contourarray[i]->printexplitcitly = false; //not the out most coutour within the whole object?
				this->contourarray[j]->includingcontours.AddTail(this->contourarray[i]);
				break;
			}
		}
	}
}


//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

Contour::Contour()
{
	pntNum = 0;
	xp = NULL;
	yp = NULL;
}

Contour::~Contour()
{
	if(xp!=NULL)
		delete [] xp;
	if(yp!=NULL)
		delete [] yp;
}


Layer::Layer()
{
	contourNum = 0;
	contourarray = NULL;
}

Layer::~Layer()
{
	for(int i=0; i<contourNum; i++)
		delete (Contour *)contourarray[i];
	delete [] (Contour **)contourarray;
}

RPFilesInterface::RPFilesInterface()
{
	lyrNum = 0;
	layerarray = NULL;
}

RPFilesInterface::~RPFilesInterface()
{
	if(layerarray!=NULL)
		delete [] (Layer *)layerarray;
}
//----SGM code by KaChun----//



VSANode::VSANode(void)
{

}

VSANode::~VSANode(void)
{

}

void VSANode::AddEdge(VSAEdge *vedge)
{
	VSAEdgeList.AddTail(vedge);
}

void VSANode::GetNormal(double &nx, double &ny, double &nz)
{
	nx=normal[0];	ny=normal[1];	nz=normal[2];
}

void VSANode::SetNormal(double nx, double ny, double nz)
{
	normal[0]=nx;	normal[1]=ny;	normal[2]=nz;
}

VSAHeapNode::VSAHeapNode(void)
{
	index=0;
}

VSAHeapNode::~VSAHeapNode(void)
{

}


VSAEdge::VSAEdge(void)
{

}

VSAEdge::~VSAEdge(void)
{

}

VSA2DNode::VSA2DNode()
{
	ProxyIndicator=0;
	ct_pt[0] = 0.0;
	ct_pt[1] = 0.0;
	area=0.0;

	meshobj = NULL;		//this will not release memory when we call deconstructor
}
VSA2DNode::~VSA2DNode()
{
}


void VSAEdge::CalLength()
{
	double x1,y1,z1,x2,y2,z2;

	pStartPoint->GetCoord3D(x1,y1,z1);	pEndPoint->GetCoord3D(x2,y2,z2);
	length=sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
}

VSAMesh::VSAMesh(void)
{

}

VSAMesh::~VSAMesh(void)
{
	ClearAll();
}


GLKObject* VSAMesh::FindEdgeListByIndex(unsigned int i)
{
	GLKPOSITION Pos;

	for(Pos=VSAEdgeList.GetHeadPosition();Pos!=NULL;)
	{
		VSAEdge *temp= (VSAEdge *)(VSAEdgeList.GetNext(Pos));
		if (temp->GetIndexNo() == i)
			return temp;
	}
}

GLKObject* VSAMesh::FindPrevEdgeListByIndex(unsigned int i)
{
	GLKPOSITION Pos;

	for(Pos=VSAEdgeList.GetHeadPosition();Pos!=NULL;)
	{
		VSAEdge *temp= (VSAEdge *)(VSAEdgeList.GetNext(Pos));
		if (temp->GetPrevIndexNo() == i)
			return temp;
	}
}

void VSAMesh::FindNextAndPrevEdgeListByIndex(VSAEdge* stpt_connected_edge, VSAEdge* edpt_connected_edge, 
											 unsigned int prev, unsigned int next)
{
	GLKPOSITION Pos;
	bool st = false, ed = false;

	for(Pos=VSAEdgeList.GetHeadPosition();Pos!=NULL;)
	{
		VSAEdge *temp= (VSAEdge *)(VSAEdgeList.GetNext(Pos));
		if (temp->GetIndexNo() == prev)
		{
			stpt_connected_edge = temp;
			st = true;
		}
		if (temp->GetPrevIndexNo() == next)
		{
			edpt_connected_edge = temp;
			ed = true;
		}
		if (st && ed) return;
			
	}


}

void VSAMesh::ClearAll()
{
	GLKPOSITION Pos;

	for(Pos=VSAEdgeList.GetHeadPosition();Pos!=NULL;)
	{
		VSAEdge *temp=
			(VSAEdge *)(VSAEdgeList.GetNext(Pos));
		delete temp;
	}
	VSAEdgeList.RemoveAll();
}





VSA::VSA()
{
	v_mesh = NULL;		//this will not release memory when we call deconstructor
	m_Nodes = NULL;
	m_Proxies = NULL;
}
VSA::~VSA()
{
	if(m_Nodes!=NULL)
	{
		for(int i=0; i<m_FaceNum; i++)
			delete (VSA2DNode *)m_Nodes[i];
		delete [] (VSA2DNode **)m_Nodes;
	}
	if(m_Proxies!=NULL)
	{
		for(int i=0; i<m_RegionNum; i++)
			delete (double *)m_Proxies[i];
		delete [] (double **)m_Proxies;
	}
}

void VSA::InitializeVSA(VSAMesh *mesh, int RegionNum, short Dimension)
{
	m_dimension = Dimension;
	m_RegionNum = RegionNum;

	if(m_RegionNum==0)	m_RegionNum++;
	m_FaceNum = mesh->GetVSAEdgeList().GetCount();

	double xx, yy, zz;
	VSAEdge *edge = (VSAEdge*)mesh->GetVSAEdgeList().GetHead();
	VSANode *node = edge->GetStartPoint();
	node->GetCoord3D(xx, yy, zz);
	layerInd = ceil(yy*100.0);

	//printf("%f %d \n", yy, layerInd);
	v_mesh = mesh;

	m_Nodes = (VSA2DNode **)new long[m_FaceNum];		//Note: we suppose the index for edges and faces are ready for use
	for(int i=0; i<m_FaceNum; i++)	
	{
		m_Nodes[i] = new VSA2DNode();
	}

	m_Proxies = (double **)new long[m_RegionNum];
	for(int i=0; i<m_RegionNum; i++)
	{
		if(m_dimension==2)
			m_Proxies[i] = new double[3];
	}
}

void VSA::PerformVSA2D(int iterNum, double desiredDistortError, bool needContourCheck)
{
	GLKGeometry geo;
	GLKMatrixLib mal;

	VSAEdge *edge;
	double st[3], ed[3];
	GLKPOSITION Pos;
	int i=0;
	
	for(Pos = v_mesh->GetVSAEdgeList().GetHeadPosition(); Pos!=NULL; )
	{
		edge = (VSAEdge *)v_mesh->GetVSAEdgeList().GetNext(Pos);
		edge->SetRegionIndex(-1);
		edge->GetStartPoint()->GetCoord3D(st[0], st[1], st[2]);
		edge->GetEndPoint()->GetCoord3D(ed[0], ed[1], ed[2]);
		edge->CalLength();
		

		m_Nodes[edge->GetIndexNo()]->meshobj = edge;
		m_Nodes[edge->GetIndexNo()]->area = edge->GetLength();
		m_Nodes[edge->GetIndexNo()]->ProxyIndicator = -1;
		m_Nodes[edge->GetIndexNo()]->ct_pt[0] = (st[0]+ed[0])/2.0;
		m_Nodes[edge->GetIndexNo()]->ct_pt[1] = (st[0]+ed[0])/2.0;
	}

	int steplength = v_mesh->GetVSAEdgeList().GetCount()/m_RegionNum;

	
	for(i=0; i < m_RegionNum; i++)
	{
		GLKPOSITION tempPos = v_mesh->GetVSAEdgeList().FindIndex(i*steplength);
		edge = (VSAEdge*)v_mesh->GetVSAEdgeList().GetAt(tempPos);

		edge->GetStartPoint()->GetCoord3D(st[0],st[1],st[2]);
		edge->GetEndPoint()->GetCoord3D(ed[0],ed[1],ed[2]);
		
		m_Nodes[edge->GetIndexNo()]->ProxyIndicator = i;


		geo.CalLineEquation(m_Proxies[i][0], m_Proxies[i][1], m_Proxies[i][2], st[0], st[2], ed[0], ed[2]);

	}

	GLKHeap *heap=new GLKHeap(m_FaceNum*2,true);
	double ***covmatrixarray = (double ***)new long[m_RegionNum];
	for(int i=0; i<m_RegionNum; i++)
		mal.CreateMatrix(covmatrixarray[i], 2, 2);	
	double *totalareaforregions = new double[m_RegionNum];
	double *regioncenterx = new double[m_RegionNum];
	double *regioncenterz = new double[m_RegionNum];
	double *mindisttforregions = new double[m_RegionNum];
	double *regiondistortionerrors = new double[m_RegionNum];
	VSA2DNode **tempvsanodelist = (VSA2DNode **)new long[m_RegionNum];

	double localmaxregiondistterror;
	int maxdisttregion;
	int maxdisttvsanodeind;

	localmaxregiondistterror = VSACoreIterations(iterNum, heap, covmatrixarray, totalareaforregions, regioncenterx, regioncenterz,
		mindisttforregions, regiondistortionerrors, tempvsanodelist, maxdisttregion, maxdisttvsanodeind, needContourCheck);
	
//	printf("leave vSA core \n");
	//if (localmaxregiondistterror > 0.0)
	if (layerInd == 1)
		printf("localmaxregiondistterror %d %d %f \n",layerInd, maxdisttregion, localmaxregiondistterror);
	delete heap;
	for(int i=0; i<m_RegionNum; i++)
		mal.DeleteMatrix(covmatrixarray[i], 2, 2);
	delete [] (double ***)covmatrixarray;
	delete [] totalareaforregions;
	delete [] regioncenterx;
	delete [] regioncenterz;
	delete [] mindisttforregions;
	delete [] regiondistortionerrors;
	delete [] (VSA2DNode **)tempvsanodelist;

	return ;

}


double VSA::VSACoreIterations(int maxiter, GLKHeap *heap, double ***covmatrixarray, double *totalareaforregions, 
							  double *regioncenterx, double *regioncenterz, double *mindisttforregions, 
							  double *regiondistortionerrors, VSA2DNode **tempvsanodelist, int &maxdistterrorregion,
							  int &maxdisttvsanodeind, bool needcontourbdcheck)
{
	GLKGeometry geo;
	GLKMatrixLib mal;

	VSANode *st_pt;
	VSANode *ed_pt;
	VSAEdge *edgewithstartpnt;
	VSAEdge *edgewithendpnt;
	VSAHeapNode *tempvsaheapnode;
	VSAHeapNode *topheapnode;

	double st[3];
	double ed[3];
	double tempdistterror;

	VSAEdge *edge;
	double p1[2], p2[2];
	double **tempmatrix;
	mal.CreateMatrix(tempmatrix, 2, 2);


	double maxregiondistterror = -1.0e10;
	double maxfacedistterror = -1.0e10;

	int count = 0;
	GLKPOSITION Pos;
	while(count<maxiter)
	{
		count++;
		for(Pos=v_mesh->GetVSAEdgeList().GetHeadPosition(); Pos!=NULL; )
		{
			edge = (VSAEdge *)v_mesh->GetVSAEdgeList().GetNext(Pos);
			if(m_Nodes[edge->GetIndexNo()]->ProxyIndicator!=-1)
			{
				st_pt = edge->GetStartPoint();
				ed_pt = edge->GetEndPoint();

				edgewithstartpnt = (VSAEdge *)st_pt->GetVSAEdgeList().GetHead();
				if(edgewithstartpnt!=NULL && edgewithstartpnt->GetIndexNo()== edge->GetIndexNo())
					edgewithstartpnt = (VSAEdge *)st_pt->GetVSAEdgeList().GetTail();
				if(needcontourbdcheck)
				{
					if(v_mesh->GetVSAEdgeList().Find(edgewithstartpnt)==NULL)             //just check if the edge is contained in current patch
						edgewithstartpnt = NULL;
				}
				
				
				edgewithendpnt = (VSAEdge *)ed_pt->GetVSAEdgeList().GetTail();
				if(edgewithendpnt!=NULL && edgewithendpnt->GetIndexNo()==edge->GetIndexNo())
					edgewithendpnt = (VSAEdge *)ed_pt->GetVSAEdgeList().GetHead();
				if(needcontourbdcheck)
				{
					if(v_mesh->GetVSAEdgeList().Find(edgewithendpnt)==NULL)
						edgewithendpnt = NULL;
				}

			
				if(edgewithstartpnt!=NULL)
				{
					if(m_Nodes[edgewithstartpnt->GetIndexNo()]->ProxyIndicator==-1)
					{
						st_pt = edgewithstartpnt->GetStartPoint();
						ed_pt = edgewithstartpnt->GetEndPoint();

						st_pt->GetCoord3D(st[0], st[1], st[2]);
						ed_pt->GetCoord3D(ed[0], ed[1], ed[2]);
						tempvsaheapnode = new VSAHeapNode();

						tempvsaheapnode->attachedObj = m_Nodes[edgewithstartpnt->GetIndexNo()];
						tempvsaheapnode->whichproxyagainst = m_Nodes[edge->GetIndexNo()]->ProxyIndicator;
						tempvsaheapnode->SetValue(EvaluateDistortionError2D(m_Proxies[tempvsaheapnode->whichproxyagainst], 
							st[0], st[2], ed[0], ed[2], m_Nodes[edgewithstartpnt->GetIndexNo()]->area));
						heap->Insert(tempvsaheapnode);

					}
				}

				if(edgewithendpnt!=NULL)
				{
					if(m_Nodes[edgewithendpnt->GetIndexNo()]->ProxyIndicator==-1)
					{
						st_pt = edgewithendpnt->GetStartPoint();
						ed_pt = edgewithendpnt->GetEndPoint();

						st_pt->GetCoord3D(st[0], st[1], st[2]);
						ed_pt->GetCoord3D(ed[0], ed[1], ed[2]);

						tempvsaheapnode = new VSAHeapNode();

						tempvsaheapnode->attachedObj = m_Nodes[edgewithendpnt->GetIndexNo()];
						tempvsaheapnode->whichproxyagainst = m_Nodes[edge->GetIndexNo()]->ProxyIndicator;
						tempvsaheapnode->SetValue(EvaluateDistortionError2D(m_Proxies[tempvsaheapnode->whichproxyagainst], 
							st[0], st[2], ed[0], ed[2], m_Nodes[edgewithendpnt->GetIndexNo()]->area));
						heap->Insert(tempvsaheapnode);
					}
				}
			}
		}

		VSA2DNode *node;
		while(!heap->ListEmpty())
		{
			tempvsaheapnode = (VSAHeapNode *)heap->RemoveTop();
			topheapnode = tempvsaheapnode;

			node = (VSA2DNode *)tempvsaheapnode->attachedObj;
			if(node->ProxyIndicator==-1)
			{
				node->ProxyIndicator = tempvsaheapnode->whichproxyagainst;

				edge = (VSAEdge *)node->meshobj;

				st_pt = edge->GetStartPoint();
				ed_pt = edge->GetEndPoint();

				edgewithstartpnt = (VSAEdge *)st_pt->GetVSAEdgeList().GetHead();
				if(edgewithstartpnt!=NULL && edgewithstartpnt->GetIndexNo()==edge->GetIndexNo())
					edgewithstartpnt = (VSAEdge *)st_pt->GetVSAEdgeList().GetTail();
				if(needcontourbdcheck)
				{
					if(v_mesh->GetVSAEdgeList().Find(edgewithstartpnt)==NULL)             //just check if the edge is contained in current patch
						edgewithstartpnt = NULL;
				}

				edgewithendpnt = (VSAEdge *)ed_pt->GetVSAEdgeList().GetTail();
				if(edgewithendpnt!=NULL && edgewithendpnt->GetIndexNo()==edge->GetIndexNo())
					edgewithendpnt = (VSAEdge *)ed_pt->GetVSAEdgeList().GetHead();
				if(needcontourbdcheck)
				{
					if(v_mesh->GetVSAEdgeList().Find(edgewithendpnt)==NULL)
						edgewithendpnt = NULL;
				}

				
				if(edgewithstartpnt!=NULL)
				{
					if(m_Nodes[edgewithstartpnt->GetIndexNo()]->ProxyIndicator==-1)
					{
						st_pt = edgewithstartpnt->GetStartPoint();
						ed_pt = edgewithstartpnt->GetEndPoint();

						st_pt->GetCoord3D(st[0], st[1], st[2]);
						ed_pt->GetCoord3D(ed[0], ed[1], ed[2]);
						tempvsaheapnode = new VSAHeapNode();

						tempvsaheapnode->attachedObj = m_Nodes[edgewithstartpnt->GetIndexNo()];
						tempvsaheapnode->whichproxyagainst = m_Nodes[edge->GetIndexNo()]->ProxyIndicator;
						tempvsaheapnode->SetValue(EvaluateDistortionError2D(m_Proxies[tempvsaheapnode->whichproxyagainst], 
							st[0], st[2], ed[0], ed[2], m_Nodes[edgewithstartpnt->GetIndexNo()]->area));
						heap->Insert(tempvsaheapnode);
					}
				}
				if(edgewithendpnt!=NULL)
				{
					if(m_Nodes[edgewithendpnt->GetIndexNo()]->ProxyIndicator==-1)
					{
						st_pt = edgewithendpnt->GetStartPoint();
						ed_pt = edgewithendpnt->GetEndPoint();

						st_pt->GetCoord3D(st[0], st[1], st[2]);
						ed_pt->GetCoord3D(ed[0], ed[1], ed[2]);

						tempvsaheapnode = new VSAHeapNode();

						tempvsaheapnode->attachedObj = m_Nodes[edgewithendpnt->GetIndexNo()];
						tempvsaheapnode->whichproxyagainst = m_Nodes[edge->GetIndexNo()]->ProxyIndicator;
						tempvsaheapnode->SetValue(EvaluateDistortionError2D(m_Proxies[tempvsaheapnode->whichproxyagainst], 
							st[0], st[2], ed[0], ed[2], m_Nodes[edgewithendpnt->GetIndexNo()]->area));
						heap->Insert(tempvsaheapnode);
					}
				}

			}
			delete topheapnode;

		}

		for(int i=0; i<m_RegionNum; i++)
		{
			covmatrixarray[i][0][0] = 0.0;	covmatrixarray[i][0][1] = 0.0;	covmatrixarray[i][1][0] = 0.0;	covmatrixarray[i][1][1] = 0.0;	
			tempmatrix[0][0] = 0.0;	tempmatrix[0][1] = 0.0;	tempmatrix[1][0] = 0.0;	tempmatrix[1][1] = 0.0;	
			totalareaforregions[i] = 0.0;
			regioncenterx[i] = 0.0;
			regioncenterz[i] = 0.0;
		}

		for(Pos=v_mesh->GetVSAEdgeList().GetHeadPosition(); Pos!=NULL; )
		{
			edge = (VSAEdge *)v_mesh->GetVSAEdgeList().GetNext(Pos);
			edge->GetStartPoint()->GetCoord3D(st[0], st[1], st[2]);
			edge->GetEndPoint()->GetCoord3D(ed[0], ed[1], ed[2]);

			p1[0] = st[0];		p1[1] = st[2];
			p2[0] = ed[0];		p2[1] = ed[2];

			mal.ComputeInertiaMatrixOfSegment(p1, p2, edge->GetLength(), tempmatrix);

			covmatrixarray[m_Nodes[edge->GetIndexNo()]->ProxyIndicator][0][0] += tempmatrix[0][0];
			covmatrixarray[m_Nodes[edge->GetIndexNo()]->ProxyIndicator][0][1] += tempmatrix[0][1];
			covmatrixarray[m_Nodes[edge->GetIndexNo()]->ProxyIndicator][1][0] += tempmatrix[1][0];
			covmatrixarray[m_Nodes[edge->GetIndexNo()]->ProxyIndicator][1][1] += tempmatrix[1][1];

			totalareaforregions[m_Nodes[edge->GetIndexNo()]->ProxyIndicator] += m_Nodes[edge->GetIndexNo()]->area;
			regioncenterx[m_Nodes[edge->GetIndexNo()]->ProxyIndicator] += m_Nodes[edge->GetIndexNo()]->ct_pt[0]*m_Nodes[edge->GetIndexNo()]->area;
			regioncenterz[m_Nodes[edge->GetIndexNo()]->ProxyIndicator] += m_Nodes[edge->GetIndexNo()]->ct_pt[2]*m_Nodes[edge->GetIndexNo()]->area;
		}

		double eigvalues[2];
		for(int i=0; i<m_RegionNum; i++)
		{
			regioncenterx[i] = regioncenterx[i]/totalareaforregions[i];
			regioncenterz[i] = regioncenterz[i]/totalareaforregions[i];

			covmatrixarray[i][0][0] -= totalareaforregions[i]*regioncenterx[i]*regioncenterx[i];
			covmatrixarray[i][0][1] -= totalareaforregions[i]*regioncenterx[i]*regioncenterz[i];
			covmatrixarray[i][1][0] -= totalareaforregions[i]*regioncenterz[i]*regioncenterx[i];
			covmatrixarray[i][1][1] -= totalareaforregions[i]*regioncenterz[i]*regioncenterz[i];

			covmatrixarray[i][0][0] *= 1.0e10;
			covmatrixarray[i][0][1] *= 1.0e10;
			covmatrixarray[i][1][0] *= 1.0e10;
			covmatrixarray[i][1][1] *= 1.0e10;

			tempmatrix[0][0] = 0.0;  tempmatrix[0][1] = 0.0;  tempmatrix[1][0] = 0.0;  tempmatrix[1][1] = 0.0;  
			eigvalues[0] = 0.0;	eigvalues[1] = 0.0;
			mal.JacobianEigensystemSolver(covmatrixarray[i], 2, tempmatrix, eigvalues, 1.0e-5, 30);
			if(eigvalues[0]<eigvalues[1])
			{
				m_Proxies[i][0] = tempmatrix[0][0];		m_Proxies[i][1] = tempmatrix[1][0];		m_Proxies[i][2] = -(m_Proxies[i][0]*regioncenterx[i]+m_Proxies[i][1]*regioncenterz[i]);
			}
			else
			{
				m_Proxies[i][0] = tempmatrix[0][1];		m_Proxies[i][1] = tempmatrix[1][1];		m_Proxies[i][2] = -(m_Proxies[i][0]*regioncenterx[i]+m_Proxies[i][1]*regioncenterz[i]);
			}
		}

		for(int i=0; i<m_RegionNum; i++)	
		{
			mindisttforregions[i] = 1.0e10;
			tempvsanodelist[i] = NULL;
			regiondistortionerrors[i] = 0.0;
		}

		for(Pos=v_mesh->GetVSAEdgeList().GetHeadPosition(); Pos!=NULL; )
		{
			edge = (VSAEdge *)v_mesh->GetVSAEdgeList().GetNext(Pos);
			edge->GetStartPoint()->GetCoord3D(st[0], st[1], st[2]);
			edge->GetEndPoint()->GetCoord3D(ed[0], ed[1], ed[2]);
			tempdistterror = EvaluateDistortionError2D(m_Proxies[m_Nodes[edge->GetIndexNo()]->ProxyIndicator], st[0], st[2], ed[0], ed[2], 
				m_Nodes[edge->GetIndexNo()]->area);

			if(tempdistterror < mindisttforregions[m_Nodes[edge->GetIndexNo()]->ProxyIndicator])
			{
				mindisttforregions[m_Nodes[edge->GetIndexNo()]->ProxyIndicator] = tempdistterror;
				tempvsanodelist[m_Nodes[edge->GetIndexNo()]->ProxyIndicator] = m_Nodes[edge->GetIndexNo()];
			}

			regiondistortionerrors[m_Nodes[edge->GetIndexNo()]->ProxyIndicator] += tempdistterror;

			edge->RegionIndex = m_Nodes[edge->GetIndexNo()]->ProxyIndicator;		//just used to verify the 2D VSA
		}

		maxregiondistterror = -1.0e10;
		maxfacedistterror = -1.0e10;
		for(int i=0; i<m_RegionNum; i++)
		{
			if(regiondistortionerrors[i]>maxregiondistterror)
			{
				maxregiondistterror = regiondistortionerrors[i];
				maxdistterrorregion = i;
			}
		}

		for(Pos=v_mesh->GetVSAEdgeList().GetHeadPosition(); Pos!=NULL; )
		{
			edge = (VSAEdge *)v_mesh->GetVSAEdgeList().GetNext(Pos);
			if(m_Nodes[edge->GetIndexNo()]->ProxyIndicator==maxdistterrorregion)
			{
				edge->GetStartPoint()->GetCoord3D(st[0], st[1], st[2]);
				edge->GetEndPoint()->GetCoord3D(ed[0], ed[1], ed[2]);
				tempdistterror = EvaluateDistortionError2D(m_Proxies[maxdistterrorregion], st[0], st[2], ed[0], ed[2], 
					m_Nodes[edge->GetIndexNo()]->area);
				if(tempdistterror>maxfacedistterror && !(m_Nodes[edge->GetIndexNo()]->ct_pt[0]==tempvsanodelist[maxdistterrorregion]->ct_pt[0]&&
					m_Nodes[edge->GetIndexNo()]->ct_pt[2]==tempvsanodelist[maxdistterrorregion]->ct_pt[2]))
				{
					maxfacedistterror = tempdistterror;
					maxdisttvsanodeind = edge->GetIndexNo();
				}
			}
		}

		for(Pos=v_mesh->GetVSAEdgeList().GetHeadPosition(); Pos!=NULL; )
		{
			edge = (VSAEdge *)v_mesh->GetVSAEdgeList().GetNext(Pos);
			m_Nodes[edge->GetIndexNo()]->ProxyIndicator = -1;
		}
		for(int i=0; i<m_RegionNum; i++)
		{
			tempvsanodelist[i]->ProxyIndicator = i;
		}
	}

	mal.DeleteMatrix(tempmatrix, 2, 2);

	return maxregiondistterror;


}

float VSA::EvaluateDistortionError2D(double *proxy, double v1, double v2, double u1, double u2, double length)
{
	double pntonline[2];
	if(abs(proxy[0])>0.1)
	{
		pntonline[1] = 0.0;  pntonline[0] = -proxy[2]/proxy[0];
	}
	else
	{
		pntonline[0] = 0.0;  pntonline[1] = -proxy[2]/proxy[1];
	}

	double d1 = (v1-pntonline[0])*proxy[0]+(v2-pntonline[1])*proxy[1];
	double d2 = (u1-pntonline[0])*proxy[0]+(u2-pntonline[1])*proxy[1];

	float result = 0.33333333333333333*length*(d1*d1+d2*d2+d1*d2);

	return result;
}

void VSA::BinaryImageInOutCorrection(double *biorigin, double bigridwidth)
{

	if(m_RegionNum==1 || m_RegionNum==2)
		return;

	VSAEdge *tempedge = (VSAEdge *)v_mesh->GetVSAEdgeList().GetHead();
	VSAEdge *firstedge = tempedge;
	VSANode *firstnode = firstedge->GetStartPoint();
	int lastregionindex = firstedge->RegionIndex;

	//VSAEdge *lastedge; 
	VSAEdge *edge;
	
	
	double lastpt[3];
	double currpt[3];
	bool isviolate = false;
	double stickstart[2];
	double stickend[2];
	double p1[3], p2[3];
	VSANode *startingnode;
	VSANode *lastnode = NULL;
	VSANode *currentnode;
	VSAEdge *newtempedge;
	VSANode *tempnode;
	fakeregionnum = m_RegionNum;
	VSAEdge *startingedgeoflastregion;
	int count=0;	
	GLKPOSITION Pos;
	GLKPOSITION Pos2;
	int aa, bb, cc, dd;

	//printf("a \n");
	for(Pos=v_mesh->GetVSAEdgeList().GetHeadPosition(); Pos!=NULL; )
	{
		tempedge = (VSAEdge *)v_mesh->GetVSAEdgeList().GetNext(Pos);
		if(tempedge->RegionIndex!=lastregionindex)
		{
			count++;
			if(count==1)
			{
				lastnode = tempedge->GetStartPoint();
				//lastedge = tempedge;
				startingnode = lastnode;
				lastregionindex = tempedge->RegionIndex;
			}
			else
			{
				currentnode = tempedge->GetStartPoint();
				currentnode->GetCoord3D(currpt[0], currpt[1], currpt[2]);
				lastnode->GetCoord3D(lastpt[0], lastpt[1], lastpt[2]);

				//printf("test 1\n");
				for(Pos2=v_mesh->GetVSANodeList().GetHeadPosition(); Pos2!= NULL ; )
				{
					tempnode = (VSANode *)v_mesh->GetVSANodeList().GetNext(Pos2);

					if (tempnode->GetIndexNo() >= lastnode->GetIndexNo()+1 && tempnode->GetIndexNo() < currentnode->GetIndexNo())
					{
						tempnode->GetStick(aa, bb, cc, dd);
						stickstart[0] = biorigin[0]+aa*bigridwidth;
						stickstart[1] = biorigin[1]+bb*bigridwidth;
						stickend[0] = biorigin[0]+cc*bigridwidth;
						stickend[1] = biorigin[1]+dd*bigridwidth;
				
						if(!this->_IsTwoSegmentIntersect(lastpt[0], lastpt[2], currpt[0], currpt[2], stickstart[0], stickstart[1], stickend[0], stickend[1]))
						{
							//printf("going into............. %d %d %d \n", lastnode->GetIndexNo(), currentnode->GetIndexNo(), fakeregionnum);
							this->_RecursiveLocalVSA(v_mesh, lastnode, currentnode, fakeregionnum, biorigin, bigridwidth);
							//printf("leaving................\n");
							break;
						}
					}
					
				}
				//printf("test 2\n");
				lastnode = currentnode;
				lastregionindex = tempedge->RegionIndex;
				startingedgeoflastregion = tempedge;
			}
		}
	}

	//printf("b \n");
	if(lastnode!=NULL)
	{
		GLKPOSITION startedgepos = v_mesh->GetVSAEdgeList().Find(startingedgeoflastregion);
		GLKPOSITION endedgepos = v_mesh->GetVSAEdgeList().Find(tempedge);

		for(Pos=startedgepos; Pos!=endedgepos; )
		{
			VSAEdge *temlocaledge = (VSAEdge *)v_mesh->GetVSAEdgeList().GetNext(Pos);
			temlocaledge->RegionIndex = fakeregionnum;
		}
		fakeregionnum++;

		currentnode = tempedge->GetEndPoint();
		currentnode->GetCoord3D(currpt[0], currpt[1], currpt[2]);
		lastnode->GetCoord3D(lastpt[0], lastpt[1], lastpt[2]);

	
		int speciallimit;
		if(lastnode->GetIndexNo() > currentnode->GetIndexNo())
			speciallimit = currentnode->GetIndexNo()+v_mesh->GetVSAEdgeList().GetCount()-1;
		else
			speciallimit = currentnode->GetIndexNo();

		for(Pos=v_mesh->GetVSANodeList().GetHeadPosition(); Pos2!= NULL ; )
		{
			tempnode = (VSANode *)v_mesh->GetVSANodeList().GetNext(Pos);

			if (tempnode->GetIndexNo() >= lastnode->GetIndexNo()+1 && tempnode->GetIndexNo() < speciallimit)
			{
				tempnode->GetStick(aa, bb, cc, dd);
				stickstart[0] = biorigin[0]+aa*bigridwidth;
				stickstart[1] = biorigin[1]+bb*bigridwidth;
				stickend[0] = biorigin[0]+cc*bigridwidth;
				stickend[1] = biorigin[1]+dd*bigridwidth;

				if(!this->_IsTwoSegmentIntersect(lastpt[0], lastpt[2], currpt[0], currpt[2], stickstart[0], stickstart[1], stickend[0], stickend[1]))
				{
					this->_RecursiveLocalVSA(v_mesh, lastnode, currentnode, fakeregionnum, biorigin, bigridwidth);
					break;
				}
			}
		}
		lastnode = firstnode;
		currentnode = startingnode;
		currentnode->GetCoord3D(currpt[0], currpt[1], currpt[2]);
		lastnode->GetCoord3D(lastpt[0], lastpt[1], lastpt[2]);

		for(Pos=v_mesh->GetVSANodeList().GetHeadPosition(); Pos2!= NULL ; )
		{
			tempnode = (VSANode *)v_mesh->GetVSANodeList().GetNext(Pos);

			if (tempnode->GetIndexNo() >= lastnode->GetIndexNo()+1 && tempnode->GetIndexNo() < currentnode->GetIndexNo())
			{
				tempnode->GetStick(aa, bb, cc, dd);
				stickstart[0] = biorigin[0]+aa*bigridwidth;
				stickstart[1] = biorigin[1]+bb*bigridwidth;
				stickend[0] = biorigin[0]+cc*bigridwidth;
				stickend[1] = biorigin[1]+dd*bigridwidth;

				if(!this->_IsTwoSegmentIntersect(lastpt[0], lastpt[2], currpt[0], currpt[2], stickstart[0], stickstart[1], stickend[0], stickend[1]))
				{
					this->_RecursiveLocalVSA(v_mesh, lastnode, currentnode, fakeregionnum, biorigin, bigridwidth);
					break;
				}
			}
		}

	}


}

bool VSA::_IsTwoSegmentIntersect(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4)
{
	GLKGeometry geo;

	double a1,b1,c1,a2,b2,c2,xx,yy;

	geo.CalLineEquation(a1,b1,c1,x1,y1,x2,y2);
	geo.CalLineEquation(a2,b2,c2,x3,y3,x4,y4);

	if (!(geo.CalTwoLinesIntersection(a1,b1,c1,a2,b2,c2,xx,yy))) return false;

	double u1;
	if (x3==x4)
		u1=(yy-y3)/(y4-y3);
	else
		u1=(xx-x3)/(x4-x3);

	double u2;
	if (x1==x2)
		u2=(yy-y1)/(y2-y1);
	else
		u2=(xx-x1)/(x2-x1);
		

	if ((u1>=0.0) && (u1<=1.0) && (u2>=0.0) && (u2<=1.0)) return true;

	return false;
}

int VSA::_RecursiveLocalVSAForDistterror(VSAMesh *parapatch, VSANode *startnode, VSANode *endnode, int &currregionnum, double desireddistterror, double &largerdistterror)
{

	GLKGeometry geo;

	VSAMesh *tempatch = new VSAMesh();
	VSAEdge *temedge = (VSAEdge *)startnode->GetVSAEdgeList().GetTail();
	GLKPOSITION startedgepos = parapatch->GetVSAEdgeList().Find(temedge);
	temedge = (VSAEdge *)endnode->GetVSAEdgeList().GetHead();
	GLKPOSITION endedgepos = parapatch->GetVSAEdgeList().Find(temedge);

	GLKPOSITION Pos;
	int localcount = 0;
	for(Pos=startedgepos; Pos!=endedgepos; )
	{
		temedge = (VSAEdge *)parapatch->GetVSAEdgeList().GetNext(Pos);
		temedge->SetIndexNo(localcount);
		localcount++;
		tempatch->GetVSAEdgeList().AddTail(temedge);
	}
	temedge = (VSAEdge *)parapatch->GetVSAEdgeList().GetNext(Pos);
	temedge->SetIndexNo(localcount);
	tempatch->GetVSAEdgeList().AddTail(temedge);

	VSA newvsaoper;
	newvsaoper.InitializeVSA(tempatch, 2, 2);
	newvsaoper.PerformVSA2D(8, 1.0e8, true);

	//find out the boundary point bewteen two region
	temedge = (VSAEdge *)tempatch->GetVSAEdgeList().GetHead();
	int firstregionind = temedge->RegionIndex;
	VSANode *midpnt;
	for(Pos=tempatch->GetVSAEdgeList().GetHeadPosition(); Pos!=NULL; )
	{
		temedge = (VSAEdge *)tempatch->GetVSAEdgeList().GetNext(Pos);
		if(temedge->RegionIndex!=firstregionind)
		{
			midpnt = temedge->GetStartPoint();
			break;
		}
	}

	double startpntcoord[3];
	double midpntcoord[3];
	double endpntcoord[3];
	startnode->GetCoord3D(startpntcoord[0], startpntcoord[1], startpntcoord[2]);
	midpnt->GetCoord3D(midpntcoord[0], midpntcoord[1], midpntcoord[2]);
	endnode->GetCoord3D(endpntcoord[0], endpntcoord[1], endpntcoord[2]);
	double v1[3];
	double v2[3];
	double fakeproxy[3];
	double length;
	double regiondistterror;
	int firstregionnum = 1;
	int secondregionnum = 1;

	geo.CalLineEquation(fakeproxy[0], fakeproxy[1], fakeproxy[2], startpntcoord[0], startpntcoord[2], midpntcoord[0], midpntcoord[2]);
	temedge = (VSAEdge *)startnode->GetVSAEdgeList().GetTail();
	startedgepos = v_mesh->GetVSAEdgeList().Find(temedge);
	temedge = (VSAEdge *)midpnt->GetVSAEdgeList().GetHead();
	endedgepos = v_mesh->GetVSAEdgeList().Find(temedge);
	regiondistterror = 0.0;
	for(Pos=startedgepos; Pos!=endedgepos; )
	{
		temedge = (VSAEdge *)v_mesh->GetVSAEdgeList().GetNext(Pos);
		temedge->GetStartPoint()->GetCoord3D(v1[0], v1[1], v1[2]);
		temedge->GetEndPoint()->GetCoord3D(v2[0], v2[1], v2[2]);
		length = sqrt((v1[0]-v2[0])*(v1[0]-v2[0])+(v1[2]-v2[2])*(v1[2]-v2[2]));
		regiondistterror += this->EvaluateDistortionError2D(fakeproxy, v1[0], v1[2], v2[0], v2[2], length);
	}
	temedge = (VSAEdge *)v_mesh->GetVSAEdgeList().GetNext(Pos);
	temedge->GetStartPoint()->GetCoord3D(v1[0], v1[1], v1[2]);
	temedge->GetEndPoint()->GetCoord3D(v2[0], v2[1], v2[2]);
	length = sqrt((v1[0]-v2[0])*(v1[0]-v2[0])+(v1[2]-v2[2])*(v1[2]-v2[2]));
	regiondistterror += this->EvaluateDistortionError2D(fakeproxy, v1[0], v1[2], v2[0], v2[2], length);
	if(regiondistterror>desireddistterror)
		firstregionnum = this->_RecursiveLocalVSAForDistterror(v_mesh, startnode, midpnt, fakeregionnum,desireddistterror, largerdistterror);
	else
	{
		if(regiondistterror>largerdistterror)
			largerdistterror = regiondistterror;
	}

	geo.CalLineEquation(fakeproxy[0], fakeproxy[1], fakeproxy[2], midpntcoord[0], midpntcoord[2], endpntcoord[0], endpntcoord[2]);
	temedge = (VSAEdge *)midpnt->GetVSAEdgeList().GetTail();
	startedgepos = v_mesh->GetVSAEdgeList().Find(temedge);
	temedge = (VSAEdge *)endnode->GetVSAEdgeList().GetHead();
	endedgepos = v_mesh->GetVSAEdgeList().Find(temedge);
	regiondistterror = 0.0;
	for(Pos=startedgepos; Pos!=endedgepos; )
	{
		temedge = (VSAEdge *)v_mesh->GetVSAEdgeList().GetNext(Pos);
		temedge->GetStartPoint()->GetCoord3D(v1[0], v1[1], v1[2]);
		temedge->GetEndPoint()->GetCoord3D(v2[0], v2[1], v2[2]);
		length = sqrt((v1[0]-v2[0])*(v1[0]-v2[0])+(v1[2]-v2[2])*(v1[2]-v2[2]));
		regiondistterror += this->EvaluateDistortionError2D(fakeproxy, v1[0], v1[2], v2[0], v2[2], length);
	}
	temedge = (VSAEdge *)v_mesh->GetVSAEdgeList().GetNext(Pos);
	temedge->GetStartPoint()->GetCoord3D(v1[0], v1[1], v1[2]);
	temedge->GetEndPoint()->GetCoord3D(v2[0], v2[1], v2[2]);
	length = sqrt((v1[0]-v2[0])*(v1[0]-v2[0])+(v1[2]-v2[2])*(v1[2]-v2[2]));
	regiondistterror += this->EvaluateDistortionError2D(fakeproxy, v1[0], v1[2], v2[0], v2[2], length);
	if(regiondistterror>desireddistterror)
		secondregionnum = this->_RecursiveLocalVSAForDistterror(v_mesh, midpnt, endnode, fakeregionnum,desireddistterror, largerdistterror);
	else
	{
		if(regiondistterror>largerdistterror)
			largerdistterror = regiondistterror;
	}
	//increase region index to the number bigger than current region number by 1 or 2
	if(firstregionnum==1)
	{
		for(Pos=tempatch->GetVSAEdgeList().GetHeadPosition(); Pos!=NULL; )
		{
			temedge = (VSAEdge *)tempatch->GetVSAEdgeList().GetNext(Pos);
			if(temedge->RegionIndex==firstregionind)
			{
				temedge->RegionIndex = currregionnum;
			}
		}
	}
	currregionnum++;
	if(secondregionnum==1)
	{
		for(Pos=tempatch->GetVSAEdgeList().GetHeadPosition(); Pos!=NULL; )
		{
			temedge = (VSAEdge *)tempatch->GetVSAEdgeList().GetNext(Pos);
			if(temedge->RegionIndex!=currregionnum-1)
			{
				temedge->RegionIndex = currregionnum;
			}
		}
	}
	currregionnum++;

	tempatch->GetVSAEdgeList().RemoveAll();
	delete tempatch;

	return firstregionnum+secondregionnum;
}

int VSA::_RecursiveLocalVSA( VSAMesh *parapatch,  VSANode *startnode, VSANode *endnode, int &currregionnum, double *biorigin, double bigridwidth)
{
	VSAMesh *tempatch = new VSAMesh();
	VSAEdge *temedge = (VSAEdge *)startnode->GetVSAEdgeList().GetTail();
	GLKPOSITION startedgepos = parapatch->GetVSAEdgeList().Find(temedge);
	temedge = (VSAEdge *)endnode->GetVSAEdgeList().GetHead();
	GLKPOSITION endedgepos = parapatch->GetVSAEdgeList().Find(temedge);
	if (layerInd == 1)  
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Go into Recursive %d %d  %d\n", startnode->GetIndexNo(), endnode->GetIndexNo(), currregionnum);

	GLKPOSITION Pos;
	int localcount = 0;
	for(Pos=startedgepos; Pos!=endedgepos; )
	{
		temedge = (VSAEdge *)parapatch->GetVSAEdgeList().GetNext(Pos);
		temedge->SetIndexNo(localcount);
		localcount++;
		tempatch->GetVSAEdgeList().AddTail(temedge);
	}

	if (layerInd == 1)
	printf("local count %d \n", localcount);
	temedge = (VSAEdge *)parapatch->GetVSAEdgeList().GetNext(Pos);
	temedge->SetIndexNo(localcount);
	tempatch->GetVSAEdgeList().AddTail(temedge);

	VSA newvsaoper;
	newvsaoper.InitializeVSA(tempatch, 2, 2);
	newvsaoper.PerformVSA2D(8, 1.0e8, true);

	//find out the boundary point bewteen two region
	temedge = (VSAEdge *)tempatch->GetVSAEdgeList().GetTail();
	int secondregionid = temedge->RegionIndex;
	temedge = (VSAEdge *)tempatch->GetVSAEdgeList().GetHead();
	int firstregionind = temedge->RegionIndex;
	VSANode *midpnt;
	for(Pos=tempatch->GetVSAEdgeList().GetHeadPosition(); Pos!=NULL; )
	{
		temedge = (VSAEdge *)tempatch->GetVSAEdgeList().GetNext(Pos);

		

		if(temedge->RegionIndex!=firstregionind)
		{
			midpnt = temedge->GetStartPoint();
			break;
		}
	}

	double startpntcoord[3];
	double midpntcoord[3];
	double endpntcoord[3];
	startnode->GetCoord3D(startpntcoord[0], startpntcoord[1], startpntcoord[2]);
	midpnt->GetCoord3D(midpntcoord[0], midpntcoord[1], midpntcoord[2]);
	endnode->GetCoord3D(endpntcoord[0], endpntcoord[1], endpntcoord[2]);
	double stickstart[2];
	double stickend[2];
	int firstregionnum = 1;
	int secondregionnum = 1;
	VSANode *node;
	int a, b, c, d;
	for(Pos=parapatch->GetVSANodeList().GetHeadPosition(); Pos!=NULL; )
	{
		node = (VSANode*)parapatch->GetVSANodeList().GetNext(Pos);

		
		if (node->GetIndexNo()>=startnode->GetIndexNo()+1 && node->GetIndexNo() < midpnt->GetIndexNo())
		{
			node->GetStick(a, b, c, d);
			stickstart[0] = biorigin[0]+a*bigridwidth;
			stickstart[1] = biorigin[1]+b*bigridwidth;
			stickend[0] = biorigin[0]+c*bigridwidth;
			stickend[1] = biorigin[1]+d*bigridwidth;

			

			if(!this->_IsTwoSegmentIntersect(startpntcoord[0], startpntcoord[2], midpntcoord[0], midpntcoord[2], stickstart[0], stickstart[1], stickend[0], stickend[1]))
			{
				firstregionnum = this->_RecursiveLocalVSA(parapatch, startnode, midpnt, currregionnum, biorigin, bigridwidth);
			}
		}
		
	}
	
	int speciallimit;
	if(midpnt->GetIndexNo() > endnode->GetIndexNo())
		speciallimit = endnode->GetIndexNo()+parapatch->GetVSAEdgeList().GetCount()-1;
	else
		speciallimit = endnode->GetIndexNo();


	for(Pos=parapatch->GetVSANodeList().GetHeadPosition(); Pos!=NULL; )
	{
		node = (VSANode*)parapatch->GetVSANodeList().GetNext(Pos);

		if (node->GetIndexNo()>=midpnt->GetIndexNo()+1 && node->GetIndexNo() < speciallimit)
		{
			node->GetStick(a, b, c, d);
			stickstart[0] = biorigin[0]+a*bigridwidth;
			stickstart[1] = biorigin[1]+b*bigridwidth;
			stickend[0] = biorigin[0]+c*bigridwidth;
			stickend[1] = biorigin[1]+d*bigridwidth;

			if(!this->_IsTwoSegmentIntersect(midpntcoord[0], midpntcoord[2], endpntcoord[0], endpntcoord[2], stickstart[0], stickstart[1], stickend[0], stickend[1]))
			{
				secondregionnum = this->_RecursiveLocalVSA(parapatch, midpnt, endnode, currregionnum, biorigin, bigridwidth);
			}
		}
	}
	
	//increase region index to the number bigger than current region number by 1 or 2
	if(firstregionnum==1)
	{
		for(Pos=tempatch->GetVSAEdgeList().GetHeadPosition(); Pos!=NULL; )
		{
			temedge = (VSAEdge *)tempatch->GetVSAEdgeList().GetNext(Pos);
			if(temedge->RegionIndex==firstregionind)
			{
				temedge->RegionIndex = currregionnum;
			}
		}
	}
	currregionnum++;
		
	if(secondregionnum==1)
	{
		for(Pos=tempatch->GetVSAEdgeList().GetHeadPosition(); Pos!=NULL; )
		{
			temedge = (VSAEdge *)tempatch->GetVSAEdgeList().GetNext(Pos);
			if(temedge->RegionIndex==secondregionid)
			{
				temedge->RegionIndex = currregionnum;
			}
		}
	}
	currregionnum++;

	tempatch->GetVSAEdgeList().RemoveAll();
	delete tempatch;

	//printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~leave Recursive~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n");
	return firstregionnum+secondregionnum;

}

double VSA::DistortionErrorCorrection(double desireddistterror)
{
	if(m_RegionNum==1 || m_RegionNum==2)
		return 0.0;

	GLKGeometry geo;

	VSAEdge *tempedge = (VSAEdge *)v_mesh->GetVSAEdgeList().GetHead();
	VSAEdge *firstedge = tempedge;
	int lastregionindex = firstedge->RegionIndex;

	double lastpntcoord[3];
	double currpntcoord[3];
	double v1[3];
	double v2[3];
	double fakeproxy[3];
	double length;
	double regiondistterror;
	VSANode *startingnode;
	VSANode *lastnode;
	VSANode *currentnode;
	GLKPOSITION startedgepos;
	GLKPOSITION endedgepos;
	int count=0;	

	double largerdistterror = 0.0;
	for(GLKPOSITION Pos=v_mesh->GetVSAEdgeList().GetHeadPosition(); Pos!=NULL; )
	{
		tempedge = (VSAEdge *)v_mesh->GetVSAEdgeList().GetNext(Pos);
		if(tempedge->RegionIndex!=lastregionindex)
		{
			count++;
			if(count==1)
			{
				lastnode = tempedge->GetStartPoint();

				startingnode = lastnode;
				lastregionindex = tempedge->RegionIndex;
			}
			else
			{
				currentnode = tempedge->GetStartPoint();
				currentnode->GetCoord3D(currpntcoord[0], currpntcoord[1], currpntcoord[2]);
				lastnode->GetCoord3D(lastpntcoord[0], lastpntcoord[1], lastpntcoord[2]);
				geo.CalLineEquation(fakeproxy[0], fakeproxy[1], fakeproxy[2], lastpntcoord[0], lastpntcoord[2], currpntcoord[0], currpntcoord[2]);

				tempedge = (VSAEdge *)lastnode->GetVSAEdgeList().GetTail();
				startedgepos = v_mesh->GetVSAEdgeList().Find(tempedge);
				tempedge = (VSAEdge *)currentnode->GetVSAEdgeList().GetHead();
				endedgepos = v_mesh->GetVSAEdgeList().Find(tempedge);

				regiondistterror = 0.0;
				GLKPOSITION Pos;
				for(Pos=startedgepos; Pos!=endedgepos; )
				{
					tempedge = (VSAEdge *)v_mesh->GetVSAEdgeList().GetNext(Pos);
					tempedge->GetStartPoint()->GetCoord3D(v1[0], v1[1], v1[2]);
					tempedge->GetEndPoint()->GetCoord3D(v2[0], v2[1], v2[2]);
					length = sqrt((v1[0]-v2[0])*(v1[0]-v2[0])+(v1[2]-v2[2])*(v1[2]-v2[2]));
					regiondistterror += this->EvaluateDistortionError2D(fakeproxy, v1[0], v1[2], v2[0], v2[2], length);
				}
				tempedge = (VSAEdge *)v_mesh->GetVSAEdgeList().GetNext(Pos);
				tempedge->GetStartPoint()->GetCoord3D(v1[0], v1[1], v1[2]);
				tempedge->GetEndPoint()->GetCoord3D(v2[0], v2[1], v2[2]);
				length = sqrt((v1[0]-v2[0])*(v1[0]-v2[0])+(v1[2]-v2[2])*(v1[2]-v2[2]));
				regiondistterror += this->EvaluateDistortionError2D(fakeproxy, v1[0], v1[2], v2[0], v2[2], length);

				if(regiondistterror>desireddistterror)
				{
					this->_RecursiveLocalVSAForDistterror(v_mesh, lastnode, currentnode, fakeregionnum,desireddistterror, largerdistterror);
					/*std::cout<<"distortion error correction performed"<<std::endl;*/
				}
				else
				{
					if(regiondistterror>largerdistterror)
						largerdistterror = regiondistterror;
				}

				lastnode = currentnode;
				lastregionindex = tempedge->RegionIndex;
			}
		}
	}
	return largerdistterror;


}

void VSA::SimplifyMeshBasedOnVSARegions2D()
{
	VSAEdge *tempedge = (VSAEdge *)v_mesh->GetVSAEdgeList().GetHead();
	VSAEdge *firstedge = tempedge;
	int lastregionindex = firstedge->RegionIndex;

	VSAMesh *newpatch = new VSAMesh();

	double lastpntcoord[3];
	double currpntcoord[3];
	VSANode *startingnode;
	VSANode *lastnode;
	VSANode *currentnode;
	VSAEdge *newtempedge;
	int count=0;	

	if(m_RegionNum==1 || m_RegionNum==2)
	{
		for(GLKPOSITION Pos=v_mesh->GetVSAEdgeList().GetHeadPosition(); Pos!=NULL; )
		{
			tempedge = (VSAEdge *)v_mesh->GetVSAEdgeList().GetNext(Pos);
			count++;
			if(count==1)
			{
				tempedge->GetStartPoint()->GetCoord3D(lastpntcoord[0], lastpntcoord[1], lastpntcoord[2]);
				lastnode = new VSANode();
				lastnode->SetCoord3D(lastpntcoord[0], lastpntcoord[1], lastpntcoord[2]);

				startingnode = lastnode;
				lastregionindex = tempedge->RegionIndex;
			}
			else
			{
				currentnode = new VSANode();
				tempedge->GetStartPoint()->GetCoord3D(currpntcoord[0], currpntcoord[1], currpntcoord[2]);
				currentnode->SetCoord3D(currpntcoord[0], currpntcoord[1], currpntcoord[2]);

				newtempedge = new VSAEdge();
				newtempedge->SetStartPoint(lastnode);
				newtempedge->SetEndPoint(currentnode);
				newtempedge->RegionIndex = lastregionindex;

				lastnode->GetVSAEdgeList().AddTail(newtempedge);
				currentnode->GetVSAEdgeList().AddTail(newtempedge);

				newpatch->GetVSAEdgeList().AddTail(newtempedge);
				newpatch->GetVSANodeList().AddTail(lastnode);

				lastnode = currentnode;
				lastregionindex = tempedge->RegionIndex;
			}
		}
		newpatch->GetVSANodeList().AddTail(lastnode);
	}
	else
	{
		for(GLKPOSITION Pos=v_mesh->GetVSAEdgeList().GetHeadPosition(); Pos!=NULL; )
		{
			tempedge = (VSAEdge *)v_mesh->GetVSAEdgeList().GetNext(Pos);
			if(tempedge->RegionIndex!=lastregionindex /*true*/) //Warning: this may be changed when you want to cancle VSA segmentation
			{
				count++;
				if(count==1)
				{
					tempedge->GetStartPoint()->GetCoord3D(lastpntcoord[0], lastpntcoord[1], lastpntcoord[2]);
					lastnode = new VSANode();
					lastnode->SetCoord3D(lastpntcoord[0], lastpntcoord[1], lastpntcoord[2]);

					startingnode = lastnode;
					lastregionindex = tempedge->RegionIndex;
				}
				else
				{
					currentnode = new VSANode();
					tempedge->GetStartPoint()->GetCoord3D(currpntcoord[0], currpntcoord[1], currpntcoord[2]);
					currentnode->SetCoord3D(currpntcoord[0], currpntcoord[1], currpntcoord[2]);

					newtempedge = new VSAEdge();
					newtempedge->SetStartPoint(lastnode);
					newtempedge->SetEndPoint(currentnode);
					newtempedge->RegionIndex = lastregionindex;

					lastnode->GetVSAEdgeList().AddTail(newtempedge);
					currentnode->GetVSAEdgeList().AddTail(newtempedge);

					newpatch->GetVSAEdgeList().AddTail(newtempedge);
					newpatch->GetVSANodeList().AddTail(lastnode);

					lastnode = currentnode;
					lastregionindex = tempedge->RegionIndex;
				}
			}
		}

		//construct connection segment(s)
		if(tempedge->RegionIndex==firstedge->RegionIndex)
		{
			newtempedge = new VSAEdge();
			newtempedge->SetStartPoint(lastnode);
			newtempedge->SetEndPoint(startingnode);
			newtempedge->RegionIndex = lastregionindex;

			lastnode->GetVSAEdgeList().AddTail(newtempedge);
			startingnode->GetVSAEdgeList().AddTail(newtempedge);

			newpatch->GetVSAEdgeList().AddTail(newtempedge);
			newpatch->GetVSANodeList().AddTail(lastnode);
		}
		else
		{
			VSANode *firstnode = new VSANode();
			double firstcoord[3];
			firstedge->GetStartPoint()->GetCoord3D(firstcoord[0], firstcoord[1], firstcoord[2]);
			firstnode->SetCoord3D(firstcoord[0], firstcoord[1], firstcoord[2]);

			//first connection segment
			newtempedge = new VSAEdge();
			newtempedge->SetStartPoint(lastnode);
			newtempedge->SetEndPoint(firstnode);
			newtempedge->RegionIndex = lastregionindex;

			lastnode->GetVSAEdgeList().AddTail(newtempedge);
			firstnode->GetVSAEdgeList().AddTail(newtempedge);

			newpatch->GetVSAEdgeList().AddTail(newtempedge);
			newpatch->GetVSANodeList().AddTail(lastnode);

			//second connection segment
			newtempedge = new VSAEdge();
			newtempedge->SetStartPoint(firstnode);
			newtempedge->SetEndPoint(startingnode);
			newtempedge->RegionIndex = firstedge->RegionIndex;

			firstnode->GetVSAEdgeList().AddTail(newtempedge);
			startingnode->GetVSAEdgeList().AddTail(newtempedge);

			newpatch->GetVSAEdgeList().AddTail(newtempedge);
			newpatch->GetVSANodeList().AddTail(firstnode);
		}
	}

	//set index for entities in new patch
	count=0;
	for(GLKPOSITION Pos=newpatch->GetVSAEdgeList().GetHeadPosition(); Pos!=NULL; )
	{
		tempedge = (VSAEdge *)newpatch->GetVSAEdgeList().GetNext(Pos);
		tempedge->SetIndexNo(count);
		count++;
	}
	count=0;
	VSANode *tempnode;
	VSANode *firstnode;
	for(GLKPOSITION Pos=newpatch->GetVSANodeList().GetHeadPosition(); Pos!=NULL; )
	{
		tempnode = (VSANode *)newpatch->GetVSANodeList().GetNext(Pos);
		if(count==0)
			firstnode = tempnode;
		tempnode->SetIndexNo(count);
		count++;
	}
	firstnode->GetCoord3D(currpntcoord[0], currpntcoord[1], currpntcoord[2]);
	VSANode *newnode = new VSANode();
	newnode->SetCoord3D(currpntcoord[0], currpntcoord[1], currpntcoord[2]);
	newnode->SetIndexNo(count);
	newpatch->GetVSANodeList().AddTail(newnode);

	v_mesh = newpatch;
}
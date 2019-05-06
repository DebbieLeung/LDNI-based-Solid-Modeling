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

#ifndef	_CCL_LDNI_CPU_SOLID
#define	_CCL_LDNI_CPU_SOLID

#include "../GLKLib/GLK.h"

#define NULL	0

typedef struct LDNIcpuSample {
	float depth,nx,ny,nz;
	bool flag;
}LDNIcpuSample;

typedef struct LDNIcpuRay {
	unsigned int sampleIndex;	// the starting index of the first sample on NEXT ray
	bool rayflag;				// the status of the ray 
}LDNIcpuRay;

class LDNIcpuSolid 
{
public:
	LDNIcpuSolid();
	virtual ~LDNIcpuSolid();

	void MallocMemory(int res);
	void FreeMemory();

	bool FileSave(char *filename);
	bool FileRead(char *filename);

	void ExpansionByNewBoundingBox(float boundingBox[]);

	int GetSampleNumber();

	void SetOrigin(float ox, float oy, float oz) {m_origin[0]=ox; m_origin[1]=oy; m_origin[2]=oz;};
	void GetOrigin(float &ox, float &oy, float &oz) {ox=m_origin[0]; oy=m_origin[1]; oz=m_origin[2];};
	int GetResolution() {return m_res;};

	void SetSampleWidth(float width) {m_sampleWidth=width;};
	double GetSampleWidth() {return m_sampleWidth;};

	LDNIcpuRay* GetRayArrayPtr(short nDir) {return m_Rays[nDir];};
	LDNIcpuSample* GetSampleArrayPtr(short nDir) {return m_SampleArray[nDir];};
	void SetSampleArrayPtr(short nDir, LDNIcpuSample *arrayPtr) {m_SampleArray[nDir]=arrayPtr;};

	void MallocSampleMemory(short nDir, int sampleNum);

private:
	LDNIcpuRay *m_Rays[3];
	// Note that: "m_Rays[][m_res*m_res-1].sampleIndex" specifies the total number of samples in m_SampleArray[]
	//			and "m_Rays[][i-1].sampleIndex" specify the index of starting sample on the i-th ray
	LDNIcpuSample *m_SampleArray[3];
	
	float m_origin[3],m_sampleWidth;	int m_res;
};

class LDNIcudaSolid;

class LDNISolidBody : public GLKEntity
{
public:
	LDNISolidBody();
	virtual ~LDNISolidBody();

	void BuildGLList(bool bWithArrow);
	void DeleteVBOGLList();
	void SetUpVBO(int i, bool m_cuda) {m_vboSize = i; m_cudaRegistered = m_cuda;};

	void CompRange();

	virtual void drawShade();
	virtual void drawProfile();
	virtual void drawMesh() {};
	virtual void drawPreMesh() {};
	virtual void drawHighLight() {};
	virtual float getRange() {return m_range;}

	void SetLighting(bool bLight) {m_Lighting=bLight;};
	bool GetLighting() {return m_Lighting;};

	void SetRange(float r) {m_range = r;};
	float GetRange() {return m_range;};

	LDNIcpuSolid *m_solid;
	LDNIcudaSolid *m_cudaSolid;
	GLuint	m_vbo;
	

private:
	bool m_Lighting;	int m_drawListID;	double m_range;
	int m_vboSize;
	GLuint m_vboPosition;	GLuint m_vboNormal;		int m_vertexNum;	bool m_cudaRegistered;
};

#endif
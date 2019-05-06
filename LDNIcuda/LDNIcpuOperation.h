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


#ifndef	_CCL_LDNI_CPU_OPERATION
#define	_CCL_LDNI_CPU_OPERATION

#include "LDNIcpuSolid.h"

#define LOGIC_UNION(insideA, insideB)	(insideA || insideB)
#define LOGIC_INTER(insideA, insideB)	(insideA && insideB)
#define LOGIC_SUBTR(insideA, insideB)	(insideA && (!insideB))

class QuadTrglMesh;

class LDNIcpuOperation
{
public:
	LDNIcpuOperation();
	~LDNIcpuOperation();

	static bool BooleanOperation(LDNIcpuSolid* &inputSolid, QuadTrglMesh *meshB, short nOperationType);
	static bool BooleanOperation(QuadTrglMesh *meshA, QuadTrglMesh *meshB, int res, short nOperationType, LDNIcpuSolid* &solid);
	static bool BRepToLDNISampling(QuadTrglMesh *mesh, LDNIcpuSolid* &solid, float boundingBox[], int res);

	static void ParallelProcessingNormalVector(LDNIcpuSolid *solid, unsigned int nSupportSize=3, float normalPara=0.15f);

private:
	//-----------------------------------------------------------------------------------------------------
	//	Functions for Boolean operations
	static void _booleanOperation(LDNIcpuSolid* solidA, LDNIcpuSolid* solidB, short nOperationType);
	static void _compBoundingCube(QuadTrglMesh *meshA, QuadTrglMesh *meshB, float boundingBox[], int res);
	static void _booleanOnRay(LDNIcpuRay *rayA, LDNIcpuSample *sampleArrA, LDNIcpuRay *rayB, LDNIcpuSample *sampleArrB, 
																	int index, int *resArrayIndex, short nOperationType);

	//-----------------------------------------------------------------------------------------------------
	//	Functions for sampling a B-rep into LNDIcpuSolid
	static void _decomposeLDNIByFBOPBO(LDNIcpuSolid *solid, int displayListIndex);
	static void _sortingSamplesOnRay(LDNIcpuSample *sampleArray, unsigned int stIndex, unsigned int size);
	static bool _shaderInitialization();
	static void _texCalProduct(int in, int &outx, int &outy);
	static unsigned char* _readShaderFile(const char *fileName);

	static void _copyCPUSolidToCUDAEmulArray(LDNIcpuSolid *cpuSolid, 
												   unsigned int* &xIndexArray, unsigned int* &yIndexArray, unsigned int* &zIndexArray,
												   float* &xNxArray, float* &yNxArray, float* &zNxArray, 
												   float* &xNyArray, float* &yNyArray, float* &zNyArray, 
												   float* &xDepthArray, float* &yDepthArray, float* &zDepthArray );
};

#endif
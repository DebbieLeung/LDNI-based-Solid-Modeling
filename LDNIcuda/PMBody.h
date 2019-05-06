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


#ifndef _CCL_POLYMESH_BODY
#define _CCL_POLYMESH_BODY

#include "../GLKLib/GLK.h"
#include "../GLKLib/GLKObList.h"
#include "../GLKLib/GLKHeap.h"
#include "../GLKLib/GLKGeometry.h"
#include "../GLKLib/GLKMatrixLib.h"


class VSAEdge;
//----SGM code by KaChun----//
class ContourMesh;
//----SGM code by KaChun----//

class VSANode : public GLKObject
{
public:
	VSANode(void);
	virtual ~VSANode(void);

	void GetCoord3D( double &x, double &y, double &z ) {x = coord3D[0]; y = coord3D[1]; z = coord3D[2];};
	void SetCoord3D( double x, double y, double z ) {coord3D[0] = x; coord3D[1] = y; coord3D[2] = z;};

	int GetIndexNo() { return indexno; };
	void SetIndexNo( const int _index = 1 ) { indexno=_index; };

	//int GetArrIndexNo() { return arrindexno; };
	//void SetArrIndexNo( const int _index = 1 ) { arrindexno=_index; };

	void SetStickDir( int x, int z ) { stickDir[0] = x; stickDir[1] = z; };
	int GetStickDirX() {return stickDir[0];};
	int GetStickDirZ() {return stickDir[1];};

	void SetStick(int st_x, int st_z, int ed_x, int ed_z) { stickst[0] = st_x; stickst[1] = st_z; sticked[0] = ed_x; sticked[1] = ed_z; };
	void GetStick(int &st_x, int &st_z, int &ed_x, int &ed_z) { st_x = stickst[0]; st_z = stickst[1]; ed_x = sticked[0]; ed_z = sticked[1];};

	void AddEdge(VSAEdge *vedge);
	GLKObList& GetVSAEdgeList() {return VSAEdgeList;};


	void SetNormal(double nx, double ny, double nz );
	void GetNormal(double &nx, double &ny, double &nz);
	
private:
	double coord3D[3];
	double normal[3];
	GLKObList VSAEdgeList;
	int  indexno;
	//int  arrindexno;
	int stickDir[2];
	int stickst[2];
	int sticked[2];
	 

};

class VSAEdge : public GLKObject
{
public:
	VSAEdge(void);
	virtual ~VSAEdge(void);

	VSANode * GetStartPoint() {return pStartPoint;};
	void SetStartPoint( VSANode * _pStartPoint = NULL ) {pStartPoint = _pStartPoint;};

	VSANode * GetEndPoint() {return pEndPoint;};
	void SetEndPoint( VSANode * _pEndPoint = NULL ){pEndPoint = _pEndPoint;};

	void CalLength();
	double GetLength() {return length; };

	void SetRegionIndex(int i) {RegionIndex = i;};
	int GetRegionIndex() {return RegionIndex;};

	unsigned int GetIndexNo() {return index;}; 
	void SetIndexNo( unsigned int _index) {index = _index;};

	unsigned int GetPrevIndexNo() {return prev_edge_index;};
	void SetPrevIndexNo(unsigned int _index) {prev_edge_index = _index;};

	int RegionIndex;
private:
	VSANode * pStartPoint;		
	VSANode * pEndPoint;
	
	double length;
	unsigned int index;
	unsigned int prev_edge_index;
};


class VSAMesh : public GLKObject
{
public:
	VSAMesh(void);
	virtual ~VSAMesh(void);

	void ClearAll();
	unsigned int GetIndexNo() {return index;}; 
	void SetIndexNo( unsigned int _index) {index = _index;};

	GLKObList& GetVSAEdgeList() {return VSAEdgeList;};
	GLKObList& GetVSANodeList() {return VSANodeList;};

	GLKObject* FindEdgeListByIndex(unsigned int i);
	GLKObject* FindPrevEdgeListByIndex(unsigned int i);
	void FindNextAndPrevEdgeListByIndex(VSAEdge* stpt_connected_edge, VSAEdge* edpt_connected_edge, unsigned int prev, unsigned int next);

private:
	GLKObList VSAEdgeList;
	GLKObList VSANodeList;
	unsigned int index;
};

class QuadTrglMesh : public GLKObject
{
public:
	QuadTrglMesh(void);
	virtual ~QuadTrglMesh(void);

	void ClearAll();
	
	void MallocMemory(int nodeNum, int faceNum);
	void SetNodePos(int nodeIndex/*starting from 1*/, float pos[]); 
	void SetFaceNodes(int faceIndex/*starting from 1*/, unsigned int verIndex1, unsigned int verIndex2, unsigned int verIndex3, unsigned int verIndex4);

	int GetFaceNumber();
	int GetNodeNumber();
	bool IsQuadFace(int faceIndex/*starting from 1*/);
	void GetFaceNodes(int faceIndex/*starting from 1*/, unsigned int &verIndex1, unsigned int &verIndex2, unsigned int &verIndex3, unsigned int &verIndex4);
	void GetNodePos(int nodeIndex/*starting from 1*/, float pos[]);
	float *GetNodeTablePtr() {return m_nodeTable;};
	unsigned int *GetFaceTablePtr() {return m_faceTable;};

	void CompNormal(int faceIndex/*starting from 1*/, float nv[]);
	void CompBoundingBox(float boundingBox[]);

	bool InputOBJFile(char *filename);
	bool OutputOBJFile(char *filename);

	bool InputSTLFile(char *filename);

	bool InputMEBFile(char *filename);	// the binary file of QUAD/TRGL mesh object
	bool OutputMEBFile(char *filename);	// the binary file of QUAD/TRGL mesh object

	bool InputOBJFileFromMapping(char *filedata);
	bool InputSTLFileFromMapping(char *filedata);

	float* GetNodeArrayPtr() {return m_nodeTable;};

	void SetMeshId(short i) {meshID = i;};
	short GetMeshId() {return meshID;};
	void SetMeshUpdateStatus(bool status) { bUpdate = status;};
	bool GetMeshUpdateStatus() {return bUpdate;};

	void FlipModel(bool nDir_X, bool nDir_Y, bool nDir_Z);
	void Transformation(float dx, float dy, float dz);
	void Scaling(float sx, float sy, float sz);
	void ShiftToOrigin();
	void ShiftToPosSystem();

private:
	int m_nodeNum, m_faceNum;
	float *m_nodeTable;	
	short meshID;
	bool bUpdate;
	unsigned int *m_faceTable;	//	Note that: the index starts from '1'
								//		when '0' is shown in the face table, it means that the vertex is not defined
};



class VSAHeapNode : public GLKHeapNode
{
public:
	VSAHeapNode();
	virtual ~VSAHeapNode();

public:
	int whichproxyagainst;
};


class VSA2DNode
{
public:
	VSA2DNode();
	~VSA2DNode();

public:
	int ProxyIndicator;
	double ct_pt[2];		//note: for generality, we just use 3d coord to represent 2d point with the coordinate in y direction being 0 
	double area;
	void *meshobj;
};


class VSA
{
public:
	VSA();
	~VSA();
	VSAMesh *v_mesh;

	void InitializeVSA(VSAMesh *mesh, int RegionNum, short Dimension);
	void PerformVSA2D(int iterNum, double desiredDistortError, bool needContourCheck);		//the parameter needcontourbdcheck is for whether you are dealing with closed or open contour
	void BinaryImageInOutCorrection(double *biorigin, double bigridwidth);
	double DistortionErrorCorrection(double desireddistterror);
	void SimplifyMeshBasedOnVSARegions2D();

	int layerInd;
private:
	
	short m_dimension;
	int m_RegionNum;
	int m_FaceNum;

	VSA2DNode **m_Nodes;
	double **m_Proxies;
	int fakeregionnum;

	double VSACoreIterations(int maxiter, GLKHeap *heap, double ***covmatrixarray, double *totalareaforregions, 
							double *regioncenterx, double *regioncenterz, double *mindisttforregions,
							double *regiondistortionerrors, VSA2DNode **tempvsanodelist, int &maxdistterrorregion,
							int &maxdisttvsanodeind, bool needcontourbdcheck);
	float EvaluateDistortionError2D(double *proxy, double v1, double v2, double u1, double u2, double length);
	bool _IsTwoSegmentIntersect(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4);
	int _RecursiveLocalVSA(VSAMesh *parapatch,  VSANode *startnode, VSANode *endnode, int &currregionnum, double *biorigin, double bigridwidth);
	int _RecursiveLocalVSAForDistterror(VSAMesh *parapatch, VSANode *startnode, VSANode *endnode, int &currregionnum, double desireddistterror, double &largerdistterror);
	
};

//----SGM code by KaChun----//
class Contour: public GLKObject
{
public:
	Contour();
	~Contour();

	void PrintContoursInSGMFormat(FILE *sgm_output);

	int pntNum;		//the vertices number for an arbitrary polygon
	double *xp;
	double *yp;

	double xmax;
	double xmin;
	double ymax;
	double ymin;
	double area;
	double perimeter;

	GLKObList includingcontours;

	bool printexplitcitly;
	bool partorsppt;		//true means part material while false means support material
};

class Layer
{
public:
	Layer();
	~Layer();

	void BuildTopologyOfContours();

	int contourNum;
	Contour **contourarray;
	double height;
	double thickness;
};

class RPFilesInterface
{
public:
	RPFilesInterface();
	~RPFilesInterface();

	bool OutputInsightSGMFile(ContourMesh* c_mesh,const char *filename);

	//void BuildTopologyForAllLayers();

public:
	double thickness;
	int lyrNum;
	Layer *layerarray;
};
//----SGM code by KaChun----//
class ContourMesh : public GLKEntity
{
public:
	ContourMesh(void);
	virtual ~ContourMesh(void);

	typedef struct RGB {
		unsigned char Blue;
		unsigned char Green;
		unsigned char Red;
	}; 

	void ClearAll();
	virtual float getRange() {return 10.0;}
	void MallocMemory(unsigned int* ContourNum, int imageSize[], int stickNum);
	
	float* GetStartNodeArrayPtr() {return m_StnodeTable;};
	float* GetEndNodeArrayPtr() {return m_EdnodeTable;};
	int* GetContourNumPtr() {return  m_ContourNum;};
	GLKObList& GetVSAMeshList() {return VSAMeshList;};


	void ArrayToContour(float* st_stick, float* ed_stick, unsigned int* id_stick);
	void ArrayToContour(float* st_stick, float* ed_stick, double imgOri[], float imgWidth);
	//void ConvertContourToVSAMesh(float* st_stick, float* ed_stick, int* stickID, int stickNum);
	void BuildContourTopology(float* st_stick, float* ed_stick, int* stickID, int stickNum, int* stickDir);
	void ArrayToImage(bool *nodes, int imageSize[]);
	void ArrayToImage(bool *outNodes, bool *InNodes, int imageSize[], int base, bool bSave, bool bDisplay);
	void WriteBMP(const char *filename, GLubyte* data, int m_SizeX, int m_SizeY);

	//void PerformVSA2D(int* stickDir);
	void PerformVSA3D();
	double PerformVSA2D(VSAMesh* vmesh, int iter, double paradistterror, int simpratio);

	int GetResolution(short i) {return iRes[i];};

	void SetThickness(float t) {thickness = t;};
	float GetThickness() {return thickness;};
	int GetTotalStickNum() {return totalstickNum;}
	int GetLayerStickNum(int layer) {return m_ContourNum[layer];}
	void SetOrigin(float a, float b, float c) {origin[0] = a; origin[1] = b; origin[2] = c;};
	void SetMeshId(short i) {meshID = i;};
	short GetMeshId() {return meshID;};
	void SetMeshUpdateStatus(bool status) { bUpdate = status;};
	bool GetMeshUpdateStatus() {return bUpdate;};
	void setRange(float r) {m_range = r;};
	void SetImageResolution(int res[]) {iRes[0] = res[0]; iRes[1] = res[1]; iRes[2] = res[2];};

	void setImageOrigin(double imgorix, double imgoriy) { imageOrigin[0] = imgorix; imageOrigin[1] = imgoriy; };
	void setSampleWidth(float s) { sampleWidth = s;};
	//float getRange() {return m_range;}

	int GetContourNum() {return VSAMeshNum;};
	bool m_drawImage;
	GLuint *tex;

	void Output_SGM_FILE(ContourMesh *supt_mesh);

	//----SGM code by KaChun----//
	//original private
	int iRes[3];
	unsigned int *ContourNum;
	float thickness;
	GLKObList VSAMeshList;
	//----SGM code by KaChun----//

private:
	int* m_ContourNum;
	float *m_StnodeTable;	
	float *m_EdnodeTable;	
	short meshID;
	int totalstickNum;
	bool bUpdate;
	float origin[3];
	float m_range;
	int VSAMeshNum;
	double imageOrigin[2];
	float sampleWidth;

	//unsigned int *m_faceTable;	//	Note that: the index starts from '1'
	
};


class PMBody : public GLKEntity
{
public:
	PMBody(void);
	virtual ~PMBody(void);

	void DeleteGLList(bool bShadeOrMesh);
	void BuildGLList(bool bShadeOrMesh, bool bContourList = false);


	virtual void drawShade();
	virtual void drawMesh();
	virtual void drawProfile();
	virtual void drawPreMesh();
	virtual void drawHighLight();
	virtual void drawContour();
	virtual void Update();
	virtual float getRange() {return m_range;}
	virtual void setRange(float r) {m_range = r;};

	void drawBox(float xx, float yy, float zz, float r);

	void ClearAll();
	void computeRange();
	
	GLKObList &GetMeshList() {return meshList;};

	void FlipModel(short nDir);
	void CompBoundingBox(float boundingBox[]);
	void Transformation(float dx, float dy, float dz);
	void ShiftToOrigin();
	void ShiftToPosSystem();
	void Scaling(float sx, float sy, float sz);

	void SetPMUpdate(bool bupdate) {bUpdate = bupdate;}; //UI tell that it has changed the content
	bool GetPMUpdate() {return bUpdate;};
	

private:
	GLKObList meshList;
	float m_range;
	int m_drawShadingListID;
	int m_drawMeshListID;
	int m_drawContourID;
	

	void _buildDrawShadeList();
	void _buildDrawMeshList();
	void _buildContourList();
	void _changeValueToColor(int nType, float & nRed, float & nGreen, float & nBlue);
};



#endif
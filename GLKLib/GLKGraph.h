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


// GLKGraph.h: interface for the GLKGraph class.
//
//////////////////////////////////////////////////////////////////////

#include "GLKObList.h"

#ifndef _CW_GLKGRAPHNODE
#define _CW_GLKGRAPHNODE

class GLKGraphNode : public GLKObject
{
public:
	GLKGraphNode() {edgeList.RemoveAll();attachedObj=NULL;};
	virtual ~GLKGraphNode() {};
	void *attachedObj;
	GLKObList edgeList;

	//---------------------------------------------------------------
	//	the following variables are for minimum cut
	double m_excess;
	int m_height;
	GLKGraphNode *nextNode;
};

#endif

#ifndef _CW_GLKGRAPHEDGE
#define _CW_GLKGRAPHEDGE

class GLKGraphEdge : public GLKObject
{
public:
	GLKGraphEdge() {startNode=NULL;	endNode=NULL;	m_weight=0.0;};
	virtual ~GLKGraphEdge() {};

	GLKGraphNode* startNode;
	GLKGraphNode* endNode;
	double m_weight;
	void *attachedObj;

	//---------------------------------------------------------------
	//	the following variables are for minimum cut
	double m_flow;
};

#endif

#ifndef _CW_GLKGRAPH
#define _CW_GLKGRAPH

class GLKGraphCutNode;

class GLKGraph  
{
public:
	GLKGraph();
	virtual ~GLKGraph();

	void AddNode(GLKGraphNode *node);
	void AddEdge(GLKGraphEdge *edge);
	void FillInEdgeLinkersOnNodes();

	void _Debug();

	//---------------------------------------------------------------------
	//	The following function is implemented by the relabel-to-front algorithm
public:
	double MinimumCut(GLKGraphNode *sourceNode, GLKGraphNode *targetNode, 
			GLKObList *sourceRegionNodeList, GLKObList *targetRegionNodeList, 
			bool bComputeMaxFlow=false);
private:
	void _initializePreflow(GLKGraphNode *sourceNode);
	void _discharge(GLKGraphNode *uNode);
	void _push(GLKGraphNode *uNode, GLKGraphEdge *edge);
	void _relable(GLKGraphNode *uNode);
	void _partitionByResidualGraph(GLKGraphNode *sourceNode, GLKGraphNode *targetNode, 
			GLKObList *sourceRegionNodeList, GLKObList *targetRegionNodeList);
	void _propagateInResidualGraph(GLKGraphNode *node, GLKObList *regionNodeList);
	double _computeMaxFlow();

private:
	void clearAll();

	GLKObList nodeList;
	GLKObList edgeList;
};

#endif

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

uniform samplerRect vertexTexture;
uniform int sizeNx;
uniform vec3 Cent;

void main( void )
{
	int ix,iy;
	
	iy=gl_Vertex.x/sizeNx;		ix=gl_Vertex.x-iy*sizeNx;
	gl_Position.xyz = texture2DRect(vertexTexture,vec2(ix,iy)).rgb-Cent;
	gl_Position.w = 1.0;
	
	iy=gl_Vertex.y/sizeNx;		ix=gl_Vertex.y-iy*sizeNx;
	gl_FrontColor.xyz = texture2DRect(vertexTexture,vec2(ix,iy)).rgb-Cent;
	gl_FrontColor.w = 1.0;
	
	iy=gl_Vertex.z/sizeNx;		ix=gl_Vertex.z-iy*sizeNx;
	gl_BackColor.xyz = texture2DRect(vertexTexture,vec2(ix,iy)).rgb-Cent;
	gl_BackColor.w = 1.0;
	
	gl_PointSize=gl_Vertex.w;
}
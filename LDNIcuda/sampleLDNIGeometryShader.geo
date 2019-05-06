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


#version 120 
#extension GL_EXT_geometry_shader4: enable

varying out vec4 color;

vec4 CalPlaneEq(vec3 P0, vec3 P1, vec3 P2)
{
	vec4 result;
	
	result.x = P0.y * ( P1.z - P2.z ) + P1.y * ( P2.z - P0.z ) + P2.y * ( P0.z - P1.z );
	result.y = P0.z * ( P1.x - P2.x ) + P1.z * ( P2.x - P0.x ) + P2.z * ( P0.x - P1.x );
	result.z = P0.x * ( P1.y - P2.y ) + P1.x * ( P2.y - P0.y ) + P2.x * ( P0.y - P1.y );
	result.w = - P0.x * ( P1.y * P2.z - P2.y * P1.z ) - P1.x * ( P2.y * P0.z - P0.y * P2.z ) - P2.x * ( P0.y * P1.z - P1.y * P0.z );

	float  tt = length(result.xyz);
	if (tt < 0.00000001) return vec4(0.0,0.0,0.0,0.0);

	result = result/tt;

	return result;
}

void main( void )
{
	color = CalPlaneEq(gl_BackColorIn[0].xyz, gl_PositionIn[0].xyz, gl_FrontColorIn[0].xyz);
	
//	if ((color.x != 0.0)  || (color.y != 0.0) || (color.z != 0.0)) 
	{ 
		gl_Position = gl_ModelViewProjectionMatrix*gl_PositionIn[0];
		EmitVertex();
		gl_Position = gl_ModelViewProjectionMatrix*gl_FrontColorIn[0];
		EmitVertex();
		gl_Position = gl_ModelViewProjectionMatrix*gl_BackColorIn[0];
		EmitVertex();	
		EndPrimitive();
	}
}

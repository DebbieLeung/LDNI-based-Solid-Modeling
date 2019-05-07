# LDNI-based-Solid-Modeling

This project provides a set of solid modeling tools which aims to help processing 3D objects with complex topology and geometry that are widely used in many industrial applications (e.g., microstructure design and manufacturing, biomedical products and applications, jewelry products, reverse engineering).

<p float="middle">
  <img src="http://ldnibasedsolidmodeling.sourceforge.net/image/Mickey%20intersect%20with%20scaffold.jpg" width="256" />
  <img src="http://ldnibasedsolidmodeling.sourceforge.net/image/bunny_dragon_toolpath.jpg" width="256" /> 
  <img src="http://ldnibasedsolidmodeling.sourceforge.net/image/bunny_dragon_toolpath_1.jpg" width="256" />
</p>

The system is completely GPU-based and heavily based on an implicit representation named LDNI.

Features
- Input/Output : Obj files
- Boolean Operations - Union, Intersection and Substraction
- Offset
- Scaffold
- Super Union - union multiple meshes in one operation
- 3D Printing
    - Generate tool path and supporting files for FDM
    - Generate stencil images for SLA

### [Old Project page](http://ldnibasedsolidmodeling.sourceforge.net/) | [Youtube 1](https://youtu.be/G75mS1VGqx0) | [Youtube 2](https://youtu.be/go3MxEF7cOs)
    
## Dependencies
* CUDA 9.0

## Usage
|   | Operation     | Description          |
|---|:---: |--------|
| 1 | <img src="http://ldnibasedsolidmodeling.sourceforge.net/image/Boolean_Lion_Sub_Mickey.jpg" width="160" /><img src="http://ldnibasedsolidmodeling.sourceforge.net/image/Boolean_Lion_Sub_Mickey_LDNI.jpg" width="160" /> | **Boolean (A Subtract B)**  <br> 1) File > Open  > Choose A.obj  <br> 2) File > Open  > Choose B.obj  <br> 3) LDNI > Cuda Boolean Operations > input "d" for subtraction <br> 4) Input sampling resolution e.g. 512, 1024, 2048  <br><br>  **Boolean (A Subtract B Union C)** <br> 5) File > Open  > Choose C.obj <br> 6) LDNI > Cuda Boolean Operations > input "u" for union  <br><br>  **Export LDNI to OBJ file** <br> 7) LDNI > Convert CUDA solid to mesh <br> 8) Input resolution e.g. 512|
| 2 |<img src="http://ldnibasedsolidmodeling.sourceforge.net/image/offset_bone.jpg" width="160" /><img src="http://ldnibasedsolidmodeling.sourceforge.net/image/offset_bone_negative.jpg" width="160" /> |**Offset (A)** <br>  1) File > Open > Choose A.obj <br> 2) LDNI > Cuda samlping (from B-rep) > input resolution e.g. 512, 1024, 2048 <br> 3) LDNI > Cuda Successive Offseting (with Normal) > input e.g. +10, -10|
| 3 |<img src="http://ldnibasedsolidmodeling.sourceforge.net/image/unit.jpg" width="160" /><img src="http://ldnibasedsolidmodeling.sourceforge.net/image/scaffold.jpg" width="160" />|**Scaffold (A)** <br> 1) File > Open > Choose A.obj <br> 2) LDNI >CUDA Scaffold Making > input resolution e.g. 512, 1024, 2048 <br> 3) Input the repeated number of unit along x, y and z axis <br>  4) Input the offset value (the interval distance between each unit along x, y and z axis) <br> 5) Input the flipping flag for each direction ( 1- flip and 0 - no flip) <br><br> **Example** <br> <img src="http://ldnibasedsolidmodeling.sourceforge.net/flip.jpg"  width="180"/><br>e.g. repeated number < 2 1 1><br>offset value <0 0 0><br>flip flag <1 0 0>|
| 4 |<img src="http://ldnibasedsolidmodeling.sourceforge.net/image/super_union.jpg" width="160" /><img src="http://ldnibasedsolidmodeling.sourceforge.net/image/super_union_ldni.jpg" width="160" />|**Super Union (A + B + C....)**<br>1) File > Open  Folder > Choose folder which stored all the components <br>2) LDNI > Cuda Super Union > input resolution e.g. 1024, 2048, 4096|
| 5 |<img src="http://ldnibasedsolidmodeling.sourceforge.net/image/FDM.jpg" width="160" /><img src="http://ldnibasedsolidmodeling.sourceforge.net/image/FDM_oneSlice.jpg" width="160" />|**FDM (output toolpath file)**<br>1) File > Open > Choose A.obj <br> 2) Mesh > Shift To Origin <br> 3) Mesh > Shift To Positive Coordinate System <br> 4) LDNI > Cuda samlping (from B-rep) > input resolution e.g. 512, 1024, 2048 <br> 5) LDNI > Generate Contour and Support on CUDA (FDM) > input image sampling width [6] e.g. 0.005|
| 6 |<img src="http://ldnibasedsolidmodeling.sourceforge.net/image/SLA.jpg" width="160" /><img src="http://ldnibasedsolidmodeling.sourceforge.net/image/SLA_oneslice.jpg" width="160" />|**SLA (output image file)** <br> 1) File > Open > Choose A.obj <br> 2) Mesh > Shift To Origin <br> 3) Mesh > Shift To Positive Coordinate System <br> 4) LDNI > Cuda samlping (from B-rep) > input resolution e.g. 512, 1024, 2048 <br> 5) LDNI > Generate Contour and Support on CUDA (FDM) > input image sampling width  e.g. 0.005 <br><br> **Input parameters for generate supporting structure [6]** <br> 6) input Thickness [thickness of layer] e.g. 0.004 <br> 7) input Anchor Radius [radius of support anchor] e.g. 0.2 <br> 8) input Threshold [self-support feature threshold] e.g. 0.2 <br> 9) input Cylinder Radius [radius of support cylinder] e.g. 1.0 <br> 10 input Pattern Thickness [thickness of the connection structure between support cylinder] e.g. 10 <br> * the numbers are varied,  depends upon the fabrication volume and the machine|

## References
1. Charlie C.L. Wang, Yuen-Shan Leung, and Yong Chen, "Solid modeling of polyhedral objects by Layered Depth-Normal Images on the GPU", Computer-Aided Design, vol.42, no.6, pp.535-544, June 2010. [[link]](https://www.sciencedirect.com/science/article/pii/S0010448510000278)

2. Yuen-Shan Leung, and Charlie C.L. Wang, "Conservative sampling of solids in image space", IEEE Computer Graphics and Applications, vol.33, no.1, pp.14-25, January/February, 2013. [[link]](https://ieeexplore.ieee.org/document/6415478)

3. Charlie C.L. Wang, and Dinesh Manocha, "GPU-based offset surface computation using point samples", Computer-Aided Design, Special Issue of 2012 Symposium on Solid and Physical Modeling, October 29-31, 2012, Dijon, France, vol.45, no.2, pp.321-330, February 2013. [[link]](https://www.sciencedirect.com/science/article/pii/S0010448512002205)

4. Shengjun Liu, and Charlie C.L. Wang, "Fast intersection-free offset surface generation from freeform models with triangular meshes", IEEE Transactions on Automation Science and Engineering, vol.8, no.2, pp.347-360, April 2011. [[link]](https://ieeexplore.ieee.org/document/5570949)

5. Pu Huang, Charlie C.L. Wang, and Yong Chen, "Intersection-free and topologically faithful slicing of implicit solid", ASME Transactions - Journal of Computing and Information Science in Engineering, vol.13, no.2, 021009 (13 pages), June 2013. [[link]](http://computingengineering.asmedigitalcollection.asme.org/article.aspx?articleid=1682448)

6. Pu Huang, Charlie C.L. Wang, and Yong Chen, "Algorithms for layered manufacturing in image space", Book Chapter, ASME Advances in Computers and Information in Engineering Research, 2014.[[link]](http://ebooks.asmedigitalcollection.asme.org/content.aspx?bookid=1348&sectionid=73139817)

7. Hanli Zhao, Charlie C.L. Wang, Yong Chen, and Xiaogang Jin, "Parallel and efficient Boolean on polygonal solids", The Visual Computer, Special Issue of Computer Graphics International 2011 (CGI 2011), vol.27, no.6-8, pp.507-517, Ottawa, Ontario, Canada, June 12-15, 2011. [[link]](https://link.springer.com/article/10.1007/s00371-011-0571-1)

8. Supplementary Technicanl Report - "Intersection-free dual contouring on uniform grids: an approach based on convex/concave analysis" [[link]](http://www.mae.cuhk.edu.hk/~cwang/pubs/TRIntersectionFreeDC.pdf)






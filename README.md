Metric-based Curve Clustering and Feature Extraction (CAD && CG 2017 short paper, August 2017)

Author: Lieyu Shi
Email: shilieyu91@gmail.com


---------------------------------------------------------------------
###########    Objective     ###########
########################################
	This is metric-based clustering for both streamlines and particle trajectories.

	We propose several linear-complexity metric enabling an efficient and scalable feature extraction and curve clustering for large-scale fluid simulation data.


---------------------------------------------------------------------
###########    Compilation and Running     ###########
######################################################
	mkdir dataset (in ${CMAKE_SOURCE_DIR}$)
	move your dataset into dataset
	./build.sh
	cd Release
	./cluster datasetName 3(dimensionality)


---------------------------------------------------------------------
###########    Dataset Format     ###########
#############################################
	Each line is a high-dimension streamline or trajectory (pathline), and format is (assume 3D vertex array consisting of line)
	x1 y1 z1 x2 y2 z2 x3 y3 z3 ...

	This code adopts an automatic filling strategy to create an equal-size matrix for metric computation. 

---------------------------------------------------------------------
###########    Output Result    ###########
###########################################
	Would output .vtk format for centroids of each cluster, closest and furthest streamlines to centroids of each cluster.

	Visualization is based on ParaView (https://www.paraview.org/)


---------------------------------------------------------------------
#####    Metric Validity and Clustering Evaluation     ####
###########################################################
	Besides visual comparions, we also provide Entropy comparison for clustering result.

	Stringline query is also provided in main.cpp for metric validity.

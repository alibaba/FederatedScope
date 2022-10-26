README for dataset ENZYMES


=== Usage ===

This folder contains the following comma separated text files 
(replace DS by the name of the dataset):

n = total number of nodes
m = total number of edges
N = number of graphs

(1) 	DS_A.txt (m lines) 
	sparse (block diagonal) adjacency matrix for all graphs,
	each line corresponds to (row, col) resp. (node_id, node_id)

(2) 	DS_graph_indicator.txt (n lines)
	column vector of graph identifiers for all nodes of all graphs,
	the value in the i-th line is the graph_id of the node with node_id i

(3) 	DS_graph_labels.txt (N lines) 
	class labels for all graphs in the dataset,
	the value in the i-th line is the class label of the graph with graph_id i

(4) 	DS_node_labels.txt (n lines)
	column vector of node labels,
	the value in the i-th line corresponds to the node with node_id i

There are OPTIONAL files if the respective information is available:

(5) 	DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt)
	labels for the edges in DS_A_sparse.txt 

(6) 	DS_edge_attributes.txt (m lines; same size as DS_A.txt)
	attributes for the edges in DS_A.txt 

(7) 	DS_node_attributes.txt (n lines) 
	matrix of node attributes,
	the comma seperated values in the i-th line is the attribute vector of the node with node_id i

(8) 	DS_graph_attributes.txt (N lines) 
	regression values for all graphs in the dataset,
	the value in the i-th line is the attribute of the graph with graph_id i


=== Description === 

ENZYMES is a dataset of protein tertiary structures obtained from (Borgwardt et al., 2005) 
consisting of 600 enzymes from the BRENDA enzyme database (Schomburg et al., 2004). 
In this case the task is to correctly assign each enzyme to one of the 6 EC top-level 
classes. 


=== Previous Use of the Dataset ===

Feragen, A., Kasenburg, N., Petersen, J., de Bruijne, M., Borgwardt, K.M.: Scalable
kernels for graphs with continuous attributes. In: C.J.C. Burges, L. Bottou, Z. Ghahra-
mani, K.Q. Weinberger (eds.) NIPS, pp. 216-224 (2013)

Neumann, M., Garnett R., Bauckhage Ch., Kersting K.: Propagation Kernels: Efficient Graph 
Kernels from Propagated Information. Under review at MLJ.


=== References ===

K. M. Borgwardt, C. S. Ong, S. Schoenauer, S. V. N. Vishwanathan, A. J. Smola, and H. P. 
Kriegel. Protein function prediction via graph kernels. Bioinformatics, 21(Suppl 1):i47–i56, 
Jun 2005.

I. Schomburg, A. Chang, C. Ebeling, M. Gremse, C. Heldt, G. Huhn, and D. Schomburg. Brenda, 
the enzyme database: updates and major new developments. Nucleic Acids Research, 32D:431–433, 2004.

from torch_geometric.datasets import Planetoid
from federatedscope.core.splitters.graph import LouvainSplitter, Analyzer

raw_dataset = Planetoid('data', 'cora')
split_dataset = Planetoid('data', 'cora', transform=LouvainSplitter(3))

raw_data, split_data = raw_dataset[0], split_dataset[0]

# Print information of raw_data and split_data.
print(f'raw_data: {raw_data}.\n split_data: {split_data}.\n')

# Build analyzer
analyzer = Analyzer(raw_data, split_data)

# Print the number of missing edge and the missing rate after split. (
# Undirected)
num_missing_edge, missing_rate = analyzer.num_missing_edge()
print(f'num_missing_edge: {num_missing_edge}, missing_rate: {missing_rate}.\n')

# Print the merge data of splits
merge_data = analyzer.fl_data()
print(f'merge_data: {merge_data}.\n')

# Print the average clustering coef of raw graph and split graph
avg_cls_raw, avg_cls_split = analyzer.average_clustering()
print(f'average_clustering_raw: {avg_cls_raw}, average_clustering_merged:'
      f' {avg_cls_split}.\n')

# Print the homophilic value of raw graph and split graph
raw_homo, split_homo = analyzer.homophily()
print(f'homophilic_value_raw: {raw_homo}, homophilic_value_split:'
      f' {split_homo}.\n')

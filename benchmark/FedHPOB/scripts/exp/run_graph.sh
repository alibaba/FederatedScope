# ******Cora*****

# tabular
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data cora optimizer.type rs
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data cora optimizer.type bo_gp
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data cora optimizer.type bo_rf
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data cora optimizer.type bo_kde
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data cora optimizer.type de

python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data cora optimizer.type hb
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data cora optimizer.type bohb
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data cora optimizer.type dehb
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data cora optimizer.type tpe_md
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data cora optimizer.type tpe_hb

# raw
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data cora optimizer.type rs benchmark.device 0
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data cora optimizer.type bo_gp benchmark.device 1
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data cora optimizer.type bo_rf benchmark.device 2
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data cora optimizer.type bo_kde benchmark.device 3
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data cora optimizer.type de benchmark.device 4

python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data cora optimizer.type hb benchmark.device 5
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data cora optimizer.type bohb benchmark.device 6
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data cora optimizer.type dehb benchmark.device 7
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data cora optimizer.type tpe_md benchmark.device 7
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data cora optimizer.type tpe_hb benchmark.device 6

# surrogate
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data cora optimizer.type rs
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data cora optimizer.type bo_gp
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data cora optimizer.type bo_rf
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data cora optimizer.type bo_kde
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data cora optimizer.type de

python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data cora optimizer.type hb
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data cora optimizer.type bohb
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data cora optimizer.type dehb
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data cora optimizer.type tpe_md
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data cora optimizer.type tpe_hb


# ******CiteSeer*****

# tabular
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data citeseer optimizer.type rs
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data citeseer optimizer.type bo_gp
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data citeseer optimizer.type bo_rf
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data citeseer optimizer.type bo_kde
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data citeseer optimizer.type de

python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data citeseer optimizer.type hb
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data citeseer optimizer.type bohb
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data citeseer optimizer.type dehb
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data citeseer optimizer.type tpe_md
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data citeseer optimizer.type tpe_hb

# raw
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data citeseer optimizer.type rs benchmark.device 0
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data citeseer optimizer.type bo_gp benchmark.device 1
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data citeseer optimizer.type bo_rf benchmark.device 2
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data citeseer optimizer.type bo_kde benchmark.device 3
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data citeseer optimizer.type de benchmark.device 4

python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data citeseer optimizer.type hb benchmark.device 5
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data citeseer optimizer.type bohb benchmark.device 6
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data citeseer optimizer.type dehb benchmark.device 7
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data citeseer optimizer.type tpe_md benchmark.device 7
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data citeseer optimizer.type tpe_hb benchmark.device 6

# surrogate
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data citeseer optimizer.type rs
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data citeseer optimizer.type bo_gp
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data citeseer optimizer.type bo_rf
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data citeseer optimizer.type bo_kde
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data citeseer optimizer.type de

python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data citeseer optimizer.type hb
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data citeseer optimizer.type bohb
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data citeseer optimizer.type dehb
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data citeseer optimizer.type tpe_md
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data citeseer optimizer.type tpe_hb


# ******Pubmed*****

# tabular
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data pubmed optimizer.type rs
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data pubmed optimizer.type bo_gp
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data pubmed optimizer.type bo_rf
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data pubmed optimizer.type bo_kde
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data pubmed optimizer.type de

python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data pubmed optimizer.type hb
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data pubmed optimizer.type bohb
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data pubmed optimizer.type dehb
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data pubmed optimizer.type tpe_md
python runner.py --cfg scripts/exp/graph.yaml benchmark.type tabular benchmark.data pubmed optimizer.type tpe_hb

# raw
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data pubmed optimizer.type rs benchmark.device 0
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data pubmed optimizer.type bo_gp benchmark.device 1
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data pubmed optimizer.type bo_rf benchmark.device 2
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data pubmed optimizer.type bo_kde benchmark.device 3
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data pubmed optimizer.type de benchmark.device 4

python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data pubmed optimizer.type hb benchmark.device 5
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data pubmed optimizer.type bohb benchmark.device 6
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data pubmed optimizer.type dehb benchmark.device 7
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data pubmed optimizer.type tpe_md benchmark.device 7
python runner.py --cfg scripts/exp/graph.yaml benchmark.type raw benchmark.data pubmed optimizer.type tpe_hb benchmark.device 6

# surrogate
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data pubmed optimizer.type rs
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data pubmed optimizer.type bo_gp
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data pubmed optimizer.type bo_rf
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data pubmed optimizer.type bo_kde
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data pubmed optimizer.type de

python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data pubmed optimizer.type hb
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data pubmed optimizer.type bohb
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data pubmed optimizer.type dehb
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data pubmed optimizer.type tpe_md
python runner.py --cfg scripts/exp/graph.yaml benchmark.type surrogate benchmark.data pubmed optimizer.type tpe_hb

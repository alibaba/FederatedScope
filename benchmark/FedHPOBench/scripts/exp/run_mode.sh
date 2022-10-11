set -e

dataset=$1
mode=$2
model=$3
device=$4
algo=$5

cd ../..
cp fedhpob/utils/runner.py . || echo "File exists."

for k in {1..5}; do
  python runner.py --cfg scripts/exp/${dataset}.yaml benchmark.device ${device} benchmark.model ${model} benchmark.type ${mode} benchmark.data ${dataset} benchmark.algo ${algo} optimizer.type rs || echo "continue"
  python runner.py --cfg scripts/exp/${dataset}.yaml benchmark.device ${device} benchmark.model ${model} benchmark.type ${mode} benchmark.data ${dataset} benchmark.algo ${algo} optimizer.type bo_gp || echo "continue"
  python runner.py --cfg scripts/exp/${dataset}.yaml benchmark.device ${device} benchmark.model ${model} benchmark.type ${mode} benchmark.data ${dataset} benchmark.algo ${algo} optimizer.type bo_rf || echo "continue"
  python runner.py --cfg scripts/exp/${dataset}.yaml benchmark.device ${device} benchmark.model ${model} benchmark.type ${mode} benchmark.data ${dataset} benchmark.algo ${algo} optimizer.type bo_kde || echo "continue"
  python runner.py --cfg scripts/exp/${dataset}.yaml benchmark.device ${device} benchmark.model ${model} benchmark.type ${mode} benchmark.data ${dataset} benchmark.algo ${algo} optimizer.type de || echo "continue"
  python runner.py --cfg scripts/exp/${dataset}.yaml benchmark.device ${device} benchmark.model ${model} benchmark.type ${mode} benchmark.data ${dataset} benchmark.algo ${algo} optimizer.type grid_search || echo "continue"

  python runner.py --cfg scripts/exp/${dataset}.yaml benchmark.device ${device} benchmark.model ${model} benchmark.type ${mode} benchmark.data ${dataset} benchmark.algo ${algo} optimizer.type hb || echo "continue"
  python runner.py --cfg scripts/exp/${dataset}.yaml benchmark.device ${device} benchmark.model ${model} benchmark.type ${mode} benchmark.data ${dataset} benchmark.algo ${algo} optimizer.type bohb || echo "continue"
  python runner.py --cfg scripts/exp/${dataset}.yaml benchmark.device ${device} benchmark.model ${model} benchmark.type ${mode} benchmark.data ${dataset} benchmark.algo ${algo} optimizer.type dehb || echo "continue"
  python runner.py --cfg scripts/exp/${dataset}.yaml benchmark.device ${device} benchmark.model ${model} benchmark.type ${mode} benchmark.data ${dataset} benchmark.algo ${algo} optimizer.type tpe_md || echo "continue"
  python runner.py --cfg scripts/exp/${dataset}.yaml benchmark.device ${device} benchmark.model ${model} benchmark.type ${mode} benchmark.data ${dataset} benchmark.algo ${algo} optimizer.type tpe_hb || echo "continue"
done

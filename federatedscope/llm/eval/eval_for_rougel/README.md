# Rouge-L

## Dolly-15K
To assess the performance of our fine-tuned model, we leverage the Rouge-L 
metric and conduct experiments with a large number of clients, utilizing the Dolly-15K dataset as our training corpus. 
The Dolly-15K dataset encompasses a total of 15,015 data points, distributed across eight distinct tasks. For a more comprehensive evaluation, we allocate the final task exclusively for evaluation purposes, while dedicating the remaining ones to the training phase. Our experimental setup involves a network of 200 clients, utilizing a Dirichlet distribution for data partitioning to emulate non-IID conditions across the client base.

To do the evaluation, run
```bash
python federatescope/eval/eval_for_rougel/eval_dolly.py --cfg federatescope/llm/baselime/xxx.yaml
```

## Natural Instructions
We also leverage the Rouge-L metric and conduct experiments with a large number of clients, utilizing the Natural Instructions (NI) dataset as our training corpus.  In the NI dataset, we allocate each of the 738 training tasks exclusively to a distinct client for model training, thereby cultivating a non-IID setting characterized by feature distribution skew. Meanwhile, evaluation is performed on separate test tasks.

To do the evaluation, run
```bash
python federatescope/eval/eval_for_rougel/eval_ni.py --cfg federatescope/llm/baselime/xxx.yaml
```
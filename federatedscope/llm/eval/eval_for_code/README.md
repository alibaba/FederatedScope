# HumanEval Usage

* Using the trained model to generate codes from prompt, and save them as a `jsonl` file.
  * `python federatedscope/llm/eval/eval_for_code/humaneval.py --cfg federatedscope/llm/baseline/llama.yaml`
  * The file name of `jsonl` should be `{cfg.federate.save_to}_humaneval_answer.jsonl`
* Use HumanEval tools to test the pass@k score
  * Installation
    * `git clone https://github.com/openai/human-eval`
    * `pip install -e human-eval`
    * uncomment the following line 59 in `human-eval/human_eval/execution.py`
      * `exec(check_program, exec_globals)`
  * Evaluate
    * `evaluate_functional_correctness {cfg.federate.save_to}_humaneval_answer.jsonl`

# HumanEvalX Usage

* Using the trained model to generate codes from prompt, and save them as 5 `jsonl` files (`['cpp', 'go', 'java', 'js', 'python']`).

  * `python federatedscope/llm/eval/eval_for_code/humanevalx.py --cfg federatedscope/llm/baseline/llama.yaml`

  * The file name of `jsonl` should be `{cfg.federate.save_to}_humanevalx_{LANGUAGE}_answer.jsonl`

* Use HumanEvalX Docker Image to test the pass@k score

  * `docker pull rishubi/codegeex:latest`

  * ```bash
    docker run -it --mount type=bind,source=$PWD,target=/workspace/fs rishubi/codegeex:latest /bin/bash -c "cd CodeGeeX; git fetch; git pull; pip install -e .; \
    bash scripts/evaluate_humaneval_x.sh ../fs/{cfg.federate.save_to}_humanevalx_cpp_answer.jsonl cpp 1; \
    bash scripts/evaluate_humaneval_x.sh ../fs/{cfg.federate.save_to}_humanevalx_go_answer.jsonl go 1; \
    bash scripts/evaluate_humaneval_x.sh ../fs/{cfg.federate.save_to}_humanevalx_java_answer.jsonl java 1; \
    bash scripts/evaluate_humaneval_x.sh ../fs/{cfg.federate.save_to}_humanevalx_js_answer.jsonl js 1; \
    bash scripts/evaluate_humaneval_x.sh ../fs/{cfg.federate.save_to}_humanevalx_python_answer.jsonl python 1; exit"
    ```

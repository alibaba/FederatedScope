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
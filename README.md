# CORE
The official code of paper "CORE: Lossless Compression for Retrieval-Augmented LLMs via Reinforcement Learning"

## Deploy the QA LLM using vLLM

1.  **Start the vLLM Service:**
    First, run `reward_llm_serve.sh` to deploy the QA LLM (Large Language Model). This model is responsible for generating answers, which are subsequently used for reward calculation.

2.  **Configure the Reward Service:**
    Please update the `verl_core/reward.yaml` configuration file with your specific vLLM deployment IP address and the model name.

3.  **Update the Configuration Path:**
    Then, navigate to line 105 in the file `verl_core/verl/workers/reward_manager/api_prime.py`, where you will find the following line of code:
    ```python
    config_path = "verl_core/reward.yaml"
    ```
    Ensure this `config_path` variable points to the correct location of your modified `reward.yaml` file.

## Start RL Training

sh examples/grpo_trainer/run_qwen2.5-1.5b_compress_nq.sh

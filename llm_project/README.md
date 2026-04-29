# LLM Project

This repository contains components for testing different LLMs and performing task decomposition tasks.

## Environment Setup

To run the scripts in this project, you should have Python 3.10 or create a Conda environment with this Python version, then install the required packages using the provided `setup.py`.

1. **Create the Conda environment (Optional)**:
   ```bash
   conda create -n comp514-fp python=3.10
   conda activate comp514-fp
   ```

2. **Install the project packages**:
   ```bash
   pip install -e .
   ```

## Testing LLMs

The script `test_llms.py` is used to test and evaluate the language models. You can check the file at `llm_project/llm/test_llms.py` to see the logic.

To run the script, you can specify arguments such as the model, prompting strategy (shots), and specific tasks:

```bash
# Example 1: Run Qwen model with zero-shot prompting
python llm_project/llm/test_llms.py --models qwen --shots zero

# Example 2: Run all models with all shot configs and save to 'results/'
python llm_project/llm/test_llms.py --models all --shots zero one few --output results/

# Example 3: Run SmolLM2 with zero-shot on specific tasks (0 and 1)
python llm_project/llm/test_llms.py --models smollm2 --shots zero --tasks 0 1
```

## Testing Task Decomposition

The script `test_decomposition.py` is used to generate the JSON output you will later use with the selected LLM model. You can find it at `llm_project/llm/test_decomposition.py`.

To run the script:
```bash
python llm_project/llm/test_decomposition.py --task "<instruction>"
```

## Extra Credits World

To launch the world, you must first build the package (`colcon build --packages-select llm_project`). After building, add the world path:
```bash
    export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:[path_to_llm_project]/models
    export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:[path_to_llm_project]/worlds
```
And run the launch command:
```bash
ros2 launch llm_project gen3_tabletop.launch.py
```

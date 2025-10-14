# Dynamic Capacity MoE (Preview)

Dynamic-Capacity Mixture-of-Experts (DCMoE) improves efficiency and adaptability in large models by introducing **dynamic expert allocation** and a flexible expert design.

 **Dynamic-Capacity Routing**: Instead of using a fixed Top-K strategy, tokens are routed with a **Top-P mechanism** that selects the minimum number of experts needed to reach a probability threshold. This allows simple tokens to use fewer experts while complex ones leverage more, reducing wasted computation.  
 **Shared Experts**: A small set of always-active experts capture common knowledge across all inputs, offloading the routed experts to specialize in domain-specific patterns.  
 **Routed Experts**: Conditionally activated experts that focus on specialized knowledge, guided by the dynamic router.  
 **Null Experts**: Parameter-free placeholders that output zeros, enabling true computation skipping for trivial tokens and extending the range of adaptive expert usage.  

## Installation

To set up the environment and install the required dependencies, please follow these steps:

```bash
# Create and activate a new conda environment
conda create -n dcmoe python=3.9
conda activate dcmoe

# Install the required packages
pip install -r requirements.txt
```

## Prepare Your Data

We provide a dataset example in the `./data` directory to help you get started. You can inspect the data format with the following Python code:

```python
import datasets

# Load the dataset from the disk
data = datasets.load_from_disk("./data")

# Print the dataset information
print(data)
```

## Training

We provide a training script in the `./script` directory. You can start the training process by running this script directly. For detailed explanations of each training argument, please refer to the comments within the script.

```bash
bash ./script/training.sh
```

## Inference

In the `./inference` directory, we provide scripts for two different inference modes to accommodate various hardware setups and VRAM constraints:

**Single-GPU Inference:** A fast inference method on a single GPU without expert parallelism (**the model must training without expert parallelism**).

```bash
python ./inference/single_gpu_without_ep.py
```

**Multi-GPU Inference:** An expert-parallel inference method that leverages multiple GPUs, ideal for larger models or setups with distributed resources.

```bash
deepspeed --master_addr "localhost" --master_port 9042 ./inference/multi_gpu_with_ep.py
```


## Training Arguments

The training script `train_unimoev2_qwen2vl.py` accepts several arguments to configure the model architecture, training process, and data paths.

### MoE Specific Arguments

These arguments are unique to our Mixture-of-Experts (MoE) model configuration and are primarily used when initializing an MoE model from a dense model (`--initialize True`).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--initialize` | `bool` | If `True`, initializes a new MoE model from a pre-trained dense model. If `False`, loads an existing MoE model checkpoint directly. Defaults to `False`. |
| `--moe_copy` | `str` | **(Only effective if `--initialize True`)**<br>Specifies the initialization strategy for MoE experts. <br>- `none`: Experts are not initialized from the dense model. <br>- `all`: All experts are initialized using the weights of the Feed-Forward Network (FFN) from the dense model. |
| `--fp32_gate` | `bool` | **(Only effective if `--initialize True`)**<br>If `True`, the gating network (router) will perform its computations in `float32` precision for better stability, even if the rest of the model is in `bf16` or `fp16`. |
| `--ep_size` | `int` | **(Only effective if `--initialize True`)**<br>Expert Parallelism size. This is the number of GPUs across which the experts are sharded. It must be a divisor of `--mlp_dynamic_expert_num`. |
| `--mlp_dynamic_expert_num` | `int` | The total number of "routed" experts (i.e., the experts chosen by the gating network). |
| `--mlp_fixed_expert_num` | `int` | **(Only effective if `--initialize True`)**<br>The number of "shared" or "fixed" experts. These experts process every token and are not part of the routing mechanism. |
| `--mlp_dynamic_null_expert_num` | `int` | **(Only effective if `--initialize True`)**<br>The number of "null" experts. These are special experts that can be used to handle tokens that don't have a high affinity for any other expert. This count is separate from `--mlp_dynamic_expert_num`. |
| `--dynamic_mlp_size_factor` | `int` | **(Only effective if `--initialize True`)**<br>The reduction factor for the intermediate size of the "routed" experts' FFNs, relative to the original dense model's FFN intermediate size. |
| `--fixed_mlp_size_factor` | `int` | **(Only effective if `--initialize True`)**<br>The reduction factor for the intermediate size of the "shared" experts' FFNs. |
| `--mlp_dynamic_top_p` | `float` | **(Only effective if `--initialize True`)**<br>If set to a value other than `0` (e.g., `0.7`), enables Top-P (nucleus) routing. The router will select the smallest set of experts whose cumulative probability is greater than this threshold. If `0`, Top-K routing is used instead. |
| `--mlp_dynamic_top_k` | `int` | **(Only effective if `--initialize True`)**<br>The value of `k` for Top-K routing. The router selects the top `k` experts for each token. This is only active when `--mlp_dynamic_top_p` is `0`. |
| `--token_drop` | `bool` | **(Only effective if `--initialize True`)**<br>If `True`, enables the token dropping mechanism to manage expert capacity. |
| `--drop_policy` | `str` | **(Only effective if `--token_drop True`)**<br>The policy for dropping tokens when an expert's capacity is exceeded. Options are `probs` or `position`. |
| `--min_capacity` | `int` | **(Only effective if `--token_drop True`)**<br>The minimum capacity for each expert, ensuring it can process at least this many tokens. |
| `--capacity_factor` | `float` | **(Only effective if `--token_drop True`)**<br>A factor used to calculate the total capacity of each expert. |
| `--l_aux_weight` | `float` | **(Only effective if `--initialize True`)**<br>The total weight for the auxiliary load balancing loss (`L_aux`), which encourages tokens to be distributed evenly across experts. |
| `--aux_balance_weight` | `float` | An additional weight applied to the auxiliary loss for output tokens. This can help stabilize training by putting more emphasis on balancing the load of important tokens. |

### Path and Directory Arguments

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--model_name_or_path` | `str` | Path to the model to be loaded. If `--initialize` is `True`, this should be the path to the pre-trained **dense** model. If `False`, this should be the path to an existing **MoE** model checkpoint. |
| `--processor_path` | `str` | Path to the pre-trained processor (tokenizer and image processor). |
| `--data_path` | `str` | Path to the JSON file containing the training data annotations. |
| `--image_root` | `str` | The root directory where the image files are stored. |
| `--output_dir` | `str` | The directory where model checkpoints and training outputs will be saved. |
| `--deepspeed` | `str` | Path to the DeepSpeed configuration file (e.g., `deepspeed_zero2.conf`). |

### Standard Training Arguments

These are standard arguments from the Hugging Face `Trainer`.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--attn_implementation` | `str` | The attention implementation to use. `sdpa` (Scaled Dot Product Attention) is recommended for efficiency with recent PyTorch versions. |
| `--bf16` | `bool` | Whether to use bfloat16 mixed-precision training. This speeds up training and reduces memory usage on compatible hardware (e.g., Ampere GPUs). |
| `--num_train_epochs` | `int` | The total number of training epochs to perform. |
| `--per_device_train_batch_size` | `int` | The batch size per GPU for training. |
| `--per_device_eval_batch_size` | `int` | The batch size per GPU for evaluation. |
| `--gradient_accumulation_steps` | `int` | The number of steps to accumulate gradients before performing an optimizer step. Effective batch size = `per_device_train_batch_size * num_gpus * gradient_accumulation_steps`. |
| `--learning_rate` | `float` | The initial learning rate for the AdamW optimizer. |
| `--weight_decay` | `float` | The weight decay (L2 regularization) to apply. |
| `--warmup_ratio` | `float` | The proportion of total training steps used for a linear learning rate warmup. |
| `--lr_scheduler_type` | `str` | The learning rate scheduler to use (e.g., `cosine`, `linear`). |
| `--logging_steps` | `int` | Log training metrics every `n` steps. |
| `--save_strategy` | `str` | The checkpoint saving strategy. `steps` saves at a regular interval. |
| `--save_steps` | `int` | Save a checkpoint every `n` steps. |
| `--save_total_limit` | `int` | The maximum number of checkpoints to keep. Older checkpoints will be deleted. |
| `--model_max_length` | `int` | The maximum sequence length for the model. Longer sequences will be truncated. |
| `--gradient_checkpointing` | `bool` | If `True`, enables gradient checkpointing to save memory at the cost of a slight slowdown in training. |
| `--dataloader_num_workers` | `int` | The number of worker processes for data loading. |
| `--evaluation_strategy` | `str` | The evaluation strategy. `no` means no evaluation is performed during training. |
| `--report_to` | `str` | The integration to report results to (e.g., `wandb`, `tensorboard`). `none` disables reporting. |
| `--run_name` | `str` | An optional name for the training run, useful for logging and tracking. |


# Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts (8 experts)
[Yunxin Li](https://yunxinli.github.io), [Shenyuan Jiang](URL), [Baotian Hu](https://faculty.hitsz.edu.cn/hubaotian), [Longyue Wang](http://www.longyuewang.com/), [Wanqi Zhong](URL), [Wenhan Luo](https://whluo.github.io/), [Lin Ma](https://forestlinma.com/), [Min Zhang](https://faculty.hitsz.edu.cn/MinZhang)

</h4>


Uni-MoE-8e represents our latest iteration of MoE-based unified multimodal model with 8 experts, designed to adeptly manage a spectrum of modalities such as audio, speech, images, text, and video. This cutting-edge framework boasts enhanced capabilities for multi-GPU training and inferencing, significantly accelerating the optimization process and expanding the scale of our model.

## 🌟 Structure

The model architecture of Uni-MoE is shown below. Three training stages contain: 1) Utilize pairs from different modalities and languages to build connectors that map these elements to a unified language space, establishing a foundation for multimodal understanding; 2) Develop modality-specific experts using cross-modal data to ensure deep understanding, preparing for a cohesive multi-expert model; 3) Incorporate multiple trained experts into LLMs and refine the unified multimodal model using the LoRA technique on mixed multimodal data. 

In the 8 expert edition of our model, we have integrated the DeepSpeed MoE architecture to facilitate the efficient distribution of experts' weights across various GPUs during both training and testing phases. This strategic design ensures balanced load allocation and enhanced parallel processing capabilities.Furthermore, we have introduced a novel LoRA integrated MLP to optimize the distribution mechanism to reduces computational complexity while ensuring the distribution function of DeepSpeed MOE still works.

<div align=center><img src="https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/model.png" height="100%" width="75%"/></div>

## ⚡️ Install

The following instructions are for Linux installation.
We would like to recommend the requirements as follows.
* Python == 3.9.16
* CUDA Version >= 11.7

1. Clone this repository and navigate to the Uni_MoE_8e folder
```bash
git clone https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs.git
cd UMOE-Scaling-Unified-Multimodal-LLMs/Uni_MoE/Uni_MoE_8e
```

2. Install Package
```Shell
conda create -n unimoe_8e python==3.9.16
conda activate unimoe_8e
pip install -r env.txt
conda install mpi4py
pip install flash-attn==2.5.6
pip install moviepy
```

3. Replace all the absolute pathnames '/path/to/' or '/data/' with your specific path to the Uni-MoE file

**(Including all the eval_x.py/inference_x.py/train_mem_x.py/data.py/demo.py files and config.json files from the model weights)**

## ⚡️ Uni-MOE Weights

To use our new version model, all weights should be downloaded.

After downloading all of them, organize the weights as follows in 'Uni_MoE/Uni_MoE_8e/checkpoint' folder:
```
└── checkpoint
    ├── Uni_MoE_8e_Experts
    ├── Uni-MoE-speech-base
    ├── Uni_MoE_8e_e2
    ├── clip-vit-large-patch14-336
    ├── whisper-small
    └── BEATs_iter3_plus_AS2M.pt
```
| Model  | Checkpoint |
|----------|-----------|
| vision encoder | [CLIP ViT-L/14 336px](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main) |
| speech encoder | [whisper small](https://huggingface.co/openai/whisper-small/tree/main) |
| audio encoder  | [BEATs_iter3+ (AS2M)](https://1drv.ms/u/s!AqeByhGUtINrgcpke6_lRSZEKD5j2Q?e=A3FpOf) |
| Uni-MoE 8-expert base | [Uni-MoE-speech-base](https://huggingface.co/Uni-MoE/Uni-MoE-speech-base) |
| Uni_MoE 8-expert experts | [Uni_MoE_8e_Experts](https://huggingface.co/Uni-MoE/Uni-MoE-8e-Experts) |
| Uni_MoE 8-expert finetune model | [Uni_MoE_8e_e2](https://huggingface.co/Uni-MoE/Uni-MoE-8e-e2) |

* Uni_MoE_8e_e2 is trained using [Uni MoE Speech 8e dataset](https://huggingface.co/datasets/VictorJsy/Uni-MoE-Training-Dataset/blob/main/Uni_MoE_Speech_v2_with_token.json) which add llava-665K for better image-text instruction tuning compared with MoE-Task2.

## 🗝️ Dataset

### Training Data
| DataSet  | Type |
|----------|-----------|
| [LLaVA-Instruct-665K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) | [imgae(coco-train2017)](http://images.cocodataset.org/zips/train2017.zip)(etc) |
| [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | [imgae(train2014)](http://images.cocodataset.org/zips/train2014.zip) |
| [Video-Instruct-Dataset](https://github.com/mbzuai-oryx/Video-ChatGPT) | [video(from youtube)](https://www.youtube.com/) |
| [RACE](https://huggingface.co/datasets/race/tree/main) | [Speech(TTS)](https://www.cs.cmu.edu/~glai1/data/race/) |
| [LibriSpeech](https://www.openslr.org/12) | [Speech(Long)](https://www.openslr.org/12) |

We use TTS technical to convert long text to speech to construct long speech understanding data.

### Evaluation Data
| DataSet  | Input Type |
|----------|----------|
| [AOKVQA](https://allenai.org/project/a-okvqa/home) | Text-Image |
| [OKVQA](https://okvqa.allenai.org/) | Text-Image |
| [VQAv2](https://visualqa.org/) | Text-Image |
| [MMBench](https://mmbench.opencompass.org.cn/home) | Text-Image |
| [POPE](https://github.com/RUCAIBox/POPE) | Text-Image |
| [TextVQA](https://textvqa.org/dataset/) | Text-Image |
| [MM-Vet](https://github.com/yuweihao/MM-Vet) | Text-Image |
| [SEEDBench(Image)](https://github.com/ailab-cvc/seed-bench?tab=readme-ov-file) | Text-Image |
| [MMBench-Audio](https://mmbench.opencompass.org.cn/home) | Text-Image-Speech(Long) |
| [English-High-School-Listening](https://huggingface.co/datasets/VictorJsy/College-Entrance-English-Examination-Listening-Part/tree/main) | Text-Speech(Long) |
| [RACE](https://huggingface.co/datasets/race/tree/main) | Text-Speech(Long) |
| [MSVD](http://www.cs.utexas.edu/users/ml/clamp/videoDescription/) |Text-Video-Audio |
| [Activitynet-QA](https://github.com/MILVLG/activitynet-qa) |Text-Video-Audio |

#### College Entrance English Examination Listening Part

We build a real speech understanding dataset to check the practical long speech recognition capabilities: [English-High-School-Listening](https://huggingface.co/datasets/VictorJsy/College-Entrance-English-Examination-Listening-Part/tree/main)
It comprises 150 questions related to long audio segments with an average length of 109 seconds, and 50 questions about short audio segments with an average length of 14 seconds.


## 🌈 How to infer and deploy your demo

1. Make sure that all the weights are downloaded and the running environment is set correctly.
2. run inference scripts [`inference_speech.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/Uni_MoE_8e/inference_speech.sh) using ```bash inference_speech.sh``` or run the following commands to inference:
3. NOTE: 8-experts model share the same Uni-MoE-speech-base, **remember to replace the content of ```config.json``` with ```8config.json``` before inference**.

```bash
cd /path/to/Uni_MoE/Uni_MoE_8e
conda activate unimoe_8e
export MASTER_PORT=10079
export GPUS_PER_NODE=2

deepspeed --num_gpus=2 --num_nodes=1 \
    --master_addr "localhost" --master_port $MASTER_PORT \
    Uni_MoE_speech/inference_new.py \
    --deepspeed ./scripts/zero2.json \
    --model_base path/to/Uni-MoE-speech-base \
    --model_path output/Uni_MoE_8e_e2 \
    --data_path /path/to/eval.json \
    --enable_deepspeed_moe True \
    --data_type vqa\
    --eval_ep_size 2 \
    --mlp_dir path/to/Uni_MoE_8e_Experts\
    --version v1 \
    --vision_tower path/to/clip-vit-large-patch14-336 \
    --audio_tower path/to/whisper-small \
    --output_dir Uni_MoE_speech_output
```


## 🌈 How to train and evaluate on datasets

Training:
1. Make sure that all the weights are downloaded and the environment is set correctly, especially for the base model.
2. Make sure that all the data are downloaded and pre-processed utilizing [`data_add_tokens_release.py`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/Uni_MoE_8e/data_preprocess/data_add_tokens_release.py).
3. Run training scripts: [`train_deepspeed_8moe_release1.slurm`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/Uni_MoE_8e/train_deepspeed_8moe_release1.slurm) using ```bash train_deepspeed_8moe_release1.slurm``` ```sbatch train_deepspeed_8moe_release1.slurm```, remember to modify the training set with your own preference.

Evaluation:
1. Prepare the evaluation set using the form as [`samples.json`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/data_sample/samples.json).
2. Run evaluation scripts: [`eval_speech.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/Uni_MoE_8e/eval_speech.sh) using ```bash eval_speech.sh``` or run the following commands to eval:
3. NOTE: 8-experts model share the same Uni-MoE-speech-base, **remember to replace the content of ```config.sjon``` with ```8config.json``` before evaluation**.
```bash
cd path/to/Uni_MoE/Uni_MoE_8e
conda activate unimoe_8e

deepspeed --num_gpus=2 --num_nodes=1 \
    --master_addr "localhost" --master_port $MASTER_PORT \
    Uni_MoE_speech/eval.py \
    --deepspeed ./scripts/zero2.json \
    --model_base checkpoints/Uni-MoE-speech-base \
    --model_path output/Uni_MoE_8e_e2 \
    --data_path path/to/eval.json \
    --enable_deepspeed_moe True \
    --data_type vqa\
    --eval_ep_size 2 \
    --mlp_dir path/to/Uni_MoE_8e_Experts\
    --version v1 \
    --vision_tower checkpoints/clip-vit-large-patch14-336 \
    --audio_tower checkpoints/whisper-small \
    --output_dir Uni_MoE_speech_eval_out.json
```
We recommend using 2x80GB GPU RAM to run all experiments.

## Citation

If you find Uni-MoE useful for your research and applications, please cite using this BibTeX:
```bibtex

@ARTICLE{li_unimoe,
  author={Li, Yunxin and Jiang, Shenyuan and Hu, Baotian and Wang, Longyue and Zhong, Wanqi and Luo, Wenhan and Ma, Lin and Zhang, Min},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Uni-MoE: Scaling Unified Multimodal LLMs With Mixture of Experts}, 
  year={2025},
  volume={47},
  number={5},
  pages={3424-3439},
  doi={10.1109/TPAMI.2025.3532688}}


```

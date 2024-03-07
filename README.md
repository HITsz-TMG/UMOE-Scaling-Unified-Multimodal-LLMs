
<p align="center">
    <img src="https://s21.ax1x.com/2024/03/07/pFrOouV.png" width="250" style="margin-bottom: 0.2;"/>
<p>
<h2 align="center"> <a href="https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/">Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts</a></h2>
<h5 align="center"> If you appreciate our project, please consider giving us a star ⭐ on GitHub to stay updated with the latest developments.  </h2>

<h5 align="center">

🚀 Welcome to the repo of **Uni-MOE**!
Uni-MoE is a multi-modal framework which is capable of processing diverse modalities, including audio, image, text, and video. Specifically, our approach integrates multimodal experts within the transformer blocks of LLMs, comprising: 1) a shared self-attention mechanism for all modalities, 2) modality-specific experts derived from feed-forward networks, and 3) a sparse routing mechanism for allocating token-level expert attention.

[![Project Page](https://img.shields.io/badge/Project_Page-todo-blue)](todo)
[![Demo](https://img.shields.io/badge/Demo-todo-orange)](todo) 
[![Paper](https://img.shields.io/badge/Paper-todo-yellow)](todo)

[Yunxin Li](https://yunxinli.github.io), [Shenyuan Jiang](URL)
</h5>

## 🔥 News
- [3/7] 🔥 We released **Uni-MOE: Scaling Unified Multimodal LLMs with Mixture of Experts**. We proposed the development of a unified Multimodal LLM (MLLM) utilizing the MoE framework, which is capable of processing diverse modalities, including audio, image, text, and video.  Checkout the [paper](TODO) and [demo](TODO).

**Usage and License Notices**: The data and checkpoint are intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA and Vicuna. The dataset and models trained using the dataset should not be used outside of research purposes.

## 🌟 Structure

The model architecture of Uni-MoE is shown below. Three training stages contain: 1) Utilize pairs from different modalities and languages to build connectors that map these elements to a unified language space, establishing a foundation for multimodal understanding; 2) Develop modality-specific experts using cross-modal data to ensure deep understanding, preparing for a cohesive multi-expert model; 3) Incorporate multiple trained experts into LLMs and refine the unified multimodal model using the LoRA technique on mixed multimodal data.

<div align=center><img src="https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/model.png" height="100%" width="75%"/></div>

## ⚡️ Install

The following instructions are for linux installation.
We recommend the requirements as follows.
* Python == 3.9.16
* CUDA Version >= 11.7

1. Clone this repository and navigate to UMOE folder
```bash
git clone https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs.git
cd UMOE-Scaling-Unified-Multimodal-LLMs/UMOE
```

2. Install Package
```Shell
conda create -n umoe python==3.9.16
conda activate umoe
pip install -r env.txt
```

3. Change all the Absolute Pathnames: "/path/to/" to your own path to UMOE file

## ⚡️ Uni-MOE Weights

To use our model, all weights should be downloaded.

After downloading all of them, organize the weights as follows in 'UMOE/checkpoint' folder:
```
└── checkpoint
    ├── UMOE-audio-base
    ├── UMOE-audio-e2
    ├── UMOE-speech-base
    ├── UMOE-speech-e2
    ├── clip-vit-large-patch14-336
    ├── whisper-small
    └── BEATs_iter3_plus_AS2M.pt
```
| Model  | Checkpoint |
|----------|-----------|
| vision encoder | [CLIP ViT-L/14 336px](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main) |
| speech encoder | [whisper small](https://huggingface.co/openai/whisper-small/tree/main) |
| audio encoder  | [Fine-tuned BEATs_iter3+ (AS2M)](https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D) |
| UMOE-audio-base-model | [UMOE/UMOE-audio-base](https://huggingface.co/UMOE/UMOE-audio-base) |
| UMOE-audio-fine-tuned-chekpoint | [UMOE/UMOE-audio-e2](https://huggingface.co/UMOE/UMOE-audio-e2) |
| UMOE-speech-base-model | [UMOE/UMOE-speech-base](https://huggingface.co/UMOE/UMOE-speech-base) |
| UMOE-speech-fine-tuned-chekpoint | [UMOE/UMOE-speech-e2](https://huggingface.co/UMOE/UMOE-speech-e2) |
* UMOE-speech refers to the MOE-Task2 and UMOE-audio refers to the MOE-Task3 in our paper.

## 🗝️ Dataset

### Training Data


### Evaluation Data



## 🌈 How to inference

1. Make sure that all the weights and downloaded and environment are set correctly.
2. run inference scripts [`inference_audio.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/UMOE/inference_audio.sh) and [`inference_speech.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/UMOE/inference_speech.sh) ```bash inference_audio.sh``````bash inference_speech.sh```or run the following commands to inference:
```
cd /path/to/UMOE
conda activate umoe
python umoe_audio/inference_all.py
```
```
cd /path/to/UMOE
conda activate umoe
python umoe_speech/inference_all.py
```


## 🌈 How to train and evaluate

Training:
1. Make use that all the weights and downloaded and environment is set correctly, especially the base model.
2. Our training data can be downloaded from: [UMOE-Speech-453k.json](url) and [UMOE-Cap-453k.json](url).
3. Relevant vision and audio files: [todo](url)
4. Run training scripts: [`finetune_audio.sh`](url) or [`finetune_speech.sh`](url), remember to modify the training set with your own preference.

Evaluation:
1. Make use that all the weights and downloaded and environment is set correctly.
2. Prepare the evaluation set using the form as [`samples.json`](url).
3. Run training scripts: [`finetune_audio.sh`](url) or [`finetune_speech.sh`](url), remember to set the right data_type and output filename.

## Citation

If you find LLaVA useful for your research and applications, please cite using this BibTeX:
```bibtex

@misc{li2024umoe,
      title={Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts}, 
      author={},
      publisher={},
      year={2024},
}

```
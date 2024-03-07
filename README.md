
<p align="center">
    <img src="https://s21.ax1x.com/2024/03/07/pFrOouV.png" width="250" style="margin-bottom: 0.2;"/>
<p>
<h2 align="center"> <a href="https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/">Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts</a></h2>
<h5 align="center"> If you appreciate our project, please consider giving us a star ⭐ on GitHub to stay updated with the latest developments.  </h2>

<h5 align="center">

🚀 Welcome to the repo of **UMOE**!
UMoE is a multi-modal framework which is capable of processing diverse modalities, including audio, image, text, and video. Specifically, our approach integrates multimodal experts within the transformer blocks of LLMs, comprising: 1) a shared self-attention mechanism for all modalities, 2) modality-specific experts derived from feed-forward networks, and 3) a sparse routing mechanism for allocating token-level expert attention.

[![Project Page](https://img.shields.io/badge/Project_Page-todo-blue)](todo)
[![Demo](https://img.shields.io/badge/Demo-todo-orange)](todo) 
[![Paper](https://img.shields.io/badge/Paper-todo-yellow)](todo)


**UMOE: Scaling Unified Multimodal LLMs with Mixture of Experts** [[Paper](url)] <br>
[Yunxin Li](url) [Shenyuan Jiang](url)


## 🔥 News
- [4/17] 🔥 We released **UMOE: Scaling Unified Multimodal LLMs with Mixture of Experts**. We proposed the development of a unified Multimodal LLM (MLLM) utilizing the MoE framework, which is capable of processing diverse modalities, including audio, image, text, and video.  Checkout the [paper](TODO) and [demo](TODO).

**Usage and License Notices**: The data and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA and Vicuna. The dataset and models trained using the dataset should not be used outside of research purposes.

## 🌟 Structure

The model architecture of UMOE is shown below. Three training stages contain: 1) Utilize pairs from different modalities and languages to build connectors that map these elements to a unified language space, establishing a foundation for multimodal understanding; 2) Develop modality-specific experts using cross-modal data to ensure deep understanding, preparing for a cohesive multi-expert model; 3) Incorporate multiple trained experts into LLMs and refine the unified multimodal model using the LoRA technique on mixed multimodal data.

<div align=center><img src="resource/model.png" height="100%" width="75%"/></div>

## ⚡️ Install

The following instructions are for linux installation.

1. Clone this repository and navigate to UMOE folder
```bash
git clone https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs.git
cd UMOE
```

2. Install Package
```Shell
conda create -n umoe python==3.9.16
conda activate umoe
pip install -r env.txt
```

3. Change all the paths: "/path/to/" to your own path to UMOE file

## ⚡️ UMOE Weights

All weights should be place in the 'UMOE/checkpoint' folder

UMOE-audio can be downloaded from: [base model](url) and [fine-tuned chekpoint](url).
UMOE-speech can be downloaded from: [base model](url) and [fine-tuned chekpoint](url).

## ⚡️ Encoder Weights

1. Download vision encoder [CLIP ViT-L/14 336px](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main).
3. Download speech encoder [whisper small](https://huggingface.co/openai/whisper-small/tree/main).
4. Download audio encoder [Fine-tuned BEATs_iter3+ (AS2M)](https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D).

## 🌈 How to inference

1. Make use that all the weights and downloaded and environment is set correctly.
2. run inference scripts: [`inference_audio.sh`](url) or [`inference_speech.sh`](url).

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

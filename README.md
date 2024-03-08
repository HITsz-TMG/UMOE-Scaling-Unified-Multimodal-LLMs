
<p align="center">
    <img src="https://s21.ax1x.com/2024/03/07/pFrOouV.png" width="250" style="margin-bottom: 0.2;"/>
<p>
<h2 align="center"> <a href="https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/">Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts</a></h2>
<h5 align="center"> If you appreciate our project, please consider giving us a star ⭐ on GitHub to stay updated with the latest developments.  </h2>

<h5 align="center">

🚀 Welcome to the repo of **Uni-MOE**!
Uni-MoE is a MoE-based unified multimodal model, which is capable of handling diverse modalities, including audio, speech, image, text, and video.

[![Project Page](https://img.shields.io/badge/Project_Page-todo-blue)](todo)
[![Demo](https://img.shields.io/badge/Demo-todo-orange)](todo) 
[![Paper](https://img.shields.io/badge/Paper-todo-yellow)](todo)

[Yunxin Li](https://yunxinli.github.io), [Shenyuan Jiang](URL), [Baotian Hu](https://faculty.hitsz.edu.cn/hubaotian), [Longyue Wang](http://www.longyuewang.com/), [Lin Ma](https://forestlinma.com/), [Min Zhang](https://faculty.hitsz.edu.cn/MinZhang)
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
cd UMOE-Scaling-Unified-Multimodal-LLMs/Uni_MoE
```

2. Install Package
```Shell
conda create -n unimoe python==3.9.16
conda activate unimoe
pip install -r env.txt
```

3. Change all the Absolute Pathnames: "/path/to/" to your own path to UMOE file

## ⚡️ Uni-MOE Weights

To use our model, all weights should be downloaded.

After downloading all of them, organize the weights as follows in 'Uni_MoE/checkpoint' folder:
```
└── checkpoint
    ├── Uni-MoE-audio-base
    ├── Uni-MoE-audio-e2
    ├── Uni-MoE-speech-base
    ├── Uni-MoE-speech-e2
    ├── clip-vit-large-patch14-336
    ├── whisper-small
    └── BEATs_iter3_plus_AS2M.pt
```
| Model  | Checkpoint |
|----------|-----------|
| vision encoder | [CLIP ViT-L/14 336px](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main) |
| speech encoder | [whisper small](https://huggingface.co/openai/whisper-small/tree/main) |
| audio encoder  | [Fine-tuned BEATs_iter3+ (AS2M)](https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D) |
| Uni-MoE-audio-base-model | [Uni-MoE/Uni-MoE-audio-base](https://huggingface.co/VictorJsy/Uni-MoE-audio-base/tree/main) |
| Uni-MoE-audio-fine-tuned-chekpoint | [Uni-MoE/Uni-MoE-audio-e2](https://huggingface.co/VictorJsy/Uni-MoE-audio-e2/tree/main) |
| Uni-MoE-speech-base-model | [Uni-MoE/Uni-MoE-speech-base](https://huggingface.co/VictorJsy/Uni-MoE-speech-base/tree/main) |
| Uni-MoE-speech-fine-tuned-chekpoint | [Uni-MoE/Uni-MoE-speech-e2](https://huggingface.co/VictorJsy/Uni-MoE-speech-e2/tree/main) |
* Uni-MoE-speech refers to the MOE-Task2 and Uni-MoE-audio refers to the MOE-Task3 in our paper.

## 🗝️ Dataset

### Training Data
| DataSet  | Type |
|----------|-----------|
| [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | [imgae(train2014)](http://images.cocodataset.org/zips/train2014.zip) |
| [Video-Instruct-Dataset](https://github.com/mbzuai-oryx/Video-ChatGPT) | [video(from youtube)](url) |
| [WavCaps](https://huggingface.co/datasets/cvssp/WavCaps/tree/main/json_files) | [audio](https://huggingface.co/datasets/cvssp/WavCaps/tree/main/Zip_files) |
| [AudioCaps](https://audiocaps.github.io/) | [audio(Cap)](https://audiocaps.github.io/) |
| [ClothoAQA](https://zenodo.org/records/6473207)  | [audio(QA)](https://zenodo.org/records/6473207) |
| [ClothoV1](https://zenodo.org/records/3490684) | [audio(Cap)](https://zenodo.org/records/3490684) |
| [MELD](https://affective-meld.github.io/) | [audio(Music)](https://affective-meld.github.io/) |
| [RACE](https://huggingface.co/datasets/race/tree/main) | [audio(TTS)](https://www.cs.cmu.edu/~glai1/data/race/) |
| [LibriSpeech](https://www.openslr.org/12) | [audio(Long)](https://www.openslr.org/12) |

### Evaluation Data
| DataSet  | Input Type |
|----------|----------|
| [AOKVQA](https://allenai.org/project/a-okvqa/home) | Text-Image |
| [OKVQA](https://okvqa.allenai.org/) | Text-Image |
| [VQAv2](https://visualqa.org/) | Text-Image |
| [ClothoAQA](https://zenodo.org/records/6473207) | Audio-Text |
| [ClothoV1](https://zenodo.org/records/3490684) | Audio-Text |
| [ClothoV2](https://zenodo.org/records/3490684) | Audio-Text |
| [MMBench](https://mmbench.opencompass.org.cn/home) | Image-Text |
| [MMBench-Audio](https://mmbench.opencompass.org.cn/home) | Text-Image-Speech(Long) |
| [English-High-School-Listening](https://huggingface.co/datasets/VictorJsy/College-Entrance-English-Examination-Listening-Part/tree/main) | Text-Speech(Long) |
| [RACE](https://huggingface.co/datasets/race/tree/main) | Text-Speech(Long) |
| [MSVD](http://www.cs.utexas.edu/users/ml/clamp/videoDescription/) |Text-Video-Audio |
| [Activitynet-QA](https://github.com/MILVLG/activitynet-qa) |Text-Video-Audio |

### College Entrance English Examination Listening Part

We build a real speech understanding dataset to check the practical long speech recognition capabilities: [English-High-School-Listening](https://huggingface.co/datasets/VictorJsy/College-Entrance-English-Examination-Listening-Part/tree/main)
It comprises 150 questions related to long audio segments which have an average length of 109 seconds, and 50 questions about short audio segments with an average length of 14 seconds.


## 🌈 How to inference

1. Make sure that all the weights are downloaded and the running environment is set correctly.
2. run inference scripts [`inference_audio.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/inference_audio.sh) and [`inference_speech.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/inference_speech.sh) using ```bash inference_audio.sh``` ```bash inference_speech.sh```or run the following commands to inference:
```bash
cd /path/to/Uni_MoE
conda activate unimoe
python Uni_MoE_audio/inference_all.py
```
```bash
cd /path/to/Uni_MoE
conda activate unimoe
python Uni_MoE_speech/inference_all.py
```


## 🌈 How to train and evaluate

Training:
1. Make use that all the weights and downloaded and environment is set correctly, especially the base model.
2. Our training data can be downloaded from: [UMOE-Speech-453k.json](url) and [UMOE-Cap-453k.json](url).
3. Relevant vision and audio files: [todo](url)
4. Run training scripts: [`finetune_audio.sh`](url) or [`finetune_speech.sh`](url) using ```bash finetune_audio.sh``` ```bash finetune_speech.sh```, remember to modify the training set with your own preference.

Evaluation:
1. Make use that all the weights and downloaded and environment is set correctly.
2. Prepare the evaluation set using the form as [`samples.json`](url).
3. Run evaluation scripts: [`eval_audio.sh`](url) or [`eval_speech.sh`](url) using ```bash eval_audio.sh``` ```bash eval_speech.sh``` or run the following commands to eval:
```bash
cd /path/to/Uni_MoE
conda activate unimoe
python Uni_MoE_audio/eval.py\
 --data_path /path/to/clotho.json\
 --data_type clothov1\
 --output test.json
```
```bash
cd /path/to/Uni_MoE
conda activate unimoe
python Uni_MoE_speech/eval.py\
 --data_path /path/to/vqa_eval.json\
 --data_type vqa\
 --output test.json
```
It is recommended to use 80GB GPU RAM for inference, training and evalutation.
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

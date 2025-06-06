
<p align="center">
    <img src="https://s21.ax1x.com/2024/05/14/pkmtDSS.png" width="250" style="margin-bottom: 0.2;"/>
<p>
<h2 align="center"> <a href="https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/">Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts</a></h2>
<h5 align="center"> If you appreciate our project, please consider giving us a star ⭐ on GitHub to stay updated with the latest developments.  </h2>

<h4 align="center">

🚀 Welcome to the repo of **Uni-MOE**!

Uni-MoE is a MoE-based unified multimodal model and can handle diverse modalities including audio, speech, image, text, and video.

[![🤗Hugging Face](https://img.shields.io/badge/🤗Hugging_Face-Uni_MoE-yellow)](https://huggingface.co/Uni-MoE)
[![Project Page](https://img.shields.io/badge/Project_Page-Uni_MoE-blue)](https://uni-moe.github.io/)
[![Demo](https://img.shields.io/badge/Demo-Local-orange)](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/tree/master?tab=readme-ov-file#-demo-video) 
[![Paper](https://img.shields.io/badge/Paper-arxiv-yellow)](https://arxiv.org/abs/2405.11273)

[![](https://trendshift.io/api/badge/repositories/10407)](https://trendshift.io/repositories/10407)

[Yunxin Li](https://yunxinli.github.io), [Shenyuan Jiang](URL), [Baotian Hu](https://faculty.hitsz.edu.cn/hubaotian), [Longyue Wang](http://www.longyuewang.com/), [Wanqi Zhong](URL), [Wenhan Luo](https://whluo.github.io/), [Lin Ma](https://forestlinma.com/), [Min Zhang](https://faculty.hitsz.edu.cn/MinZhang)
</h4>



## 🔥 News

- [1/9]  🔥 Our paper has been accepted by **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)**, 2025.

- [8/28] 🔥 We release our video evaluation benchmark [VideoVista](https://videovista.github.io/) and the automatically generated video instruction tuning data [VideoVista-Train](https://huggingface.co/datasets/Uni-MoE/VideoVista_Train).

- [5/31] 🔥 The checkpoint of Uni-MoE-v2 with 8 experts is now available for downloading and inference. For more details, please refer to the [Uni_MoE_v2_weights](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE_v2/README.md#%EF%B8%8F-uni-moe-weights) table. 
- [4/28] 🔥 We have upgraded the Uni-MoE codebase to facilitate training across multiple Nodes and GPUs. Explore this enhanced functionality in our revamped [fine-tuning script](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/finetune_speech_dp.sh). Additionally, we have introduced a version that integrates distributed MoE modules. This enhancement allows for training our model with parallel processing at both the expert and modality levels, enhancing efficiency and scalability. For more details, please refer to the [Uni_MoE_v2](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/tree/master/Uni_MoE_v2) documentation. 
- [3/7] 🔥 We released **Uni-MOE: Scaling Unified Multimodal LLMs with Mixture of Experts**. We proposed the development of a unified Multimodal LLM (MLLM) utilizing the MoE framework, which can process diverse modalities, including audio, image, text, and video.  Checkout the [paper](https://arxiv.org/abs/2405.11273) and [demo](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/tree/master?tab=readme-ov-file#-demo-video).

**Usage and License Notices**: The data and checkpoint are intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA and Vicuna. The dataset and models trained using the dataset should not be used outside of research purposes.

## 🎨 Case Show

<div align=center><img src="https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/case_figure.png" height="100%" width="75%"/></div>

## 📀 Demo Video

Demo 2 contains the real-time understanding of speech (Starting from 30S).

https://private-user-images.githubusercontent.com/45393746/331798338-dfc848a2-1fd2-4f8d-9274-f21f7118ecd9.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTYwMzUwOTUsIm5iZiI6MTcxNjAzNDc5NSwicGF0aCI6Ii80NTM5Mzc0Ni8zMzE3OTgzMzgtZGZjODQ4YTItMWZkMi00ZjhkLTkyNzQtZjIxZjcxMThlY2Q5Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA1MTglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNTE4VDEyMTk1NVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTYzYTNmZDNlM2FhOGE3MmM1MzM0Mzk4YTdlYTg3NTgzOTBmNzMyMjM4OTljYTA0ODQ0YmEzZDVlYmFhOWUwMzUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.vrqxhusaq3J_ULQKbeGOxEJH3wry6GjXLxwrFrP0jao

https://private-user-images.githubusercontent.com/45393746/331798343-fcd3eb7e-3dfa-4470-a2e6-b9b140efe0fa.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTYwMzUyNTEsIm5iZiI6MTcxNjAzNDk1MSwicGF0aCI6Ii80NTM5Mzc0Ni8zMzE3OTgzNDMtZmNkM2ViN2UtM2RmYS00NDcwLWEyZTYtYjliMTQwZWZlMGZhLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA1MTglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNTE4VDEyMjIzMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTIyNWU5OTM0NjM1MTgzMWIxNWI4MDllYzU5NWNlOTUxMGI1NzQ5MzkyNmRlNDFlMTY0YzYzMTJmZjk4ZjJmMWEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.Uz3PBfKbEjl5ZOfUSXrAaQQLrvKwCFK2uNPTjtKG3dU


## 🌟 Structure

The model architecture of Uni-MoE is shown below. Three training stages contain: 1) Utilize pairs from different modalities and languages to build connectors that map these elements to a unified language space, establishing a foundation for multimodal understanding; 2) Develop modality-specific experts using cross-modal data to ensure deep understanding, preparing for a cohesive multi-expert model; 3) Incorporate multiple trained experts into LLMs and refine the unified multimodal model using the LoRA technique on mixed multimodal data.

<div align=center><img src="https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/model.png" height="100%" width="75%"/></div>

## ⚡️ Install

The following instructions are for Linux installation.
We would like to recommend the requirements as follows.
* Python == 3.9.16
* CUDA Version >= 11.7

1. Clone this repository and navigate to the Uni-MoE folder
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

3. Replace all the absolute pathnames '/path/to/' with your specific path to the Uni-MoE file
**(Including all the eval_x.py/inference_x.py/train_mem_x.py/data.py/demo.py files and config.json files from the model weights)**

## ⚡️ Uni-MOE Weights

To use our model, all weights should be downloaded.

After downloading all of them, organize the weights as follows in 'Uni_MoE/checkpoint' folder:
```
└── checkpoint
    ├── Uni-MoE-audio-base
    ├── Uni-MoE-audio-e2
    ├── Uni-MoE-speech-base
    ├── Uni-MoE-speech-e2
    ├── Uni-MoE-speech-base-interval
    ├── Uni-MoE-speech-v1.5
    ├── clip-vit-large-patch14-336
    ├── whisper-small
    └── BEATs_iter3_plus_AS2M.pt
```
| Model  | Checkpoint |
|----------|-----------|
| vision encoder | [CLIP ViT-L/14 336px](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main) |
| speech encoder | [whisper small](https://huggingface.co/openai/whisper-small/tree/main) |
| audio encoder  | [BEATs_iter3+ (AS2M)](https://1drv.ms/u/s!AqeByhGUtINrgcpke6_lRSZEKD5j2Q?e=A3FpOf) |
| Uni-MoE-audio-base-model | [Uni-MoE/Uni-MoE-audio-base](https://huggingface.co/VictorJsy/Uni-MoE-audio-base/tree/main) |
| Uni-MoE-audio-fine-tuned-chekpoint | [Uni-MoE/Uni-MoE-audio-e2](https://huggingface.co/VictorJsy/Uni-MoE-audio-e2/tree/main) |
| Uni-MoE-speech-base-model | [Uni-MoE/Uni-MoE-speech-base](https://huggingface.co/VictorJsy/Uni-MoE-speech-base/tree/main) |
| Uni-MoE-speech-fine-tuned-chekpoint | [Uni-MoE/Uni-MoE-speech-e2](https://huggingface.co/VictorJsy/Uni-MoE-speech-e2/tree/main) |
| Uni-MoE-speech-base-interval | [Uni-MoE/Uni-MoE-speech-base-interval](https://huggingface.co/VictorJsy/Uni-MoE-speech-base-interval) |
| Uni-MoE-speech-v1.5  | [Uni-MoE/Uni-MoE-speech-v1.5](https://huggingface.co/VictorJsy/Uni-MoE-speech-v1.5) |

* Uni-MoE-speech refers to the MOE-Task2 and Uni-MoE-audio refers to the MOE-Task3 in our paper.
* 'Uni-MoE-base' is the backbone containing LLMs and trained parameters gained from Training Stage 2: Training Modality-Specific Expert.

## 🗝️ Dataset

### Training Data
| DataSet  | Type |
|----------|-----------|
| [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | [image(train2014)](http://images.cocodataset.org/zips/train2014.zip) |
| [Video-Instruct-Dataset](https://github.com/mbzuai-oryx/Video-ChatGPT) | [video(from youtube)](https://www.youtube.com/) |
| [WavCaps](https://huggingface.co/datasets/cvssp/WavCaps/tree/main/json_files) | [audio](https://huggingface.co/datasets/cvssp/WavCaps/tree/main/Zip_files) |
| [AudioCaps](https://audiocaps.github.io/) | [audio(Cap)](https://audiocaps.github.io/) |
| [ClothoAQA](https://zenodo.org/records/6473207)  | [audio(QA)](https://zenodo.org/records/6473207) |
| [ClothoV1](https://zenodo.org/records/3490684) | [audio(Cap)](https://zenodo.org/records/3490684) |
| [MELD](https://affective-meld.github.io/) | [audio(Music)](https://affective-meld.github.io/) |
| [RACE](https://huggingface.co/datasets/race/tree/main) | [Speech(TTS)](https://www.cs.cmu.edu/~glai1/data/race/) |
| [LibriSpeech](https://www.openslr.org/12) | [Speech(Long)](https://www.openslr.org/12) |

We use TTS technical to convert long text to speech to construct long speech understanding data.

### Evaluation Data
| DataSet  | Input Type |
|----------|----------|
| [AOKVQA](https://allenai.org/project/a-okvqa/home) | Text-Image |
| [OKVQA](https://okvqa.allenai.org/) | Text-Image |
| [VQAv2](https://visualqa.org/) | Text-Image |
| [ClothoAQA](https://zenodo.org/records/6473207) | Text-Audio |
| [ClothoV1](https://zenodo.org/records/3490684) | Text-Audio |
| [ClothoV2](https://zenodo.org/records/3490684) | Text-Audio |
| [POPE](https://github.com/RUCAIBox/POPE) | Text-Image |
| [TextVQA](https://textvqa.org/dataset/) | Text-Image |
| [MM-Vet](https://github.com/yuweihao/MM-Vet) | Text-Image |
| [SEEDBench(Image)](https://github.com/ailab-cvc/seed-bench?tab=readme-ov-file) | Text-Image |
| [MMBench](https://mmbench.opencompass.org.cn/home) | Text-Image |
| [MMBench-Audio](https://mmbench.opencompass.org.cn/home) | Text-Image-Speech(Long) |
| [English-High-School-Listening](https://huggingface.co/datasets/VictorJsy/College-Entrance-English-Examination-Listening-Part/tree/main) | Text-Speech(Long) |
| [RACE](https://huggingface.co/datasets/race/tree/main) | Text-Speech(Long) |
| [MSVD](http://www.cs.utexas.edu/users/ml/clamp/videoDescription/) |Text-Video-Audio |
| [Activitynet-QA](https://github.com/MILVLG/activitynet-qa) |Text-Video-Audio |

#### College Entrance English Examination Listening Part

We build a real speech understanding dataset to check the practical long speech recognition capabilities: [English-High-School-Listening](https://huggingface.co/datasets/VictorJsy/College-Entrance-English-Examination-Listening-Part/tree/main)
It comprises 150 questions related to long audio segments with an average length of 109 seconds, and 50 questions about short audio segments with an average length of 14 seconds.

### Experimental Results

<div align=center><img src="https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni-MoE-Experiments.png" height="100%" width="90%"/></div>

## 🌈 How to infer and deploy your demo

1. Make sure that all the weights are downloaded and the running environment is set correctly.
2. run inference scripts [`inference_audio.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/inference_audio.sh) and [`inference_speech.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/inference_speech.sh) using ```bash inference_audio.sh``` ```bash inference_speech.sh``` or run the following commands to inference:
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
To launch the online demo ( It is highly recommended to launch the demo with [Uni-MoE-speech-v1.5](https://huggingface.co/VictorJsy/Uni-MoE-speech-v1.5) that need the basic parameters of [Uni-MoE-speech-base-interval](https://huggingface.co/VictorJsy/Uni-MoE-speech-base-interval)), run:
```bash
cd /path/to/Uni_MoE
conda activate unimoe
python demo/demo.py
python demo/app.py
```


## 🌈 How to train and evaluate on datasets

Training:
1. Make sure that all the weights are downloaded and the environment is set correctly, especially for the base model.
2. Our training data can be downloaded from [UMOE-Speech-453k.json](https://huggingface.co/datasets/Uni-MoE/Uni-MoE-Training-Dataset/blob/main/Uni_MoE_Speech.json) and [UMOE-Cap-453k.json](https://huggingface.co/datasets/Uni-MoE/Uni-MoE-Training-Dataset/blob/main/Uni_MoE_Cap.json).
3. Relevant vision and audio files: [Dataset](#Training-Data)
4. Run training scripts: [`finetune_audio.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/finetune_audio.sh) or [`finetune_speech.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/finetune_speech.sh) using ```bash finetune_audio.sh``` ```bash finetune_speech.sh```, remember to modify the training set with your own preference.
5. For multiple GPUs training, run training scripts: [`finetune_speech_dp.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/finetune_speech_dp.sh) using ```bash finetune_speech_dp.sh```, remember to modify the training set with your own preference.

Evaluation:
1. Prepare the evaluation set using the form as [`samples.json`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/data_sample/samples.json).
2. Run evaluation scripts: [`eval_audio.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/eval_audio.sh) or [`eval_speech.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/eval_speech.sh) using ```bash eval_audio.sh``` ```bash eval_speech.sh``` or run the following commands to eval:
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
We recommend using 80GB GPU RAM to run all experiments.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs&type=Date)](https://star-history.com/#HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs&Date)


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

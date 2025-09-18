
<p align="center">
    <img src="https://s21.ax1x.com/2024/05/14/pkmtDSS.png" width="250" style="margin-bottom: 0.2;"/>
<p>
<h4 align="center">

🚀 Welcome to the repo of **Uni-MOE**

Uni-MoE is a MoE-based unified multimodal model and can understand and generate omnimodalities.

[![🤗Hugging Face](https://img.shields.io/badge/🤗Hugging_Face-UniMoE-yellow)](https://huggingface.co/Uni-MoE)
[![Project Page](https://img.shields.io/badge/Project_Page-UniMoE-blue)](https://uni-moe.github.io/)
[![Demo](https://img.shields.io/badge/Demo-UniMoE-orange)](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/tree/master?tab=readme-ov-file#-demo-video) 
[![Paper](https://img.shields.io/badge/Arxiv-UniMoE-yellow)](https://arxiv.org/abs/2405.11273)

[![](https://trendshift.io/api/badge/repositories/10407)](https://trendshift.io/repositories/10407)

</h4>

<h4 align="center"> If you appreciate our project, please consider giving us a star ⭐ on GitHub to stay updated with the latest developments.  </h4>



## 🔥 News

- [2025/8/6] We release a better Uni-MoE v1.5 at modelscope [here](https://www.modelscope.cn/models/victorjsyy/Uni-MoE) with a unified speech encoding approach.

- [2025/1/9]  🔥 Our paper has been accepted by **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)**, 2025.

- [2024/8/28] 🔥 We release our video evaluation benchmark [VideoVista](https://videovista.github.io/) and the automatically generated video instruction tuning data [VideoVista-Train](https://huggingface.co/datasets/Uni-MoE/VideoVista_Train).

- [2024/5/31] 🔥 The checkpoint of Uni-MoE-v2 with 8 experts is now available for downloading and inference. For more details, please refer to the [Uni_MoE_v2_weights](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/Uni_MoE_8e/README.md#%EF%B8%8F-uni-moe-weights) table. 
- [2024/4/28] 🔥 We have upgraded the Uni-MoE codebase to facilitate training across multiple Nodes and GPUs. Explore this enhanced functionality in our revamped [fine-tuning script](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/finetune_speech_dp.sh). Additionally, we have introduced a version that integrates distributed MoE modules. This enhancement allows for training our model with parallel processing at both the expert and modality levels, enhancing efficiency and scalability. For more details, please refer to the [Uni_MoE_v2](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/tree/master/Uni_MoE/Uni_MoE_8e) documentation. 
- [2024/3/7] 🔥 We released **Uni-MOE: Scaling Unified Multimodal LLMs with Mixture of Experts**. We proposed the development of a unified Multimodal LLM (MLLM) utilizing the MoE framework, which can process diverse modalities, including audio, image, text, and video.  Checkout the [paper](https://arxiv.org/abs/2405.11273) and [demo](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/tree/master?tab=readme-ov-file#-demo-video).



## 🎨 Case Show

The cases of Uni-MoE

<div align=center><img src="https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/case_figure.png" height="100%" width="75%"/></div>

## 📀 Demo Video

Demo 2 contains the real-time understanding of speech (Starting from 30S).

https://private-user-images.githubusercontent.com/45393746/331798338-dfc848a2-1fd2-4f8d-9274-f21f7118ecd9.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTYwMzUwOTUsIm5iZiI6MTcxNjAzNDc5NSwicGF0aCI6Ii80NTM5Mzc0Ni8zMzE3OTgzMzgtZGZjODQ4YTItMWZkMi00ZjhkLTkyNzQtZjIxZjcxMThlY2Q5Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA1MTglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNTE4VDEyMTk1NVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTYzYTNmZDNlM2FhOGE3MmM1MzM0Mzk4YTdlYTg3NTgzOTBmNzMyMjM4OTljYTA0ODQ0YmEzZDVlYmFhOWUwMzUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.vrqxhusaq3J_ULQKbeGOxEJH3wry6GjXLxwrFrP0jao

https://private-user-images.githubusercontent.com/45393746/331798343-fcd3eb7e-3dfa-4470-a2e6-b9b140efe0fa.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTYwMzUyNTEsIm5iZiI6MTcxNjAzNDk1MSwicGF0aCI6Ii80NTM5Mzc0Ni8zMzE3OTgzNDMtZmNkM2ViN2UtM2RmYS00NDcwLWEyZTYtYjliMTQwZWZlMGZhLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA1MTglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNTE4VDEyMjIzMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTIyNWU5OTM0NjM1MTgzMWIxNWI4MDllYzU5NWNlOTUxMGI1NzQ5MzkyNmRlNDFlMTY0YzYzMTJmZjk4ZjJmMWEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.Uz3PBfKbEjl5ZOfUSXrAaQQLrvKwCFK2uNPTjtKG3dU


## 🌟 Model Structure

The model architecture of Uni-MoE is shown below. Three training stages contain: 1) Utilize pairs from different modalities and languages to build connectors that map these elements to a unified language space, establishing a foundation for multimodal understanding; 2) Develop modality-specific experts using cross-modal data to ensure deep understanding, preparing for a cohesive multi-expert model; 3) Incorporate multiple trained experts into LLMs and refine the unified multimodal model using the LoRA technique on mixed multimodal data.

<div align=center><img src="https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/model.png" height="100%" width="75%"/></div>


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

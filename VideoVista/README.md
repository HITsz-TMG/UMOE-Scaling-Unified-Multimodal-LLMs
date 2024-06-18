# VideoVista: A Versatile Benchmark for Video Understanding and Reasoning

<font size=2><div align='center' >  [[📖 arXiv Paper](https://arxiv.org/abs/2406.11303)] [[📊 Dataset ](https://huggingface.co/datasets/Uni-MoE/VideoVista)] </div></font>

If you like our project, please consider giving us a star ⭐ on the Uni-MoE repository to stay updated with the latest developments.

---
## 🔥 News

**`2024.06.15`** 🚀 The evaluation code for VideoVista is released!

**`2024.06.11`** 🚀 We release VideoVista, a Versatile Benchmark for Video Understanding and Reasoning. You can download the evaluation part of VideoVista from [hugging face](https://huggingface.co/datasets/Uni-MoE/VideoVista).  We will release all 100k training data and enhanced Uni-MoE-v2 soon.

## 🌟 VideoVista Overview

Despite significant breakthroughs in video analysis driven by the rapid development of large multimodal models (LMMs), there remains a lack of a versatile evaluation benchmark to comprehensively assess these models' performance in video understanding and reasoning. 

We introduce **VideoVista**, a video benchmark that integrates challenges across diverse content categories, durations, and abilities.  Specifically, VideoVista comprises **25,000** questions derived from **3,400** videos spanning **14** categories (e.g., Howto, Film, and Entertainment) with durations ranging from a few seconds to over 10 minutes. Besides, it encompasses **19** types of understanding tasks (e.g., anomaly detection, interaction understanding) and **8** reasoning tasks (e.g., logical reasoning, causal reasoning).



[//]: # ()
[//]: # (<figure style="margin: 0; text-align: center;">)

[//]: # (    <img src="asset/data_stastic.png" alt="Image 1" style="width: 80%; display: block; margin: 0 auto;"/>)

[//]: # (    <figcaption>Figure 1: Sample Count for Task Types</figcaption>)

[//]: # (</figure>)

<div align=center><img src="asset/data_stastic.png" height="60%" width="60%"/></div>

We develop an automatic video annotation framework that efficiently creates large-scale training and evaluates VideoQA datasets. The automatic process is shown in the following figure.

[//]: # ()
[//]: # (<figure style="margin: 0; text-align: center;">)

[//]: # (    <img src="asset/model.png" alt="Image 1" style="width: 90%; display: block; margin: 0 auto;"/>)

[//]: # (    <figcaption>Figure 1: Sample Count for Task Types</figcaption>)

[//]: # (</figure>)

<div align=center><img src="asset/model.png" height="60%" width="60%"/></div>


## 🗝️ Dataset Statistics
<table>
  <tr>
    <td style="text-align: center;">
      <img src="asset/Time.png" alt="Image 1" style="width: 100%;"/>
      <figcaption>The statistics of 14 video categories</figcaption>
    </td>
    <td style="text-align: center;">
      <img src="asset/Category.png" alt="Image 2" style="width: 100%;"/>
      <figcaption>The distribution of video durations</figcaption>
    </td>
  </tr>
</table>


[//]: # (<div style="display: flex; justify-content: space-between;">)

[//]: # (    <figure style="width: 45%; text-align: center;">)

[//]: # (        <img src="asset/Time.png" alt="Image 1" style="width: 100%;"/>)

[//]: # (        <figcaption>Figure 2: The statistics of 14 video categories</figcaption>)

[//]: # (    </figure>)

[//]: # (    <figure style="width: 45%; text-align: center;">)

[//]: # (        <img src="asset/Category.png" alt="Image 2" style="width: 100%;"/>)

[//]: # (        <figcaption>Figure 3: The distribution of video durations &#40;minute&#41;</figcaption>)

[//]: # (    </figure>)

[//]: # (</div>)

## 🔍 Dataset Example

[//]: # (<figure style="margin: 0; text-align: center;">)

[//]: # (    <img src="asset/Case.png" alt="Image 1" style="width: 80%; display: block; margin: 0 auto;"/>)

[//]: # (    <figcaption>Figure 4: Example for Task Types</figcaption>)

[//]: # (</figure>)

<div align=center><img src="asset/Case.png" height="90%" width="90%"/></div>

[//]: # (## Dataset Examples)

[//]: # ()
[//]: # ([//]: # &#40;![image/png]&#40;assest/Example.jpg&#41;&#41;)
[//]: # ()
[//]: # (<p align="center">)

[//]: # (    <img src="asset/Example.png" width="100%" height="100%">)

[//]: # (</p>)

[//]: # (For each type of task, we also provide a text-based example.)

[//]: # (![image/png]&#40;asset/Case.jpg&#41;)

## 🌈 Evaluation

You can download the JSON file and VIDEOS from [Huggingface-VideoVista](https://huggingface.co/datasets/Uni-MoE/VideoVista).

When downloading videos, you may need to download all the zip files with different suffixes. After that, just extract the file using follow command.

```shell
cat merged.zip* > merged.zip
unzip merged.zip
```
The video_name attribute in the JSON file corresponds one-to-one with the video name.

When evaluating the Relation Reasoning-Image task, you also need to download [relation images](https://huggingface.co/datasets/Uni-MoE/VideoVista/blob/main/relation_images.zip) to obtain the queried images.

During evaluation, you can add a Model_Answer attribute to the original VideoVista.json file to store the model's prediction results, and then use the [evaluation code](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/VideoVista/evaluation/evaluation_videovista.py) in the evaluation directory.

For Chinese users experiencing slow download speeds with Hugging Face, you can use [HF-Mirror](https://hf-mirror.com/) to speed up downloads.


## Citation


---
license: apache-2.0
task_categories:
- question-answering
language:
- en
size_categories:
- 10K<n<100K
viewer: false
---

# VideoVista: A Versatile Benchmark for Video Understanding and Reasoning

<font size=2><div align='center' >  [[📖 arXiv Paper(Coming Soon)](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/tree/master/VideoVista)] [[📊 Dataset (Coming Soon)](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/tree/master/VideoVista)] </div></font>

---
## 🔥 News
**`2024.06.11`** 🚀 We are very proud to launch VideoVista, A Versatile Benchmark for Video Understanding and Reasoning.

##  🌟 VideoVista Overview

Despite significant breakthroughs in video analysis driven by the rapid development of large multimodal models (LMMs), there remains a lack of a versatile evaluation benchmark to comprehensively assess these models' performance in video understanding and reasoning. 

We introduce **VideoVista**, a video benchmark that integrates challenges across diverse content categories, durations, and abilities.  Specifically, VideoVista comprises **25,000** questions derived from **3,400** videos spanning **14** categories (e.g., Howto, Film, and Entertainment) with durations ranging from a few seconds to over 10 minutes. Besides, it encompasses **19** types of understanding tasks (e.g., anomaly detection, interaction understanding) and **8** reasoning tasks (e.g., logical reasoning, causal reasoning).


<figure style="margin: 0; text-align: center;">
    <img src="asset/data_stastic.png" alt="Image 1" style="width: 80%; display: block; margin: 0 auto;"/>
    <figcaption>Figure 1: Sample Count for Task Types</figcaption>
</figure>

[//]: # (<div style="display: flex; justify-content: space-between;">)

[//]: # (    <img src="asset/Time.png" alt="Image 1" style="width: 48%;"/>)

[//]: # (    <img src="asset/Category.png" alt="Image 2" style="width: 48%;"/>)

[//]: # (</div>)

<div style="display: flex; justify-content: space-between; align-items: center;">
    <figure style="margin: 0; text-align: center;">
        <img src="asset/Time.png" alt="Image 1" style="width: 90%;"/>
        <figcaption>Figure 2: The statistics of 14 video categories</figcaption>
    </figure>
    <figure style="margin: 0; text-align: center;">
        <img src="asset/Category.png" alt="Image 2" style="width: 90%;"/>
        <figcaption>Figure 3: The distribution of video durations (minute)</figcaption>
    </figure>
</div>

## 🔍 Dataset Example

<figure style="margin: 0; text-align: center;">
    <img src="asset/Case.png" alt="Image 1" style="width: 80%; display: block; margin: 0 auto;"/>
    <figcaption>Figure 4: Example for Task Types</figcaption>
</figure>

[//]: # (## Dataset Examples)

[//]: # ()
[//]: # ([//]: # &#40;![image/png]&#40;assest/Example.jpg&#41;&#41;)
[//]: # ()
[//]: # (<p align="center">)

[//]: # (    <img src="asset/Example.png" width="100%" height="100%">)

[//]: # (</p>)

[//]: # (For each type of task, we also provide a text-based example.)

[//]: # (![image/png]&#40;asset/Case.jpg&#41;)


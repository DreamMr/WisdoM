# WisdoM<img src="wisdom.png" width="20px"/>: Improving Multimodal Sentiment Analysis by Fusing Contextual World Knowledge

# Contents
- [Overview](#overview)
- [Install](#install)
- [Demo](#demo)

# Overview
Sentiment analysis is rapidly advancing by utilizing various data modalities (e.g., text, image). However, most previous works relied on superficial information, neglecting the incorporation of contextual world knowledge (e.g., background information derived from but beyond the given image and text pairs) and thereby restricting their ability to achieve better multimodal sentiment analysis (MSA).
In this paper, we proposed a plug-in framework named WisdoM, to leverage the contextual world knowledge induced from the large vision-language models (LVLMs) for enhanced MSA. WisdoM utilizes LVLMs to comprehensively analyze both images and corresponding texts, simultaneously generating pertinent context. To reduce the noise in the context, we also introduce a training-free contextual fusion mechanism. Experiments across diverse granularities of MSA tasks consistently demonstrate that our approach has substantial improvements (brings an average +1.96% F1 score among five advanced methods) over several state-of-the-art methods.

# Install
- Context Generation 

    We use LLaVA-v1.5 generate the context. Please follow the instruction of [LLaVA](https://github.com/haotian-liu/LLaVA) to prepare the environment.


# Demo
To run a demo of the project, execute the following command:
```
python run.py
```
We also provide a Jupyter Notebook [demo](./run.ipynb). You can view and interact with it.

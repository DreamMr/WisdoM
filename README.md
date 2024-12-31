# WisdoM<img src="wisdom.png" width="20px"/>: Improving Multimodal Sentiment Analysis by Fusing Contextual World Knowledge

[**🤗 Model**](https://huggingface.co/DreamMr/MMICL_MSED) | [**📖 Paper**](https://dl.acm.org/doi/abs/10.1145/3664647.3681403)

# News
**[2025.01.01]**   🎉 Happy New Year! We have released the **pretrained weight [MMICL-MSED](https://huggingface.co/DreamMr/MMICL_MSED)** and the **evaluation code**.

**[2024.07.21]**   🥳 Our work was accepted by MM 2024.

# Contents
- [Overview](#overview)
- [Installation](#installation)
- [Demo](#demo)
- [Evaluation on MSED dataset](#evaluation)

# Overview
Sentiment analysis is rapidly advancing by utilizing various data modalities (e.g., text, image). However, most previous works relied on superficial information, neglecting the incorporation of contextual world knowledge (e.g., background information derived from but beyond the given image and text pairs) and thereby restricting their ability to achieve better multimodal sentiment analysis (MSA).
In this paper, we proposed a plug-in framework named WisdoM, to leverage the contextual world knowledge induced from the large vision-language models (LVLMs) for enhanced MSA. WisdoM utilizes LVLMs to comprehensively analyze both images and corresponding texts, simultaneously generating pertinent context. To reduce the noise in the context, we also introduce a training-free contextual fusion mechanism. Experiments across diverse granularities of MSA tasks consistently demonstrate that our approach has substantial improvements (brings an average +1.96% F1 score among five advanced methods) over several state-of-the-art methods.

# Installation
```
pip install -r requirements.txt
```
- Context Generation 

    We use LLaVA-v1.5 generate the context. Please follow the instruction of [LLaVA](https://github.com/haotian-liu/LLaVA) to prepare the environment.


# Demo
To run a demo of the project, execute the following command:
```
python run.py
```
We also provide a Jupyter Notebook [demo](./run.ipynb). You can view and interact with it.


# Evaluation on MSED dataset
1. Prepare data

    Download [MSED](https://github.com/MSEDdataset/MSED) images and put it under `./eval/data/test/images`

    **NOTE:** We provide the MSED test set data and the context used in the experiments in the `./data`.

2. Start evaluation!

    ```
    cd scripts
    bash run_eval_mmicl_msed.sh
    ```
    **config_path:** Configuration file path for evaluation. If you want to evaluate MMICL w/ WisdoM, please set `use_wisdom: true` in `./eval/configs/msed_sc.yaml`.

    **out:** the path of evaluation result.



## 📧 Contact
- Wenbin Wang: wangwenbin97@whu.edu.cn 

## ✒️ Citation
```
@inproceedings{wang2024wisdom,
  title={Wisdom: Improving multimodal sentiment analysis by fusing contextual world knowledge},
  author={Wang, Wenbin and Ding, Liang and Shen, Li and Luo, Yong and Hu, Han and Tao, Dacheng},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={2282--2291},
  year={2024}
}
```

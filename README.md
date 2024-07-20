# Main Experiment

```
python   3.8.0
pytorch  1.13.1
```



# Abstract

Currently, news content with images and texts on social media is widely spread, prompting significant interest in multi-modal fake news detection. However, existing research in this field focuses on largescale annotated data to train models. Furthermore, data scarcity characterizes the initial stages of fake news propagation. Hence, addressing the challenge of few-shot multi-modal fake news detection becomes essential. In scenarios of limited data availability, current research inadequately utilizes the information inherent in each modality, leading to underutilization of modal information. To address the above challenges, in the paper, we propose a novel detection approach called Promptbased Adaptive Fusion(ProAF). Specifically, to enhance the model’s comprehension of news content, we extract supplementary information from two modalities to facilitate timely guidance for model training. Then the model employs adaptive fusion to integrate the output predictions of different prompts during training, effectively enhancing the robust performance of the model. Experimental results on two datasets illustrate that our model surpasses existing methods, representing a significant advancement in few-shot multi-modal fake news detection.



# Data Availability Statement

Datasets used in the experiments are publicly available and can be downloaded from the following links: PolitiFact, Gossip: https://github.com/KaiDMML/FakeNewsNet.



# Train and test

```
python a_main.py
```



# Doubts

If you have any questions about the code file, please contact the author at ouyangq0011@163.com

ps: The first public code experiment, there may be a lot of inconsiderate places, but also please understand the researchers.



# Citation

If you find this repo helpful, please cite the following paper:

```
@inproceedings{Ouyang-etal-2024, 
title = "Enhancing Few-Shot Multi-Modal Fake News Detection through Adaptive Fusion", 
author = "Qiang Ouyang, Nankai Lin, Yongmei Zhou, Aimin Yang and Dong Zhou", 
booktitle = "The 8th APWeb-WAIM joint international conference on Web and Big Data: APWeb-WAIM 2024", 
year = "2024", 
```


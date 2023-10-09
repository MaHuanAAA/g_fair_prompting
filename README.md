# Fairness-guided Few-shot Prompting for Large Language Models

This repository is the code for paper "Fairness-guided Few-shot Prompting for Large Language Models".

## Setup

In a conda env with pytorch / cuda available, run:
```
pip install -r requirements.txt
```


## Download

For convenience, we pack the dataset and it is available on https://drive.google.com/file/d/1vyomvGBrXEnzE21P-nJr20OeZhF4zy5h/view?usp=sharing. Please change the `ROOT_DIR` in `utils.py` after downloading the datasets. The pretrained LLM BLOOM is available on https://huggingface.co/bigscience/bloom.

## Inference

The provided `start.sh` can be run on multi-gpu automatically and you need `8*A100 GPUs` for inference:
```
sh ./start.sh bloom
```




## Reference


```
@inproceedings{
ma2023fairness,
title={Fairness-guided Few-shot Prompting for Large Language Models},
author={Huan Ma and Changqing Zhang and Yatao Bian and Lemao Liu and Zhirui Zhang and Peilin Zhao and Shu Zhang and Huazhu Fu and Qinghua Hu and Bingzhe Wu},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)},
year={2023},
}
```

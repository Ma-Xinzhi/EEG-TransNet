# EEG-TransNet
## Attention-based convolutional neural network with multi-modal temporal information fusion for motor imagery EEG decoding [[paper]](https://doi.org/10.1016/j.compbiomed.2024.108504)
This is the PyTorch implementation of attention-based convolutional neural network with multi-modal temporal information fusion for MI-EEG decoding.
## Network Architecture
![Network architecture](https://github.com/Ma-Xinzhi/EEG-TransNet/blob/main/network_architecture.png)
The proposed network is designed with the aim of extracting multi-modal temporal information and learning more comprehensive global dependencies. It is composed of the following four parts:
1. Feature extraction module: The multi-modal temporal information is extracted from two distinct perspectives: average and variance.
2. Self-attention module: The shared self-attention module is designed to capture global dependencies along these two feature dimensions.
3. Convolutional encoder: The convolutional encoder is then designed to explore the relationship between average-pooled and variance-pooled features and fuse them into more discriminative features.
4. Classification: A fully connected (FC) layer finally classifies features from the convolutional encoder into given classes.
## Requirements
* PyTorch 1.7
* Python 3.7
* mne 0.23
## Datasets
* [BCI_competition_IV2a](https://www.bbci.de/competition/iv/)
* [BCI_competition_IV2b](https://www.bbci.de/competition/iv/)
* [High Gamma Dataset](https://gin.g-node.org/robintibor/high-gamma-dataset)
## Results
The classification results for our proposed network and other competing architectures are as follows:
![Results1](https://github.com/Ma-Xinzhi/EEG-TransNet/blob/main/result_1.png)
![Results2](https://github.com/Ma-Xinzhi/EEG-TransNet/blob/main/result_2.png)
## Citation
If you find this code useful, please cite us in your paper.
> @article{ma2024attention,\
  title={Attention-based convolutional neural network with multi-modal temporal information fusion for motor imagery EEG decoding},\
  author={Ma, Xinzhi and Chen, Weihai and Pei, Zhongcai and Zhang, Yue and Chen, Jianer},\
  journal={Computers in Biology and Medicine},\
  pages={108504},\
  year={2024},\
  publisher={Elsevier}\
}

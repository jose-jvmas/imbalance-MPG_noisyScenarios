<p align='center'>
  <a href=''><img src='https://i.imgur.com/Iu7CvC1.png' alt='PRAIG-logo' width='100'></a>
</p>

<h1 align='center'>Insights into imbalance-aware Multilabel Prototype Generation mechanisms for <i>k</i>-Nearest Neighbor classification in noisy scenarios</h1>

<h4 align='center'>Full text available <a href='' target='_blank'>soon</a>.</h4>

<p align='center'>
  <img src='https://img.shields.io/badge/python-3.8.18-orange' alt='Python'>
</p>



<p align='center'>
  <a href='#about'>About</a> •
  <a href='#contents'>Contents</a> •
  <a href='#how-to-use'> How To Use</a> •
  <a href='#citation'>Citation</a> •
  <a href='#acknowledgments'>Acknowledgments</a>
</p>


## About

This repository contains the source code and scripts to reproduce the experiments described in:

> **"Insights into imbalance-aware Multilabel Prototype Generation mechanisms for <em>k</em>-Nearest Neighbor classification in noisy scenarios"**  
> *Jose J. Valero-Mas, Carlos Penarrubia, Francisco J. Castellanos, Antonio Javier Gallego, Jorge Calvo-Zaragoza*  
> Pattern Recognition (2025).

<br/>
This work expands the proof-of-concept one by Penarrubia et al. [^1] and proposes a series of mechanisms to palliate two limitations found <b>multilabel Prototype Generation</b> for the <b><i>k</i>-Nearest Neighbor</b> classifier:

1. Addressing scenarios with <b>label-level imbalance</b>, which constitutes an inherent issue in multilabel learning.

2. Mitigatin the existing <b>label-level noise</b> in the data</b> assortments.
<br/>



## Contents

The repository is structured as follows:

- *Experiments.py* : Main script for performing the experimentation included in the manuscript.
- *MPG/* : Implementations of the base multilabel Prototype Generation methods together with the proposed extensions for imbalanced and noisy scenarios. These strategies are:
	- Multilabel Reduction through Homogeneous Clustering (MRHC) [^2]
	- Multilabel Chen (MChen) [^3]
	- Multilabel Reduction through Space Partitioning, version 3 (MRSP3) [^3]
- *Metrics.py*: Class including the evaluation metrics.
- *requirements.txt*: Python libraries to install

 [^1]: Penarrubia, C. ,  Valero-Mas, J. J., Gallego, A. J., & Calvo-Zaragoza, J. (2023). Addressing Class Imbalance in Multilabel Prototype Generation for k-Nearest Neighbor Classification. In: Proceedings of the 11th Iberian Conference on Pattern Recognition and Image Analysis, Alicante, Spain, June 27-30, pp. 15-27.

 [^2]: Ougiaroglou, S., Filippakis, P., & Evangelidis, G. (2021). Prototype generation for multi-label nearest neighbours classification. In: Proceedings of the 16th International Conference on Hybrid Artificial Intelligent Systems, Bilbao, Spain, September 22–24, pp. 172-183.
 
 [^3]: Valero-Mas, J. J., Gallego, A. J., Alonso-Jiménez, P., & Serra, X. (2023). Multilabel Prototype Generation for data reduction in K-Nearest Neighbour classification. Pattern Recognition, 135, 109190.

## How To Use

Follow these steps to set up the environment and reproduce the experiments.

### 1. Create a virtual environment

It is recommended to use a virtual environment to avoid dependency conflicts:

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
```

### 2. Install dependencies

Install required packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Run the experiments

To execute all the experiments, use the following script:

```bash
python Experiments.py
```


## Citation

```bibtex
@article{valeromas:PR:2023,
  title={Insights into imbalance-aware Multilabel Prototype Generation mechanisms for k-Nearest Neighbor classification in noisy scenarios},
  author={Valero-Mas, Jose J. and and Penarrubia, Carlos and Castellanos, Francisco J. and Gallego, Antonio Javier and Calvo-Zaragoza, Jorge},
  journal={Pattern Recognition},
  year={2025},
  publisher={Elsevier}
}
```


## Acknowledgments

This work was partially funded by the Generalitat Valenciana through projects: 
- SmallOMR (CIAICO/2023/255).
- MUltimodal and Self-supervised Approaches for MUsic Transcription (CIGE/2023/216).

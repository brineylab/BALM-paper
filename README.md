[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8253367.svg)](https://doi.org/10.5281/zenodo.8253367)
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://jupyter.org/try)
![GitHub](https://img.shields.io/github/license/briney/antiref)

# Improving antibody language models with native pairing

To determine whether and the extent to which training with natively paired antibody sequence data can improve antibody-specific language models (**LMs**), we trained two baseline antibody language model (**BALM**) variants: ***BALM-paired***, which is trained using only natively paired training data, and ***BALM-unpaied***, which is trained using the same antibody sequences but without pairing information. Additionally, we performed full fine-tuning of the state-of-the-art general protein LM ESM-2 using the same natively paired dataset used to train BALM-paired. The Jupyter notebooks in this repository contain all code necessary to re-train each of these models from scratch:

* [**BALM-paired**](BALM-paired.ipynb): downloads training data (if necessary) and trains BALM-paired.
* [**BALM-unpaired**](BALM-unpaired.ipynb): downloads training data (if necessary) and trains BALM-unpaired.
* [**ESM-2 fine-tuning**](ESM2_fine-tuning.ipynb): downloads training data (if necessary) and performs full fine-tuning of ESM-2.

### pre-trained models
Weights for each of the aforementioned models can be downloaded from [Zenodo](https://zenodo.org/record/8253367).

### how should I cite BALM?
BALM has been published as a preprint on arXiv, and can be cited as:


Additionally, Zenodo provides a unique, citable DOI for each version of a deposited dataset. The DOI for the current version of BALM (v2023.08.17) is 10.5281/zenodo.8253367, so an appropriate citation would be:

```
Burbach, Sarah, & Briney, Bryan. (2023). Improving antibody language models with native pairing (v2023.08.17) 
[Data set]. Zenodo. https://doi.org/10.5281/zenodo.8253367
```


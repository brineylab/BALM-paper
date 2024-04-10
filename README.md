[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8253367.svg)](https://doi.org/10.5281/zenodo.8253367)
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://jupyter.org/try)
![GitHub](https://img.shields.io/github/license/briney/antiref)

# Improving antibody language models with native pairing

To determine whether and the extent to which training with natively paired antibody sequence data can improve antibody-specific language models (**LMs**), we trained three baseline antibody language model (**BALM**) variants: ***BALM-paired***, which is trained using only natively paired training data, ***BALM-shuffled***, which is trained using randomly paired trianing data, and ***BALM-unpaied***, which is trained using the same antibody sequences but without pairing information. Additionally, we performed full fine-tuning of the state-of-the-art general protein LM ESM-2 using the same natively paired dataset used to train BALM-paired. The Jupyter notebooks in this repository contain all code necessary to re-train each of these models from scratch:

* [**BALM-paired**](BALM-paired.ipynb): downloads training data (if necessary) and trains BALM-paired.
* [**BALM-shuffled**](BALM-paired.ipynb): training data will need to be processed to randomly shuffle the pairing, then use the same training script as BALM-paired
* [**BALM-unpaired**](BALM-unpaired.ipynb): downloads training data (if necessary) and trains BALM-unpaired.
* [**ESM-2 fine-tuning**](ESM2_fine-tuning.ipynb): downloads training data (if necessary) and performs full fine-tuning of ESM-2.

### pre-trained models
Weights for each of the aforementioned models can be downloaded from [Zenodo](https://zenodo.org/records/10684811).

### how should I cite BALM?
BALM has been published in [Patterns](https://www.cell.com/patterns/fulltext/S2666-3899(24)00075-8), and can be cited as:

```
Burbach, S.M., & Briney, B. (2024). Improving antibody language models with native pairing.
Patterns. https://doi.org/10.1016/j.patter.2024.100967

```

The current version of the BALM dataset (v2024.02.20) can be cited as:

```
Burbach SM, Briney B. Improving antibody language models with native pairing (v2024.02.20) [Data set].
Zenodo. 2023. https://doi.org/10.5281/zenodo.10684811
```


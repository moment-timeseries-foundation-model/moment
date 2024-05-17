<div align="center">
<h1>MOMENT: A Family of Open Time-series Foundation Models</h1>

[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2402.03885&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2402.03885)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E)](https://huggingface.co/AutonLab/MOMENT-1-large)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E)](https://huggingface.co/datasets/AutonLab/Timeseries-PILE)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/license/MIT)
</div>

## 🔥 News
- MOMENT was accepted at ICML 2024!
- We are working on releasing the MOMENT research code, so you can pre-train your own time series foundation model, with your own data, and reproduce experiments from [our paper](https://arxiv.org/abs/2402.03885)! Stay tuned for updates!

## 📖 Introduction
We introduce MOMENT, a family of open-source foundation models for general-purpose time-series analysis. Pre-training large models on time-series data is challenging due to (1) the absence a large and cohesive public time-series repository, and (2) diverse time-series characteristics which make multi-dataset training onerous. Additionally, (3) experimental benchmarks to evaluate these models especially in scenarios with limited resources, time, and supervision, are still in its nascent stages. To address these challenges, we compile a large and diverse collection of public time-series, called the Time-series Pile, and systematically tackle time-series-specific challenges to unlock large-scale multi-dataset pre-training. Finally, we build on recent work to design a benchmark to evaluate time-series foundation models on diverse tasks and datasets in limited supervision settings. Experiments on this benchmark demonstrate the effectiveness of our pre-trained models with minimal data and task-specific fine-tuning. Finally, we present several interesting empirical observations about large pre-trained time-series models.

### MOMENT: One Model, Many Time Series Tasks, Less SupervisiondMOMENT can solve multiple time series analysis tasks well in settings with limited supervision, for example zero-shot anomaly detection and imputation, and unsupervised represenation learning for classification!

<div align="center">
    <img src="assets/moment_comparison .png" width="40%">
</div>

### MOMENT Architecture in a Nutshell

A time series is broken into disjoint fixed-length sub-sequences called patches, and each patch is mapped into a D-dimensional patch embedding. During pre-training, we mask patches uniformly at random by replacing their patch embeddings using a special mask embedding `[MASK]`. The goal of pre-training is to learn patch embeddings which can be used to reconstruct the input time series using a light-weight reconstruction head.

<div align="center">
    <img src="assets/moment_architecture.png" width="60%">
</div>



## 🧑‍💻 Usage

Install the package using:
```bash
pip install git+https://github.com/moment-timeseries-foundation-model/moment.git
```

To load the pre-trained model for one of the tasks, use one of the following code snippets:

**Forecasting**
```python
from moment import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={
        'task_name': 'forecasting',
        'forecast_horizon': 96
    },
)
model.init()
```

**Classification**
```python
from moment import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={
        'task_name': 'classification',
        'n_channels': 1,
        'num_class': 2
    },
)
model.init()
```

**Anomaly Detection, Imputation, and Pre-training**
```python
from moment import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={"task_name": "reconstruction"},
)
mode.init()
```

**Representation Learning**
```python
from moment import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={'task_name': 'embedding'},
)
```


## 🧑‍🏫 Tutorials

<!-- We provide tutorials to demonstrate how to use and fine-tune our pre-trained model on various tasks. -->
Here is the list of tutorials to get started with MOMENT for various tasks:
- [Forecasting](./tutorials/forecasting.ipynb)
- [Classification](./tutorials/classification.ipynb)
- [Anomaly Detection](./tutorials/anomaly_detection.ipynb)
- [Imputation](./tutorials/imputation.ipynb)
- [Representation Learning](./tutorials/representation_learning.ipynb)

> [!TIP]
> Have more questions about using MOMENT? Checkout [Frequently Asked Questions](https://docs.google.com/document/d/18P3-ghnFXO57Wyvg6IuMNOzDyHpR0RxQlkvBWig2DiI/edit?usp=sharing), and you might find your answer!

## BibTeX

```bibtex
@inproceedings{goswami2024moment,
  title={MOMENT: A Family of Open Time-series Foundation Models},
  author={Mononito Goswami and Konrad Szafer and Arjun Choudhry and Yifu Cai and Shuo Li and Artur Dubrawski},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```


## ⛑️ Research Code
We designed this codebase to be extremely lightweight, and in the process removed a lot of code! We are working on releasing (complete but messier) research code, which will include code to handly different datasets, and scripts for pre-training, fine-tuning and evaluating MOMENT alongside other baselines. An early version of this code is available on [Anonymous Github](https://anonymous.4open.science/r/BETT-773F/README.md).

## ➕ Contributions
We encourage researchers to contribute their methods and datasets to MOMENT. We are actively working on contributing guidelines. Stay tuned for updates!

## 📰 Coverage
- [Moment: A Family of Open Time-Series Foundation Models](https://ai.plainenglish.io/moment-a-family-of-open-time-series-foundation-models-80f5135ca35b), Medium post by Samuel Chazy
- [MOMENT: A Foundation Model for Time Series Forecasting, Classification, Anomaly Detection](https://towardsdatascience.com/moment-a-foundation-model-for-time-series-forecasting-classification-anomaly-detection-1e35f5b6ca76), Towards Datascience by Nikos Kafritsas
- [CMU Researchers Propose MOMENT: A Family of Open-Source Machine Learning Foundation Models for General-Purpose Time Series Analysis](https://www.marktechpost.com/2024/05/15/cmu-researchers-propose-moment-a-family-of-open-source-machine-learning-foundation-models-for-general-purpose-time-series-analysis/), MarketTechPost article by Mohammad Asjad
- [ARTIFICIAL INTELLIGENCEThe Rise of Time-Series Foundation Models for Data Analysis and Forecasting](https://www.unite.ai/the-rise-of-time-series-foundation-models-for-data-analysis-and-forecasting/), Unite AI blog by 
Dr. Tehseen Zia
- [Time Series AI: MOMENT Model](https://www.youtube.com/watch?v=D87XbbdB11M), Webinar hosted by [Gradient AI](https://gradient.ai/)


## 🤟 Contemporary Work
There's a lot of cool work on building time series forecasting foundation models! Here's an incomplete list. Checkout Table 9 in our [paper](https://arxiv.org/abs/2402.03885) for qualitative comparisons with these studies: 
- TimeGPT-1 by [Nixtla](https://www.nixtla.io/), [[Paper](https://arxiv.org/abs/2310.03589), [API](https://github.com/Nixtla/nixtla)]
- Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting by Morgan Stanley and ServiceNow Research, [[Paper](https://arxiv.org/abs/2310.08278), [Code](https://github.com/time-series-foundation-models/lag-llama), [Hugging Face](https://huggingface.co/time-series-foundation-models/Lag-Llama)]
- Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series by IBM, [[Paper](https://arxiv.org/abs/2401.03955), [Hugging Face](https://huggingface.co/ibm-granite/granite-timeseries-ttm-v1)]
- Moirai: A Time Series Foundation Model for Universal Forecasting [[Paper](https://arxiv.org/abs/2402.02592), [Code](https://github.com/SalesforceAIResearch/uni2ts), [Hugging Face](https://huggingface.co/Salesforce/moirai-1.0-R-large)]
- A decoder-only foundation model for time-series forecasting by Google, [[Paper](https://arxiv.org/abs/2310.10688), [Code](https://github.com/google-research/timesfm), [Hugging Face](https://huggingface.co/google/timesfm-1.0-200m)]
- Chronos: Learning the Language of Time Series by Amazon, [[Paper](https://arxiv.org/abs/2403.07815), [Code](https://github.com/amazon-science/chronos-forecasting), [Hugging Face](https://huggingface.co/collections/amazon/chronos-models-65f1791d630a8d57cb718444)]

There's also some recent work on solving multiple time series modeling tasks in addition to forecasting: 
- TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis [[Paper](https://arxiv.org/abs/2402.16412), [Code](https://github.com/SaberaTalukder/TOTEM)]

## 🪪 License

MIT License

Copyright (c) 2024 Auton Lab, Carnegie Mellon University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

See [MIT LICENSE](https://github.com/mononitogoswami/labelerrors/blob/main/LICENSE) for details.

<img align="right" height ="120px" src="assets/cmu_logo.png">
<img align="right" height ="110px" src="assets/autonlab_logo.png">

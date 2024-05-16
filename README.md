<div align="center">
# MOMENT: A Family of Open Time-series Foundation Models    

[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2402.03885&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2402.03885)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/AutonLab/MOMENT-1-large)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/license/MIT)
</div>



## Introduction
We introduce MOMENT, a family of open-source foundation models for general-purpose time-series analysis. Pre-training large models on time-series data is challenging due to (1) the absence a large and cohesive public time-series repository, and (2) diverse time-series characteristics which make multi-dataset training onerous. Additionally, (3) experimental benchmarks to evaluate these models especially in scenarios with limited resources, time, and supervision, are still in its nascent stages. To address these challenges, we compile a large and diverse collection of public time-series, called the Time-series Pile, and systematically tackle time-series-specific challenges to unlock large-scale multi-dataset pre-training. Finally, we build on recent work to design a benchmark to evaluate time-series foundation models on diverse tasks and datasets in limited supervision settings. Experiments on this benchmark demonstrate the effectiveness of our pre-trained models with minimal data and task-specific fine-tuning. Finally, we present several interesting empirical observations about large pre-trained time-series models.

[Paper](https://arxiv.org/abs/2402.03885) | 
[Model Card](https://huggingface.co/AutonLab/MOMENT-1-large) | 
[Time-series Pile](https://huggingface.co/datasets/AutonLab/Timeseries-PILE)

## Usage

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

## Tutorials

<!-- We provide tutorials to demonstrate how to use and fine-tune our pre-trained model on various tasks. -->
Here is the list of tutorials to get started with MOMENT for various tasks:
- [Forecasting](./tutorials/forecasting.ipynb)
- [Classification](./tutorials/classification.ipynb)
- [Anomaly Detection](./tutorials/anomaly_detection.ipynb)
- [Imputation](./tutorials/imputation.ipynb)
- [Representation Learning](./tutorials/representation_learning.ipynb)

## BibTeX

```bibtex
@inproceedings{goswami2024moment,
  title={MOMENT: A Family of Open Time-series Foundation Models},
  author={Mononito Goswami and Konrad Szafer and Arjun Choudhry and Yifu Cai and Shuo Li and Artur Dubrawski},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

## Research Code
We designed this codebase to be extremely lightweight, and in the process removed a lot of code! We are working on releasing (complete but messier) research code, which will include code to handly different datasets, and scripts for pre-training, fine-tuning and evaluating MOMENT alongside other baselines. An early version of this code is available on [Anonymous Github](https://anonymous.4open.science/r/BETT-773F/README.md).

## Contributions
We encourage researchers to contribute their methods and datasets to MOMENT. We are actively working on contributing guidelines. Stay tuned for updates!

## Coverage


## License

MIT License

Copyright (c) 2024 Auton Lab, Carnegie Mellon University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

See [MIT LICENSE](https://github.com/mononitogoswami/labelerrors/blob/main/LICENSE) for details.

<img align="right" height ="120px" src="assets/cmu_logo.png">
<img align="right" height ="110px" src="assets/autonlab_logo.png">

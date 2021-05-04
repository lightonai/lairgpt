# <img src="https://cloud.lighton.ai/wp-content/uploads/2020/01/LightOnCloud.png" width=50/>LairGPT

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  [![Twitter](https://img.shields.io/twitter/follow/LightOnIO?style=social)](https://twitter.com/LightOnIO)

A Python package in Pytorch by [LightOn AI Research](https://lair.lighton.ai/) that allows to perform inference
with [PAGnol models](https://lair.lighton.ai/pagnol/).
You can test the generation capabilities of PAGnol on our [interactive demo website](https://pagnol.lighton.ai/).

## Install

### Requirements

The package is tested with Python 3.9. After cloning this repository, you can create a `conda` environment
with the necessary dependencies from its root by

```
conda env create --file=environment.yml
```

If you prefer control on your environment, the dependencies are

```
omegaconf==2.0
pytorch==1.8.1
tokenizers==0.10
wget==3.2
```

### pip

Simply run `pip install .` from the root of this repository.

## Text generation

The simplest way to generate text with PAGnol using `lairgpt` is

```
from lairgpt.models import PAGnol

pagnol = PAGnol.small()
pagnol("Salut PAGnol, comment ça va ?")
```

We include a demo script `main.py` in this repository that takes the path to models and tokenizers, and an input text, and generates sentences from it.
To use it:

```
python main.py --size large --text "LightOn est une startup technologique"
```

To generate text we rely on the `infer` method of the `TextGenerator` class that takes the usual parameters:
- `mode`: (default: `"nucleus"`)
  - `"greedy"`: always select the most likely word as its next word.
  - `"top-k"`:  filter to the K most likely next words and redistribute the probability mass among only those K next words.
  - `"nucleus"`: filter to the smallest possible set of words whose cumulative probability exceeds the probability `p` and redistribute the probability mass among this set of words.
- `temperature`: a control over randomness. As this value approaches zero, the model becomes more deterministic. (default: `1.0`)
- `k`: size of the set of words to consider for "top-k" sampling (default: `5`)
- `p`: a control over diversity in nucleus sampling. A value of 0.5 means that half of the options are considered. (default: `0.9`)
- `max_decoding_steps`: number of tokens to generate. (default: `32`)
- `skip_eos`: when `True`, generation does not stop at end of sentence. (default: `True`)

## <img src="https://cloud.lighton.ai/wp-content/uploads/2020/01/LightOnCloud.png" width=50/> More on LightOn

LightOn is a company that produces hardware for machine learning.
To lease a LightOn Appliance, please visit: https://lighton.ai/lighton-appliance/

To request access to LightOn Cloud and try our photonic co-processor, please visit: https://cloud.lighton.ai/
For researchers, we also have a LightOn Cloud for Research program, please visit https://cloud.lighton.ai/lighton-research/ for more information.

## Citation

We will soon have a preprint on arXiv, stay tuned ;)

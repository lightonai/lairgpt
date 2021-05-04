# PAGnol: French Generative Models -- Model Card

This model card was inspired by [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993), *Mitchell et al.* 2018. 

## Model details 
* **Model description:** PAGNol is a series of large-scale generative models for the French language;
* **Model architecture:** [GPT](https://arxiv.org/abs/2005.14165)\-like, trained with a language modelling objective, with up to 1.5B parameters; 
* **Model history:** PAGnol models were trained over the month of April 2021, and trained on data cutting off in 2019;
* **Dataset:** trained on a dataset built with [CCNet](https://arxiv.org/abs/1911.00359), similar to the one used for [CamemBERT-Large](https://camembert-model.fr/), some research models trained with [OSCAR](https://oscar-corpus.com/);
* **Organization:** [LightOn](https://lighton.ai/), in cooperation with the [ALMAnaCH](http://almanach.inria.fr/index-en.html) team of Inria;
* **Licence:** PAGnol models and inference code are available under the [MIT Licence](https://github.com/lightonai/lairgpt/blob/master/LICENSE);
* **Paper:** coming soon, more information available [here](https://lair.lighton.ai/pagnol/);
* **Contact:** pagnol@lighton.ai.

## Intended uses

**PAGnol is geared towards free-form text generation.** It is best used for creative writing, without strong constraints on its outputs. 
It may be fine-tuned to specific forms and styles of text generation. However, in line with previous work ([T5](https://arxiv.org/abs/1910.10683), Raffel et al. 2019), we found it is not competitive when finetuned on specific tasks (classification, question answering, etc.)

PAGnol trained on OSCAR is downloadable only for research purposes. 

**We encourage further research on PAGnol zero-shot abilities as well as bias and fairness issues.** If you are interested in these topics, you can get in touch with us.


## Limitations and bias 

**PAGnol is not grounded, and cannot distinguish between facts and fiction.**

To enhance output quality, we trained PAGnol on [CCNet](https://arxiv.org/abs/1911.00359), a dataset filtered by a language model to match Wikipedia-like writing.
However, this does not prevent PAGnol from generating offensive, suggestive, or biased content. **Additional filtering of its outputs is recommended for most use cases.**

Use judgement and discretion before deploying PAGnol. **In particular, we recommend performing a study of use-case specific biases and limitations.**

PAGnol trained on OSCAR is significantly more likely to produce explicit and offensive content.


## Evaluation

We will share additional data on PAGnol end-task performance in a variety of contexts (discriminative/generative tasks, fine-tuning/few-shot) soon.
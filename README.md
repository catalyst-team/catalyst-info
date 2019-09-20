<div align="center">

![Catalyst logo](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png)

**Catalyst info**
 
[![Pipi version](https://img.shields.io/pypi/v/catalyst.svg)](https://pypi.org/project/catalyst/)
[![Docs](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fcatalyst%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://catalyst-team.github.io/catalyst/index.html)
[![PyPI Status](https://pepy.tech/badge/catalyst)](https://pepy.tech/project/catalyst)
[![License](https://img.shields.io/github/license/catalyst-team/catalyst.svg)](LICENSE)

[![Build Status](https://travis-ci.com/catalyst-team/catalyst.svg?branch=master)](https://travis-ci.com/catalyst-team/catalyst)
[![Telegram](./pics/telegram.svg)](https://t.me/catalyst_team)
[![Gitter](https://badges.gitter.im/catalyst-team/community.svg)](https://gitter.im/catalyst-team/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Slack](./pics/slack.svg)](https://opendatascience.slack.com/messages/CGK4KQBHD)
[![Donate](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/third_party_pics/patreon.png)](https://www.patreon.com/catalyst_team)

**Catalyst-info** is a series of posts about [Catalyst library](https://github.com/catalyst-team/catalyst) development and its [ecosystem](https://github.com/catalyst-team)

</div>

## Catalyst-info #3. Runners
catalyst-version: `19.09.4` date: `2019-09-20`

Hi, everybody! This is Catalyst-Team and the new issue of Catalyst-info #3.
Today we will talk about an important framework concept - [Runner](https://github.com/catalyst-team/catalyst/blob/master/catalyst/dl/core/runner.py).

---

There are two classes at the head of Catalyst.DL philosophy:

- `Experiment` is a class that contains information about the experiment - a model, a criterion, an optimizer, a scheduler and their hyperparameters. It also contains information about the data and the columns used. In general, the Experiment knows what to run. It is very important and we will talk about it next time.
- `Runner` is a class that knows how to run an experiment. It contains all the logic of how to run the experiment, stages (another distinctive feature of Catalyst), epoch and batches.

---

Runner's overall concept:

```python
for stage in experimnet.stages:
    for epoch in stage.epochs:
        for loader in epoch.loaders:
            for batch_in in loader:
                batch_out = runner.forward(batch_in)
                metrics = metrics_fn(batch_in, batch_out)
                optimize_fn(metrics)
```

Runner has only one abstract method - `forward`, which is responsible for the logic of processing incoming data by the model.

---

Runner uses the `RunnerState` [class](https://github.com/catalyst-team/catalyst/blob/master/catalyst/dl/core/state.py#L15) to communicate with Callbacks.

It records the current Runner parameters. For example, `batch_in` and `batch_out` , `metrics` and many others.

---

In addition, if you look at the classification and segmentation tasks, you can see a lot in common.  For example, only Experiment will be different for such tasks, not Runner. For this purpose, `SupervisedRunner` [appeared in Catalyst](https://github.com/catalyst-team/catalyst/blob/master/catalyst/dl/runner/supervised.py#L17).

Specialized for these tasks, it additionally implements methods `train `, `infer ` and `predict_loader `. The basic purpose - to give additional syntactic sugar for faster and more convenient R&D. Suitable both for work in Notebook API, and in Config API.

---

Additionally, for integration with Weights & Biases, there are realizations `WandbRunner ` and `SupervisedWandbRunner `. They do the same thing, but additionally log all the information on the wandb.app, which is very convenient if you have a lot of experiments.

---

And finally, we're working on [GANRunner](https://github.com/catalyst-team/catalyst/pull/365) now.

That will bring everyone's favorite GANs to Catalyst.
Let's [make GAN reproducible](catalyst-team/catalyst#365) once again!


## Catalyst-info #2. Tracing with Torch.Jit
catalyst-version: `19.08.6` date: `2019-08-27`

Hey, everybody! This is the Catalyst-info :tada: part two!

Today's post grew out of the question is any method to trace a Catalyst checkpoint with [torch.jit](https://pytorch.org/docs/stable/jit.html).

**What's it for?**

Traceability of Pytorch models allows you to speed up the model inference and allows you to run it not only with Python, but also with C++. It becomes like a binary file, without any code requirements – one step from research to production.

Additionally it can reduce the size of the Catalyst-checkpoint, removing all but the model.

 before tracing
![image 1](./pics/2/1.png)

 after tracing
![image 2](./pics/2/2.png)

---

**How do you get the checkpoint in Catalyst?**

To do this, there is a [command](https://github.com/catalyst-team/catalyst/blob/master/catalyst/dl/scripts/trace.py) `catalyst-dl trace <logdir>`

For example...
```bash
catalyst-dl trace /path/to/logs
```
---
For model's tracing, Catalyst uses the same code that was dumped during experiment, so that you can always recreate your model, even if the code in the production has already changed – reproducibility first :+1: 

---
You are free to choose which of the checkpoints you want to trace (default is `best`) by the argument `--checkpoint` or, shortly, `-c`
```bash
catalyst-dl trace /path/to/logs -c last
# or
catalyst-dl trace /path/to/logs --checkpoint stage1.1
```

In this case the output will look like this:
![image 3](./pics/2/3.png)

---

The `forward` method is executed by default, but this can be changed by selecting the necessary method in the `--method` argument, for example, our model has `inference` method:

```bash
catalyst-dl trace /path/to/logs --method inference
# or
catalyst-dl trace /path/to/logs -m inference
```

---
By default, traced models are saved in `logdir/trace`, but you can change it using one of the flags:
1. `--out-dir` changes the directory in which the model is saved, but the name of the model is generated by Catalyst, for example `--out-dir /path/to/output/`
2. `--out-model` indicates the path to a new file, for example `--out-model /path/to/output/traced-model-1.pth`

---
**How do I download the model after training?**

Once we've traced the model, it can be loaded into the python as
```python
model = torch.jit.load(path)
```
and in C++.
```cpp
module = torch::jit::load(path);
```

---
From interesting facts, in a format "and also ...": it is possible to trace a model not only in `eval` mode, but also in `train` + in addition to specify that we need to accumulate gradients. To change the mode to `train`:
```bash
catalyst-dl trace /path/to/logs --mode train
```

To indicate that we need gradients
```bash
catalyst-dl trace /path/to/logs --with-grad
```
These flags can be combined



## Catalyst-info #1. Segmentation models
catalyst-version: `19.08.6` date: `2019-08-22`

### Hello, everyone!

After the release of Catalyst in February it has a lot of new features, which, unfortunately, not everyone still knows about. Finally, we came up with an idea to post a random fact about catalyst every. So, the first release of catalyst-info!

---

In Catalyst we all have implemented our favorite Unet's: 
`Unet`, `Linknet`, `FPNUnet`, `PSPnet` and their brothers with resnet-encoders 
`ResnetUnet`, `ResnetLinknet`, `ResnetFPNUnet`, `ResnetPSPnet`. 
Any `Resnet` model can be fitted with any pre-trained encoder (resnet18, resnet34, resnet50, resnet101, resnet152)

Usage
```python
from catalyst.contrib.models.segmentation import ResnetUnet # or any other
model = ResnetUnet(arch="resnet34", pretrained=True)
```
It's easy to load up a `state_dict`
```python
model = ResnetUnet(arch="resnet34", pretrained=False, encoder_params=dict(state_dict="/model/path/resnet34-5c106cde.pth")
```
---

[Link to the model's code.](https://github.com/catalyst-team/catalyst/tree/master/catalyst/contrib/models/segmentation)

All models have a common general structure `encoder-bridge-decoder-head`, 
each of this part can be adjusted separately or even replaced by their own modules!
```python
# In the UnetMetaSpec class
def forward(self, x: torch.Tensor) -> torch.Tensor:
    encoder_features: List[torch.Tensor] = self.encoder(x)
    bridge_features: List[torch.Tensor] = self.bridge(encoder_features)
    decoder_features: List[torch.Tensor] = self.decoder(bridge_features)
    output: torch.Tensor = self.head(decoder_features)
    return output
```

To bolt your model as an encoder for segmentation, you need to inherit it from 
`catalyst.contrib.models.segmentation.encoder.core.EncoderSpec` ([Code](https://github.com/catalyst-team/catalyst/blob/master/catalyst/contrib/models/segmentation/encoder/core.py#L11)).

---

When creating your own block (for any `encoder/bridge/decoder/head`) using the [function](https://github.com/catalyst-team/catalyst/blob/master/catalyst/contrib/models/segmentation/blocks/core.py#L10) `_get_block` 
you can specify the `complexity` parameter, which will create a sequence of [complexity times by](https://github.com/catalyst-team/catalyst/blob/master/catalyst/contrib/models/segmentation/blocks/core.py#L34) `Conv2d + BN + activation`

---

The Upsample part can be specified [either by interpolation or by convolution](https://github.com/catalyst-team/catalyst/blob/febcb66ade07b231348fd8e19614bdd37d548125/catalyst/contrib/models/segmentation/head/unet.py#L18).

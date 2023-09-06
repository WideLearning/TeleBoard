# TeleBoard

## Installation
Run `pip install .` from the root of the repository (`Teleboard/`) before using it.

## Using visualizer
First, you can select what do you want to look at: weights, weight updates, or gradients.
Also you select what epoch and what layer you are interested in.

Then, for each configuration the currently supported plots are:
- Loss: Train & validation, everyone does it. Doesn't depend on configuration.
- Layer distributions: Approximate distribution of log-magnitudes of weights/updates/gradients, might help notice some failure modes related to exploding gradients or too strong weight norm regularization. Doesn't depend on selected layer, all layers are shown in one plot.
- Layer cosine similarities: Cosine similarities between weights/updates/gradients in this step and the previous one. High numbers might mean that you can safely increase the learning rate, low numbers mean that something is not working, and maybe you should decrease the learning rate. And if different layers have vastly different cosine similarities, it might mean a problem with gradient flow.
- Scaled / squashed layer paths: It is a 2d projection (axes are simply $(1, 1, 1, 1, \dots,)$ and $(1, -1, 1, -1, \dots)$) of weights/updates/gradients, with some transformation to make it fit into the limited plot space. Here you also can notice some patterns that will be useful for debugging the model, but it is harder to specify them explicitly.

![Screenshot](screenshot.png)

## Using tracker
There are three types of tracking currently supported:
- Neptune: Log collected statistics to Neptune.ai, so they are stored in the cloud.
- File: Save the statistics into a file in local storate.
- Console: It is a debug option, doesn't save statistics anywhere, just prints some of them to the standard output.

For a sample code showing how to connect the tracker to your model see [here](https://github.com/WideLearning/Kaggle/blob/titanic/titanic.py).
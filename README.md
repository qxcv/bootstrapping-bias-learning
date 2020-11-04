# EE227BT project: Bootstrapping bias models from priors

This repo was forked from the [`learning_biases`
repo](https://github.com/HumanCompatibleAI/learning_biases), which is the source
for Rohin Shah's paper ["On the Feasibility of Learning, Rather than Assuming,
Human Biases"](http://proceedings.mlr.press/v97/shah19a.html). What's in each
directory:

- `old/`: all the code from the `learning_biases` repo. We could probably just
  delete this, but it is nice to have it there to quickly cross-reference.
- `pref_bootstrap/`: most of the actual code for our project. This is a Python
  module that can be installed with the `setup.py` script.
- `notebooks/`: Jupyter/Colab notebooks. Hopefully most of our final experiment
  code can fit in just a couple of self-explanatory Jupyter notebooks.

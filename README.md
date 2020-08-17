# jax-practice


this is a repo implementing a simple wrapper around the recently released
autograd library, jax (https://github.com/google/jax), to easily create a
feedforward network for classification with a particular focus on convolutional
networks for image classification. i created this as an exercise to familiarize
myself with jax, and because of this purpose, there is absolutely no comment but
it's short enough for you to read and understand as long as you're familiar with
any existing framework, such as PyTorch and Tensorflow, for deep learning
(“Clean code should read like well-written prose” — Robert C. Martin)

- basic_cnn.ipynb, basic_ffnet.ipynb: jupyter notebooks demonstrating the use of
  this library.
- flax_test.ipynb: jupyter notebook that should run if jax is corrctly
  installed.
- utils.py: implements some helpful methods especially focusing on merging and
  manipulating pythong dictionaries.
- layers.py: implements various popular layers, including linear, 2d
  convolution, batch normalization and variou activation functions
- functionals.py: implements some stateless layers.
- model.py: implements the Model class that turns a list of layers into a full
  model while maintaining the state of the model, such as parameters and
  buffers.
- optimizers.py: implements SGD and Adam.

enjoy!

-- k

# NeuralFlows
Library for deep learning. 
<br>
Currently convolutional neural networks with auto differentiation are supported.
<br>
For learning there are two optimizers to use - minibatch gradient descent and momentum gradient descent<br>
For any kind of image edition there are util functions in Utils class that may be used accordingly including image resampling, interpolations and matrix transformations.
<br>

CUDA support is in progress as well as parallel convolutions on multiple devices for better performance.
<br>

Because CUDA is currently unspported - the experimental compile flag COMPILE_WITH_CUDA is set to OFF by default.
<br>


creation of a simple network with a `<28, 28, 3>` input tensor and a seed for pseudo-random generator of value `1` looks following 
```cpp
cn::Network network(28, 28, 3, 1);

```

From now on we can easily append next layers of to network. For example that's how to create 2 convolutional layers with 2 kernels of size `<3, 3, n>`, `ReLU` activation and normalization.

```cpp
network.appendConvolutionLayer({3, 3}, 2);
network.appendReLULayer();
network.appendBatchNormalizationLayer();
network.appendConvolutionLayer({3, 3}, 2);
network.appendReLULayer();
network.appendBatchNormalizationLayer();
```
Now after creation of the convolution layers we'd love to add some FFlayers, but firstly we should max pool with `<2,2>` kernel and flatten the tensor output from the previous layer. At the end we'll put a softmax to get an output as probablility.
```cpp
network.appendMaxPoolingLayer({2, 2});
network.appendFlatteningLayer();
network.appendFFLayer(10);
network.appendSigmoidLayer();
network.appendFFLayer(10);
network.appendSoftmaxLayer();
```

Then it is time to finally tell the network that it should get ready and to initialize random weights.
```cpp
network.ready();
network.initRandom();
```
Wonderful! We have created a simple `CNN`. The only problem that it does not know very much and it will behave no better than pseudo-random number generator. In order to fix that issue we should learn it. Fot that purpose we'll use minibatch gradient descent optimizer implemented by a `cn::MBGD` class.

```cpp
cn::MBGD mbgd(network, 0.1, 40);//network ref, learning rate, minibatch size
```

Once we have an input tensor of type either `cn::Tensor<byte>` or `cn::Tensor<double>` we are ready to feed our network.
```cpp
network.feed(tensor);
```
After each feed we can call an optimizer method `propagate()`. It will calculate all the gradients in a network and propagate them once the batch size is full.
```cpp
mbgd.propagate();
```

<br>
Depends on nlohmann/json library. https://github.com/nlohmann/json

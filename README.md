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

<br>
Depends on nlohmann/json library. https://github.com/nlohmann/json

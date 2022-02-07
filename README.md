# NeuralFlows
Library for deep learning. 
CUDA is not supported by all layers yet. The experimental compile flag COMPILE_WITH_CUDA is set to OFF by default.
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
After each feed we can call an optimizer's method `propagate()`. It will calculate all the gradients in a network and propagate them once the batch size is full.
```cpp
mbgd.propagate();
```

Unfortunately there would be absolutely no use from the network if we couldn't even save it. Therefore network derives from class `JSONEncodable`. By calling a `jsonEncode()` method, we'll get an `nlohmann::json` object, which can be easily saved to file.
```cpp
std::fstream file(filePath, std::ios::out);
file << network.jsonEncode();
file.close();
```

Since it can be saved, there should also be a possibility to deserialize such `json` file and create a network from it. Just use an overloaded constructor.
```cpp
using JSON = nlohmann::json;
std::ifstream ifs("test.json");
JSON json = json::parse(ifs);

Network network(json);
```
it will already have a ready state, therefore it is unnecessary to call neither `network.ready();` nor `network.initRandom();`.



<br>
Depends on `nlohmann/json` library. https://github.com/nlohmann/json

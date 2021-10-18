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

<br>
In few days there is going to be added recurrent layers with differentiation through _time.

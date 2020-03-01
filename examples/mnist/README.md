
The examples in this directory have been adapted from the [TensorFlow
tutorials](https://www.tensorflow.org/versions/r0.7/tutorials/mnist/pros/index.html).
To execute these examples, you will have to unzip the [MNIST data
files](http://yann.lecun.com/exdb/mnist/) in `data/`.

## Linear Classifier

The code can be found in `linear.ml`.

We first load the MNIST data. This is done using the MNIST helper module,
labels are returned using one-hot encoding.  Train images and labels are used
when training the model.  Test images and labels are used to estimate the
validation error.

```ocaml
  let { Dataset_helper.train_images; train_labels; test_images; test_labels } =
    Mnist_helper.read_files ~with_caching:true ()
  in
```

After that two tensors are initialized to hold the weights and biases for the
linear classifier. `requires_grad` is used when creating the tensors to inform
torch that we will compute some gradients with respect to these tensors.

```ocaml
  let ws = Tensor.zeros [image_dim; label_count] ~requires_grad:true in
  let bs = Tensor.zeros [label_count] ~requires_grad:true in
```

Using these the model is defined as multiplying an input by the weight matrix
and adding the bias. A softmax function is used to transform the output into a
probability distribution.

```ocaml
  let model xs = Tensor.(softmax (mm xs ws + bs)) in
```

We use gradient descent to minimize cross-entropy with respect to variables
`ws` and `bs` and iterate this a couple hundred times.

Rather than using an optimizer we perform the gradient descent updates manually.
This is only to illustrate how gradients can be computed and used. Other examples
such as `nn.ml` or `conv.ml` use an Adam optimizer.

```ocaml
  for index = 1 to 200 do
    (* Compute the cross-entropy loss. *)
    let loss = Tensor.(mean (- train_labels * log (model train_images +f 1e-6))) in

    (* Compute the gradients via backpropagation. *)
    Tensor.backward loss;

    (* Apply gradient descent, disable gradient tracking for these. *)
    Tensor.(no_grad (fun () ->
        ws -= grad ws * learning_rate;
        bs -= grad bs * learning_rate));
    Tensor.zero_grad ws;
    Tensor.zero_grad bs;
```

Running this code should build a model that has ~92% accuracy.

## A Simple Neural-Network

The code can be found in `nn.ml`, accuracy should reach ~96%.

## Convolutional Neural-Network

The code can be found in `conv.ml`, accuracy should reach ~99%.

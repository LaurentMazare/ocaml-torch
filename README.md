# ocaml-torch
__ocaml-torch__ provides some ocaml bindings for the [PyTorch](https://pytorch.org) tensor library.
This brings to OCaml NumPy-like tensor computations with GPU acceleration and tape-based automatic
differentiation.
These bindings use the [PyTorch C++ API](https://pytorch.org/cppdocs/) and are mostly automatically generated.
Note that until PyTorch reaches 1.0 there are no stable releases for libtorch so there
may be some compilation issues.

## Installation

### Option 1: Using PyTorch Conda package (recommended)
Conda packages for PyTorch 1.0 (preview release) can be used via the following command.
```bash
conda create -n torch
source activate torch
conda install pytorch-nightly-cpu=1.0.0.dev20181116 -c pytorch
# Or for the CUDA version
# conda install pytorch-nightly=1.0.0.dev20181116 -c pytorch

git clone https://github.com/LaurentMazare/ocaml-torch.git
cd ocaml-torch
make all
```

### Option 2: Using PyTorch pre-built Binaries
The libtorch library can be downloaded from the [PyTorch
website](https://pytorch.org/resources) ([latest cpu
version](https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip)).

Download and extract the libtorch library then to build all the examples run:

```bash
export LIBTORCH=/path/to/libtorch
git clone https://github.com/LaurentMazare/ocaml-torch.git
cd ocaml-torch
make all
```

You can then test that everything works well with the following example:
```bash
./_build/default/examples/basics/torch_tensor.exe
```

## Examples and Tutorials

Below is an example of a linear model trained on the MNIST dataset ([full
code](https://github.com/LaurentMazare/ocaml-torch/blob/master/examples/mnist/linear.ml)).

```ocaml
  (* Create two tensors to store model weights. *)
  let ws = Tensor.zeros [image_dim; label_count] ~requires_grad:true in
  let bs = Tensor.zeros [label_count] ~requires_grad:true in

  let model xs = Tensor.(mm xs ws + bs) in
  for index = 1 to 100 do
    (* Compute the cross-entropy loss. *)
    let loss =
      Tensor.cross_entropy_for_logits (model train_images) ~targets:train_labels
    in

    Tensor.backward loss;

    (* Apply gradient descent, disable gradient tracking for these. *)
    Tensor.(no_grad (fun () ->
        ws -= grad ws * f learning_rate;
        bs -= grad bs * f learning_rate));

    (* Compute the validation error. *)
    let test_accuracy =
      Tensor.(sum (argmax (model test_images) = test_labels) |> float_value)
      |> fun sum -> sum /. test_samples
    in
    printf "%d %f %.2f%%\n%!" index (Tensor.float_value loss) (100. *. test_accuracy);
  end

```

* A more detailed [MNIST tutorial](https://github.com/LaurentMazare/ocaml-torch/tree/master/examples/mnist).
* Some [ResNet examples on CIFAR-10](https://github.com/LaurentMazare/ocaml-torch/tree/master/examples/cifar).
* A simplified version of
  [char-rnn](https://github.com/LaurentMazare/ocaml-torch/blob/master/examples/char_rnn)
  illustrating character level language modeling using Recurrent Neural Networks.
* [Generative Adverserial Networks](https://github.com/LaurentMazare/ocaml-torch/blob/master/examples/gan).
* [Finetuning a ResNet-18 model](https://github.com/LaurentMazare/ocaml-torch/blob/master/examples/pretrained/finetuning.ml).

## Models and Weights

Various pre-trained computer vision models are implemented in the
[vision library](https://github.com/LaurentMazare/ocaml-torch/tree/master/src/vision).
The weight files can be downloaded at the following links:

* ResNet-18 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/resnet18.ot).
* ResNet-34 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/resnet34.ot).
* ResNet-50 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/resnet50.ot).
* ResNet-101 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/resnet101.ot).
* ResNet-152 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/resnet152.ot).
* DenseNet-121 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/densenet121.ot).
* DenseNet-161 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/densenet161.ot).
* DenseNet-169 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/densenet169.ot).
* SqueezeNet 1.0 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/squeezenet1_0.ot).
* SqueezeNet 1.1 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/squeezenet1_1.ot).
* VGG-13 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/vgg13.ot).
* VGG-16 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/vgg16.ot).

Running the pre-trained models on some sample images can the easily be done via the following commands.
```bash
make all
_build/default/examples/pretrained/predict.exe path/to/resnet18.ot tiger.jpg
```

## Interactive Mode

__ocaml-torch__ works well with utop and jupyter, just run `make utop` or `make jupyter`.

## TODO

* Use a GADT to add type constraints to tensor elements.
* Make it easier to use/import datasets.
* Add an opam package (this may have to wait until libtorch has stable releases).

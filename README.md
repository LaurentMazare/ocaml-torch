# ocaml-torch
__ocaml-torch__ provides some ocaml bindings for the [PyTorch](https://pytorch.org) tensor library.
This brings to OCaml NumPy-like tensor computations with GPU acceleration and tape-based automatic
differentiation.
These bindings use the [PyTorch C++ API](https://pytorch.org/cppdocs/) and are mostly automatically generated.


The libtorch library can be downloaded from the [PyTorch website](https://pytorch.org/resources) ([latest cpu version](https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip)).
Note that until PyTorch reaches 1.0 there are no stable releases for libtorch so there
may be some compilation issues.

## Usage
Extract the libtorch library in a `LIBTORCH` directory then to build examples run:

```bash
LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH \
LIBRARY_PATH=$LIBTORCH/lib:$LIBRARY_PATH \
CPATH=$LIBTORCH/include:libtorch/include/torch/csrc/api/include:$CPATH \
make all
```

After that examples can be run with:
```bash
LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH \
./_build/default/examples/basics/torch_tensor.exe
```

## Examples and Tutorials

Below is an example of a linear model trained on the MNIST dataset ([full
code](https://github.com/LaurentMazare/ocaml-torch/blob/master/examples/mnist/linear.ml)).

```ocaml
  (* Create two tensors to store model weights. *)
  let ws = Tensor.zeros [image_dim; label_count] ~requires_grad:true in
  let bs = Tensor.zeros [label_count] ~requires_grad:true in

  let model xs = Tensor.(softmax (mm xs ws + bs)) in
  for index = 1 to 100 do
    (* Compute the cross-entropy loss. *)
    let loss = Tensor.(mean (- train_labels * log (model train_images +f 1e-6))) in

    Tensor.backward loss;

    (* Apply gradient descent, disable gradient tracking for these. *)
    Tensor.(no_grad ws ~f:(fun ws -> ws -= grad ws *f learning_rate));
    Tensor.(no_grad bs ~f:(fun bs -> bs -= grad bs *f learning_rate));

    (* Compute the validation error. *)
    let test_accuracy =
      Tensor.(sum (argmax (model test_images) = argmax test_labels) |> float_value)
      |> fun sum -> sum /. test_samples
    in
    printf "%d %f %.2f%%\n%!" index (Tensor.float_value loss) (100. *. test_accuracy);
  end

```

* A more detailed [MNIST tutorial](https://github.com/LaurentMazare/ocaml-torch/tree/master/examples/mnist).
* [Generative Adverserial Networks](https://github.com/LaurentMazare/ocaml-torch/blob/master/examples/gan).

## TODO

* Use a GADT to add type constraints to tensor elements.
* Make it easier to use/import datasets.
* Model weights import.
* Add more complex examples.
    * RNN/LSTM.
    * Deep CNN on CIFAR-10 or ImageNet, maybe a ResNet ?
* Add an opam package (this may have to wait until libtorch has stable releases).
* utop/jupyter integration.

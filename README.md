# ocaml-torch
Experimental [PyTorch](https://pytorch.org) bindings in ocaml using [C++ API](https://pytorch.org/cppdocs/).
The libtorch library can be downloaded from the [PyTorch website](https://pytorch.org/resources) ([latest cpu version](https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip)).

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

## Examples

These bindings are very much a work in progress. However it should already
be possible to train a linear model on the MNIST dataset using them.
This is done in the following [example](https://github.com/LaurentMazare/ocaml-torch/blob/master/examples/mnist/linear.ml).

```ocaml
  (* Create two tensors to store model weights. *)
  let ws = Tensor.zeros Mnist_helper. [image_dim; label_count] ~requires_grad:true in
  let bs = Tensor.zeros [Mnist_helper.label_count] ~requires_grad:true in

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

## TODO

* Add more operators and more complex examples.
* Wrap some optimizer, e.g. Adam.

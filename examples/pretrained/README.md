# Transfer Learning Tutorial

This tutorial follows the lines of the
[PyTorch Transfer Learning Tutorial](https://github.com/LaurentMazare/ocaml-torch.git).

We will use transfer learning to leverage a pretrained ResNet model on a small dataset.
This dataset is made of images of ants and bees that we want to classify,
there are roughly 240 training images and 150 validation images each of them of size
224x224. The dataset is so small that training even the simplest convolutional neural
network on it would be very difficult.

Instead the original tutorial proposes two alternatives to train the classifier.

- *Finetuning the pretrained model.* We start from a ResNet-18 model pretrained on imagenet
1000 categories, replace the last layer by a binary classifier and train the resulting model
as usual.
- *Using a pretrained model as feature extractor.* The pretrained model weights are frozen and
we run this model and store the outputs of the last layer before the final classifier.
We then train a binary classifier on the resulting features.

We will focus on the second alternative but first we need to get the code
building and running and we also have to download the dataset and pretrained
weights.

## Installation Instructions
Run the following commands to download the latest ocaml-torch version
and install its dependencies (including the cpu version of libtorch).

```bash
git clone https://github.com/LaurentMazare/ocaml-torch.git
cd ocaml-torch
opam install . --deps-only
```

The ants and bees dataset can be downloaded [here](https://download.pytorch.org/tutorial/hymenoptera_data.zip).
You can download the weights for a ResNet-18 network pretrained on imagenet,
[resnet18.ot](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/resnet18.ot).

Once this is done and the dataset has been extracted we can build and run the code with:
```bash
dune build examples/pretrained/finetuning.exe
dune exec examples/pretrained/finetuning.exe resnet18.ot hymenoptera_data
```

## Loading the Data

Let us now have a look at the code from `finetuning.ml`.
The dataset is loaded via some helper functions.

```ocaml
  let dataset = Imagenet.load_dataset ~dir:Sys.argv.(2) ~classes () in
  Dataset_helper.print_summary dataset;
```

The `print_summary` function prints the dimensions of the tensors that have
been created. For training the tensor has shape `243x3x224x224`, this
corresponds to 243 images of height and width both 224 with 3 channels
(PyTorch uses the NCHW ordering for image data). The testing image
tensor has dimensions `152x3x224x224` so there are 152 images with the
same size as used in training.


## Using a Pretrained ResNet as Feature Extractor

The pixel data from the dataset is converted to features by running
a pre-trained ResNet model. This is done in the following function:

```ocaml
let precompute_activations dataset ~model_path =
  let dataset =
    let frozen_vs = Var_store.create ~frozen:true ~name:"rn" () in
    let pretrained_model = Resnet.resnet18 frozen_vs in
    Serialize.load_multi_
      ~named_tensors:(Var_store.all_vars frozen_vs) ~filename:model_path;
    Dataset_helper.map dataset ~batch_size:4 ~f:(fun _ ~batch_images ~batch_labels ->
      let activations = Layer.apply_ pretrained_model batch_images ~is_training:false in
      Tensor.copy activations, batch_labels)
  in
  Dataset_helper.print_summary dataset;
  dataset
```

This snippet performs the following steps:
- A variable store `frozen_vs` is created. Variable stores are used to hold
  trainable variables. However in this case no training is performed on the
  variables so we use `~frozen:true` which should slightly speed-up the model
  evaluation.
- A ResNet-18 model is created using this variable store. At this point the
  model weights are randomly initialized.
- `Serialize.load_multi_` loads the weights stored in a given file and copy their values
  to some tensors. Tensors are named in the serialized file in a way that matches
  the names we used when creating the ResNet model.
- Finally for each tensor of the training and testing datasets, `Layer.apply_ pretrained_model`
  performs a forward pass on the model and returns the resulting tensor. In this
  case the result is a vector of 512 values per sample.

## Training a Linear Layer on top of the Extracted Features

Now that we have precomputed the output of the ResNet model on our training and
testing images we will train a linear binary classifier to recognize ants vs bees.

We start by defining a model, for this we need a variable store to hold the
trainable variables.

```ocaml
  let train_vs = Var_store.create ~name:"rn-vs" () in
  let fc1 = Layer.linear train_vs ~input_dim:512 (List.length classes) in
  let model xs = Layer.apply fc1 xs in
```

We will use stochastic gradient descent to minimize the cross-entropy loss
on the classification task. To do this we create a `sgd` optimizer and then
iterate on the training dataset. After each epoch the accuracy is computed
on the testing set and printed.

```ocaml
  let sgd = Optimizer.sgd train_vs ~learning_rate:0.001 ~momentum:0.9 in
  for epoch_idx = 1 to 20 do
    Dataset_helper.iter dataset ~batch_size ~f:(fun _ ~batch_images ~batch_labels ->
      (* Perform a training step. *)
    );
    (* Compute and print the validation accuracy. *)
  done
```

On each training step the model output is computed through a forward pass. The
cross-entropy loss is then evaluated on the resulting logits using the training labels.
The backward pass then evaluates gradients for the trainable variables of our
model and these variables are updated by the optimizer.
```ocaml
      (* Perform a training step. *)
      let predicted = model batch_images in
      let loss = Tensor.cross_entropy_for_logits predicted ~targets:batch_labels in
      Optimizer.backward_step sgd ~loss);
```

After each epoch the accuracy is evaluated on the testing set and printed out.
```ocaml
    (* Compute and print the validation accuracy. *)
    let test_accuracy =
      Dataset_helper.batch_accuracy dataset `test ~batch_size ~predict:model
    in
    Stdio.printf "%3d   test accuracy: %.2f%%\n%!"
      epoch_idx
      (100. *. test_accuracy);
```

This should result in a `94%` accuracy on the testing set.
The whole code for this example can be found in [finetune.ml](finetune.ml).

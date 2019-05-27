(* Finetuning an ImageNet trained model.
   This is based on the PyTorch tutorial "Finetuning Torchvision Models".
   The final accuracy is ~94%.
     https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

   The ants and bees dataset can be found at the following link.
     https://download.pytorch.org/tutorial/hymenoptera_data.zip

   Network weights can be found here:
     https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/resnet18.ot
*)

open Base
open Torch
open Torch_vision

let batch_size = 8
let classes = [ "ants"; "bees" ]

(* Precompute the last layer of the pre-trained model on the whole dataset. *)
let precompute_activations dataset ~model_path =
  let dataset =
    let frozen_vs = Var_store.create ~frozen:true ~name:"rn" () in
    let pretrained_model = Resnet.resnet18 frozen_vs in
    Stdio.printf "Loading weights from %s.\n%!" model_path;
    Serialize.load_multi_
      ~named_tensors:(Var_store.all_vars frozen_vs)
      ~filename:model_path;
    Stdio.printf "Precomputing activations, this can take a minute...\n%!";
    Dataset_helper.map dataset ~batch_size:4 ~f:(fun _ ~batch_images ~batch_labels ->
        let activations =
          Layer.apply_ pretrained_model batch_images ~is_training:false
        in
        Tensor.copy activations, batch_labels)
  in
  Dataset_helper.print_summary dataset;
  dataset

let () =
  if Array.length Sys.argv <> 3
  then Printf.failwithf "usage: %s resnet18.ot dataset-path" Sys.argv.(0) ();
  let dataset = Imagenet.load_dataset ~dir:Sys.argv.(2) ~classes () in
  Dataset_helper.print_summary dataset;
  let dataset = precompute_activations dataset ~model_path:Sys.argv.(1) in
  let train_vs = Var_store.create ~name:"rn-vs" () in
  let fc1 = Layer.linear train_vs ~input_dim:512 (List.length classes) in
  let model xs = Layer.apply fc1 xs in
  let sgd = Optimizer.sgd train_vs ~learning_rate:0.001 ~momentum:0.9 in
  for epoch_idx = 1 to 20 do
    let sum_loss = ref 0. in
    Dataset_helper.iter dataset ~batch_size ~f:(fun _ ~batch_images ~batch_labels ->
        let predicted = model batch_images in
        (* Compute the cross-entropy loss. *)
        let loss = Tensor.cross_entropy_for_logits predicted ~targets:batch_labels in
        sum_loss := !sum_loss +. Tensor.float_value loss;
        Optimizer.backward_step sgd ~loss);
    (* Compute the validation error. *)
    let test_accuracy =
      Dataset_helper.batch_accuracy dataset `test ~batch_size ~predict:model
    in
    Stdio.printf
      "%3d   train loss: %f   test accuracy: %.2f%%\n%!"
      epoch_idx
      (!sum_loss /. Float.of_int (Dataset_helper.batches_per_epoch dataset ~batch_size))
      (100. *. test_accuracy)
  done

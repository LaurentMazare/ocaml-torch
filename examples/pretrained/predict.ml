(* Evaluation using a pre-trained ResNet model.
   The pre-trained weight file can be found at:
     https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/resnet18.ot
     https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/resnet34.ot
     https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/densenet121.ot
     https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/vgg13.ot
     https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/vgg16.ot
     https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/vgg19.ot
     https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/squeezenet1_0.ot
     https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/squeezenet1_1.ot
     https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/alexnet.ot
     https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/inception-v3.ot
     https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/mobilenet-v2.ot
     https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/efficientnet-b0.ot
     https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/efficientnet-b1.ot
     https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/efficientnet-b2.ot
     https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/efficientnet-b3.ot
     https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/efficientnet-b4.ot
*)
open Base
open Torch
open Torch_vision

let () =
  let module Sys = Caml.Sys in
  if Array.length Sys.argv <> 3
  then Printf.failwithf "usage: %s resnet18.ot input.png" Sys.argv.(0) ();
  let image = Imagenet.load_image Sys.argv.(2) in
  let vs = Var_store.create ~name:"rn" ~device:Cpu () in
  let model =
    match Caml.Filename.basename Sys.argv.(1) with
    | "vgg11.ot" -> Vgg.vgg11 vs ~num_classes:1000
    | "vgg13.ot" -> Vgg.vgg13 vs ~num_classes:1000
    | "vgg16.ot" -> Vgg.vgg16 vs ~num_classes:1000
    | "vgg19.ot" -> Vgg.vgg19 vs ~num_classes:1000
    | "squeezenet1_0.ot" -> Squeezenet.squeezenet1_0 vs ~num_classes:1000
    | "squeezenet1_1.ot" -> Squeezenet.squeezenet1_1 vs ~num_classes:1000
    | "densenet121.ot" -> Densenet.densenet121 vs ~num_classes:1000
    | "densenet161.ot" -> Densenet.densenet161 vs ~num_classes:1000
    | "densenet169.ot" -> Densenet.densenet169 vs ~num_classes:1000
    | "resnet34.ot" -> Resnet.resnet34 vs ~num_classes:1000
    | "resnet50.ot" -> Resnet.resnet50 vs ~num_classes:1000
    | "resnet101.ot" -> Resnet.resnet101 vs ~num_classes:1000
    | "resnet152.ot" -> Resnet.resnet152 vs ~num_classes:1000
    | "resnet18.ot" -> Resnet.resnet18 vs ~num_classes:1000
    | "mobilenet-v2.ot" -> Mobilenet.v2 vs ~num_classes:1000
    | "alexnet.ot" -> Alexnet.alexnet vs ~num_classes:1000
    | "inception-v3.ot" -> Inception.v3 vs ~num_classes:1000
    | "efficientnet-b0.ot" -> Efficientnet.b0 vs ~num_classes:1000
    | "efficientnet-b1.ot" -> Efficientnet.b1 vs ~num_classes:1000
    | "efficientnet-b2.ot" -> Efficientnet.b2 vs ~num_classes:1000
    | "efficientnet-b3.ot" -> Efficientnet.b3 vs ~num_classes:1000
    | "efficientnet-b4.ot" -> Efficientnet.b4 vs ~num_classes:1000
    | "efficientnet-b5.ot" -> Efficientnet.b5 vs ~num_classes:1000
    | "efficientnet-b6.ot" -> Efficientnet.b6 vs ~num_classes:1000
    | "efficientnet-b7.ot" -> Efficientnet.b7 vs ~num_classes:1000
    | otherwise ->
      Printf.failwithf "unsupported model %s, try with resnet18.ot" otherwise ()
  in
  Stdio.printf "Loading weights from %s\n%!" Sys.argv.(1);
  Serialize.load_multi_ ~named_tensors:(Var_store.all_vars vs) ~filename:Sys.argv.(1);
  let probabilities =
    Layer.forward_ model image ~is_training:false
    |> Tensor.softmax ~dim:(-1) ~dtype:(T Float)
  in
  Imagenet.Classes.top probabilities ~k:5
  |> List.iter ~f:(fun (name, probability) ->
         Stdio.printf "%s: %.2f%%\n%!" name (100. *. probability))

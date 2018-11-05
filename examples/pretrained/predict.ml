(* Evaluation using a pre-trained ResNet model. *)
open Base
open Torch

let () =
  if Array.length Sys.argv <> 3
  then Printf.sprintf "usage: %s weights.ot input.png" Sys.argv.(0) |> failwith;
  let device =
    if Cuda.is_available ()
    then begin
      Stdio.printf "Using cuda, devices: %d\n%!" (Cuda.device_count ());
      Torch_core.Device.Cuda
    end else Torch_core.Device.Cpu
  in
  let image = Imagenet.load_image Sys.argv.(2) in
  let vs = Var_store.create ~name:"rn" ~device () in
  let model = Resnet.resnet18 vs ~num_classes:1000 in
  Stdio.printf "Loading weights from %s\n%!" Sys.argv.(1);
  Serialize.load_multi_ ~named_tensors:(Var_store.all_vars vs) ~filename:Sys.argv.(1);
  let probabilities =
    Layer.apply_ model image ~is_training:false
    |> Tensor.softmax
  in
  Imagenet.Classes.top probabilities ~k:5
  |> List.iter ~f:(fun (name, probability) ->
    Stdio.printf "%s: %.2f%%\n%!" name (100. *. probability))

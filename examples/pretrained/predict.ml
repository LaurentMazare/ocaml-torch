(* Evaluation using a pre-trained ResNet model. *)
open Base
open Torch

let load_image image_file =
  let image = ImageLib.openfile image_file in
  Stdio.printf "w: %d h: %d  %d\n%!" image.width image.height image.max_val;
  match image.pixels with
  | RGB (Pix8 red, Pix8 green, Pix8 blue) ->
    let convert pixels kind =
      let imagenet_mean, imagenet_std =
        match kind with
        | `blue -> 0.485, 0.229
        | `green -> 0.456, 0.224
        | `red -> 0.406, 0.225
      in
      Bigarray.genarray_of_array2 pixels
      |> Tensor.of_bigarray
      (* Crop/resize to 224x224 or use adaptive pooling ? *)
      |> Tensor.to_type ~type_:Float
      |> fun xs -> Tensor.((xs / f 255. - f imagenet_mean) / f imagenet_std)
    in
    let image =
      Tensor.stack [ convert blue `blue; convert green `green; convert red `red ] ~dim:0
      |> Tensor.transpose ~dim0:1 ~dim1:2
    in
    Tensor.view image ~size:(1 :: Tensor.shape image)
  | _ -> failwith "unexpected pixmaps"

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
  let vs = Var_store.create ~name:"rn" ~device () in
  let model = Resnet.resnet18 vs ~num_classes:1000 in
  Stdio.printf "Loading weights from %s\n%!" Sys.argv.(1);
  Serialize.load_multi_ ~named_tensors:(Var_store.all_vars vs) ~filename:Sys.argv.(1);
  let image = load_image Sys.argv.(2) in
  let probabilities =
    Layer.apply_ model image ~is_training:false
    |> Tensor.softmax
  in
  let a, b = Tensor.topk probabilities ~k:5 ~dim:1 ~largest:true ~sorted:true in
  Tensor.print a;
  Tensor.print b

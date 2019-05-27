(* This is inspired by the Neural Style tutorial from PyTorch.org
   https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
*)
open Base
open Torch
open Torch_vision

let style_weight = 1e6
let learning_rate = 1e-1
let total_steps = 3000
let style_indexes = [ 0; 2; 5; 7; 10 ]
let content_indexes = [ 7 ]

let gram_matrix m =
  let a, b, c, d = Tensor.shape4_exn m in
  let m = Tensor.view m ~size:[ a * b; c * d ] in
  let g = Tensor.mm m (Tensor.tr m) in
  Tensor.( / ) g (Float.of_int (a * b * c * d) |> Tensor.f)

let style_loss m1 m2 = Tensor.mse_loss (gram_matrix m1) (gram_matrix m2)

let load_pretrained_vgg ~filename ~device =
  let vs = Var_store.create ~name:"vgg" ~device () in
  let max_layer = 1 + List.reduce_exn (style_indexes @ content_indexes) ~f:Int.max in
  let model = Vgg.vgg16_layers vs ~max_layer ~batch_norm:false |> Staged.unstage in
  Stdio.printf "Loading weights from %s\n%!" filename;
  Serialize.load_multi_ ~named_tensors:(Var_store.all_vars vs) ~filename;
  Var_store.freeze vs;
  model

let () =
  let device = Device.cuda_if_available () in
  let style_img, content_img, filename =
    match Sys.argv with
    | [| _; style_img; content_img; filename |] -> style_img, content_img, filename
    | _ ->
      Printf.failwithf "usage: %s style_img.png content_img.png vgg16.ot" Sys.argv.(0) ()
  in
  let model = load_pretrained_vgg ~filename ~device in
  let style_img =
    Imagenet.load_image_no_resize_and_crop style_img |> Tensor.to_device ~device
  in
  let content_img =
    Imagenet.load_image_no_resize_and_crop content_img |> Tensor.to_device ~device
  in
  let style_layers, content_layers =
    let detach = Map.map ~f:Tensor.detach in
    Tensor.no_grad (fun () -> model style_img |> detach, model content_img |> detach)
  in
  let vs = Var_store.create ~name:"optim" ~device () in
  let input_var = Var_store.new_var_copy vs ~src:content_img ~name:"in" in
  let optimizer = Optimizer.adam vs ~learning_rate in
  for step_idx = 1 to total_steps do
    Optimizer.zero_grad optimizer;
    let input_layers = model input_var in
    let style_loss =
      List.map style_indexes ~f:(fun l ->
          style_loss (Map.find_exn input_layers l) (Map.find_exn style_layers l))
      |> List.reduce_exn ~f:Tensor.( + )
    in
    let content_loss =
      List.map content_indexes ~f:(fun l ->
          Tensor.mse_loss (Map.find_exn input_layers l) (Map.find_exn content_layers l))
      |> List.reduce_exn ~f:Tensor.( + )
    in
    let loss = Tensor.((style_loss * f style_weight) + content_loss) in
    Tensor.backward loss;
    Optimizer.step optimizer;
    Tensor.no_grad (fun () -> ignore (Imagenet.clamp_ input_var : Tensor.t));
    Stdio.printf
      "%d %.4f %.4f %.4f\n%!"
      step_idx
      (Tensor.float_value loss)
      (Tensor.float_value style_loss)
      (Tensor.float_value content_loss);
    Caml.Gc.full_major ();
    if step_idx % 10 = 0
    then Imagenet.write_image input_var ~filename:(Printf.sprintf "out%d.png" step_idx)
  done;
  Imagenet.write_image input_var ~filename:"out.png"

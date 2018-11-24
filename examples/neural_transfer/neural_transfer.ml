(* This is inspired by the Neural Style tutorial from PyTorch.org
   https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
*)
open Base
open Torch
open Torch_vision

let gram_matrix m =
  let a, b, c, d = Tensor.shape4_exn m in
  let m = Tensor.view m ~size:[ a * b; c * d ] in
  let g = Tensor.mm m (Tensor.tr m) in
  Tensor.(/) g (Float.of_int (a * b * c * d) |> Tensor.f)

let style_loss m1 m2 =
  Tensor.mse_loss (gram_matrix m1) (gram_matrix m2)

let load_pretrained_vgg ~filename =
  let vs = Var_store.create ~name:"vgg" () in
  let model = Vgg.vgg16_layers vs ~batch_norm:false |> Staged.unstage in
  Stdio.printf "Loading weights from %s\n%!" filename;
  Serialize.load_multi_ ~named_tensors:(Var_store.all_vars vs) ~filename;
  model

let () =
  let style_img, content_img, filename =
    match Sys.argv with
    | [| _; style_img; content_img; filename |] -> style_img, content_img, filename
    | _ ->
      Printf.failwithf "usage: %s style_img.png content_img.png vgg16.ot" Sys.argv.(0) ()
  in
  let model = load_pretrained_vgg ~filename in
  let style_img = Imagenet.load_image style_img in
  let content_img = Imagenet.load_image content_img in
  let style_layers = model style_img in
  let content_layers = model content_img in
  let optimizer = Optimizer.adam (failwith "TODO") ~learning_rate:1e-4 in
  for step_idx = 1 to 300 do
    Optimizer.zero_grad optimizer;
    let loss =
      ignore (style_layers, content_layers, style_loss);
      failwith "TODO"
    in
    Tensor.backward loss;
    Optimizer.step optimizer;
    Stdio.printf "%d\n" step_idx;
  done

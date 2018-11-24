(* This is inspired by the Neural Style tutorial from PyTorch.org
   https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
*)
open Base
open Torch
open Torch_vision

let style_weight = 1e6

let style_indexes = [ 0; 2; 5; 7; 10 ]
let content_indexes = [ 7 ]

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

  let vs = Var_store.create ~name:"optim" () in
  let input_var = Var_store.new_var_copy vs ~src:content_img ~name:N.(root / "in") in
  let optimizer = Optimizer.adam vs ~learning_rate:1e-4 in
  for step_idx = 1 to 300 do
    Optimizer.zero_grad optimizer;
    let input_layers = model input_var in
    let style_loss =
      List.fold style_indexes ~init:(Tensor.f 0.) ~f:(fun acc l ->
          let input_ = Map.find_exn input_layers l in
          let style = Map.find_exn style_layers l in
          Tensor.(acc + style_loss input_ style))
    in
    let content_loss =
      List.fold content_indexes ~init:(Tensor.f 0.) ~f:(fun acc l ->
          let input_ = Map.find_exn input_layers l in
          let content = Map.find_exn content_layers l in
          Tensor.(acc + mse_loss input_ content))
    in
    let loss = Tensor.(style_loss * f style_weight + content_loss) in
    Tensor.backward loss;
    Optimizer.step optimizer;
    Tensor.no_grad (fun () ->
        (* TODO: clamp to 0/1. *)
        ());
    Stdio.printf "%d %.4f %.4f\n"
      step_idx (Tensor.float_value style_loss) (Tensor.float_value content_loss);
    Caml.Gc.full_major ();
  done;
  Imagenet.write_image input_var ~filename:"out.png"

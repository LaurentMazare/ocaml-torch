(* Inference code from Progressive Growing of GANs.

   This is based on the PyTorch implementation:
     https://github.com/ptrblck/prog_gans_pytorch_inference
   Which itself is based on the Theano/Lasagne implementation of
     Progressive Growing of GANs paper from tkarras:
     https://github.com/tkarras/progressive_growing_of_gans
*)
open Base
open Torch

let leaky_relu xs = Tensor.(max xs (xs * f 0.2))
let pixel_norm xs = Tensor.(xs / (sqrt (mean2 (xs * xs) ~dim:1 ~keepdim:true) + f 1e-8))

let w_scale_layer vs ~n ~size =
  let name s = N.(n / "wscale" / s) in
  let scale_ = Var_store.new_var vs ~shape:[1] ~init:Zeros ~name:(name "scale") in
  let b = Var_store.new_var vs ~shape:[size] ~init:Zeros ~name:(name "b") in
  Layer.of_fn (fun xs ->
      let b = Tensor.view b ~size:[1; size; 1; 1] in
      let x0, _, x2, x3 = Tensor.shape4_exn xs in
      Tensor.((xs * scale_) + expand b ~implicit:false ~size:[x0; size; x2; x3]) )

let norm_conv_block vs ~n ~ksize ~padding ~input_dim dim =
  let conv =
    Layer.conv2d_
      vs
      ~n:N.(n / "conv")
      ~ksize
      ~stride:1
      ~padding
      ~use_bias:false
      ~input_dim
      dim
  in
  let wscale = w_scale_layer vs ~n ~size:dim in
  Layer.of_fn (fun xs ->
      pixel_norm xs |> Layer.apply conv |> Layer.apply wscale |> leaky_relu )

let norm_upscale_conv_block vs ~n ~ksize ~padding ~input_dim dim =
  let conv =
    Layer.conv2d_
      vs
      ~n:N.(n / "conv")
      ~ksize
      ~stride:1
      ~padding
      ~use_bias:false
      ~input_dim
      dim
  in
  let wscale = w_scale_layer vs ~n ~size:dim in
  Layer.of_fn (fun xs ->
      let _, _, h, w = Tensor.shape4_exn xs in
      pixel_norm xs
      |> Tensor.upsample_nearest2d ~output_size:[2 * h; 2 * w]
      |> Layer.apply conv
      |> Layer.apply wscale
      |> leaky_relu )

let create_generator vs =
  let features =
    List.mapi
      ~f:(fun i l -> l ~n:N.(root / "features" / Int.to_string i))
      [ norm_conv_block vs ~ksize:4 ~padding:3 ~input_dim:512 512
      ; norm_conv_block vs ~ksize:3 ~padding:1 ~input_dim:512 512
      ; norm_upscale_conv_block vs ~ksize:3 ~padding:1 ~input_dim:512 512
      ; norm_conv_block vs ~ksize:3 ~padding:1 ~input_dim:512 512
      ; norm_upscale_conv_block vs ~ksize:3 ~padding:1 ~input_dim:512 512
      ; norm_conv_block vs ~ksize:3 ~padding:1 ~input_dim:512 512
      ; norm_upscale_conv_block vs ~ksize:3 ~padding:1 ~input_dim:512 512
      ; norm_conv_block vs ~ksize:3 ~padding:1 ~input_dim:512 512
      ; norm_upscale_conv_block vs ~ksize:3 ~padding:1 ~input_dim:512 256
      ; norm_conv_block vs ~ksize:3 ~padding:1 ~input_dim:256 256
      ; norm_upscale_conv_block vs ~ksize:3 ~padding:1 ~input_dim:256 128
      ; norm_conv_block vs ~ksize:3 ~padding:1 ~input_dim:128 128
      ; norm_upscale_conv_block vs ~ksize:3 ~padding:1 ~input_dim:128 64
      ; norm_conv_block vs ~ksize:3 ~padding:1 ~input_dim:64 64
      ; norm_upscale_conv_block vs ~ksize:3 ~padding:1 ~input_dim:64 32
      ; norm_conv_block vs ~ksize:3 ~padding:1 ~input_dim:32 32
      ; norm_upscale_conv_block vs ~ksize:3 ~padding:1 ~input_dim:32 16
      ; norm_conv_block vs ~ksize:3 ~padding:1 ~input_dim:16 16 ]
    |> Layer.fold
  in
  let conv =
    Layer.conv2d_
      vs
      ~n:N.(root / "output" / "conv")
      ~ksize:1
      ~stride:1
      ~padding:0
      ~use_bias:false
      ~input_dim:16
      3
  in
  let wscale = w_scale_layer vs ~n:N.(root / "output") ~size:3 in
  Layer.of_fn (fun xs ->
      Layer.apply features xs |> pixel_norm |> Layer.apply conv |> Layer.apply wscale )

let () =
  if Array.length Sys.argv <> 2
  then Printf.failwithf "usage: %s prog-gan.ot" Sys.argv.(0) ();
  Torch_core.Wrapper.manual_seed 42;
  let vs = Var_store.create ~name:"vs" () in
  let generator = create_generator vs in
  Serialize.load_multi_ ~named_tensors:(Var_store.all_vars vs) ~filename:Sys.argv.(1);
  for i = 1 to 50 do
    let image = Tensor.(randn [1; 512; 1; 1]) |> Layer.apply generator in
    let image = Tensor.(image - minimum image) in
    let image = Tensor.(image / maximum image * f 255.) in
    Torch_vision.Image.write_image image ~filename:(Printf.sprintf "out%d.png" i);
    Caml.Gc.full_major ()
  done

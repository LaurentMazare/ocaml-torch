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

let pixel_norm xs =
  Tensor.(xs / (sqrt (mean4 (xs * xs) ~dim:1 ~keepdim:true) + f 1e-8))

let w_scale_layer _vs ~size =
  let scale = Tensor.randn [ 1 ] in
  let b = Tensor.randn [ size ] |> Tensor.view ~size:[ 1; size; 1; 1 ] in
  Layer.of_fn (fun xs ->
      let x0, _, x2, x3 = Tensor.shape4_exn xs in
      Tensor.(xs * scale + expand b ~implicit:true ~size:[ x0; size; x2; x3 ]))

let norm_conv_block vs ~ksize ~padding ~input_dim n =
  let conv = Layer.conv2d_ vs ~ksize ~stride:1 ~padding ~use_bias:false ~input_dim n in
  let wscale = w_scale_layer vs ~size:n in
  Layer.of_fn (fun xs ->
      pixel_norm xs
      |> Layer.apply conv
      |> Layer.apply wscale
      |> leaky_relu)

let norm_upscale_conv_block vs ~ksize ~padding ~input_dim n =
  let conv = Layer.conv2d_ vs ~ksize ~stride:1 ~padding ~use_bias:false ~input_dim n in
  let wscale = w_scale_layer vs ~size:n in
  Layer.of_fn (fun xs ->
      let _, _, h, w = Tensor.shape4_exn xs in
      pixel_norm xs
      |> Tensor.upsample_nearest2d ~output_size:[ 2*h; 2*w ]
      |> Layer.apply conv
      |> Layer.apply wscale
      |> leaky_relu)

let create_generator vs =
  let features =
    Layer.fold
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
      ; norm_conv_block vs ~ksize:3 ~padding:1 ~input_dim:16 16
      ]
  in
  let conv =
    Layer.conv2d_ vs ~ksize:1 ~stride:1 ~padding:0 ~use_bias:false ~input_dim:16 3
  in
  let wscale = w_scale_layer vs ~size:3 in
  Layer.of_fn (fun xs ->
      Layer.apply features xs
      |> pixel_norm
      |> Layer.apply conv
      |> Layer.apply wscale)

let () =
  if Array.length Sys.argv <> 2
  then Printf.failwithf "usage: %s weights.ot" Sys.argv.(0) ();
  let vs = Var_store.create ~name:"vs" () in
  let generator = create_generator vs in
  let image = Tensor.randn [ 1; 512; 1; 1 ] |> Layer.apply generator in
  Torch_vision.Image.write_image image ~filename:"out.png"

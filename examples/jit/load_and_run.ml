open Base
open Torch

let () =
  let model_filename, image_filename =
    match Caml.Sys.argv with
    | [| _; model_filename; image_filename |] -> model_filename, image_filename
    | argv -> Printf.failwithf "usage: %s model.pt image.png" argv.(0) ()
  in
  let model = Module.load model_filename in
  let image = Torch_vision.Imagenet.load_image image_filename in
  Module.forward model [ image ]
  |> Tensor.softmax ~dim:(-1)
  |> Torch_vision.Imagenet.Classes.top ~k:5
  |> List.iter ~f:(fun (class_name, p) ->
      Stdio.printf "%s: %f\n%!" class_name p)

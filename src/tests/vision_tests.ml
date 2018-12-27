open Base
open Torch
open Torch_vision

let%expect_test _ =
  let width, height = 512, 384 in
  let image_tensor = Tensor.zeros [ 3; height; width ] in
  let filename = Caml.Filename.temp_file "torchtest" ".png" in
  for i = 0 to height - 1 do
    for j = 0 to width - 1 do
      Tensor.(image_tensor.%.{[0; i; j]} <- Float.of_int i /. Float.of_int height);
      Tensor.(image_tensor.%.{[1; i; j]} <- Float.of_int j /. Float.of_int width);
    done;
  done;
  let image_tensor = Tensor.(image_tensor * f 255.) in
  Image.write_image image_tensor ~filename;
  let loaded_tensor =
    Image.load_image filename
    |> Or_error.ok_exn
    |> Tensor.to_type ~type_:Float
  in
  Stdio.printf "%f %f\n%!"
    Tensor.(minimum (image_tensor - loaded_tensor) |> float_value)
    Tensor.(maximum (image_tensor - loaded_tensor) |> float_value);
  [%expect{| 0.000000 0.998047 |}];
  Unix.unlink filename

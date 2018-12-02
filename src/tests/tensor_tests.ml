open Base
open Sexplib.Conv
open Torch

let array1_to_list ba =
  let ba = Bigarray.array1_of_genarray ba in
  List.init (Bigarray.Array1.dim ba) ~f:(Bigarray.Array1.get ba)

let%expect_test _ =
  let t = Tensor.(f 41. + f 1.) in
  Stdio.printf !"%{sexp:float}\n" (Tensor.to_float0_exn t);
  [%expect{|
        42
      |}];
  let t = Tensor.float_vec [ 1.; 42.; 1337. ] in
  Stdio.printf !"%{sexp:float array}\n" Tensor.(to_float1_exn (t * t));
  [%expect{|
        (1 1764 1787569)
      |}];
  Stdio.printf !"%{sexp:float array}\n" Tensor.(to_float1_exn (t + f 1.5));
  [%expect{|
        (2.5 43.5 1338.5)
      |}]

let%expect_test _ =
  let open Tensor in
  let t = zeros [4; 2] in
  t.%.{[1; 1]} <- 42.0;
  t.%.{[3; 0]} <- 1.337;
  for i = 0 to 3 do
    Stdio.printf "%f %f\n" t.%.{[i; 0]} t.%.{[i; 1]};
  done;
  [%expect{|
        0.000000 0.000000
        0.000000 42.000000
        0.000000 0.000000
        1.337000 0.000000
      |}]

let%expect_test _ =
  let open Tensor in
  let t = zeros [5; 2] in
  t += f 1.;
  narrow t ~dim:0 ~start:1 ~length:3 += f 2.;
  narrow t ~dim:1 ~start:1 ~length:1 -= f 3.;
  Stdio.printf !"%{sexp:float array array}\n" (Tensor.to_float2_exn t);
  [%expect{| ((1 -2) (3 0) (3 0) (3 0) (1 -2)) |}]

let%expect_test _ =
  let t = List.init 5 ~f:Float.of_int |> Tensor.float_vec in
  let ba = Tensor.to_bigarray t ~kind:Float32 in
  let ba_narrow =
    Tensor.narrow t ~dim:0 ~start:1 ~length:3
    |> Tensor.to_bigarray ~kind:Float32
  in
  Stdio.printf !"%{sexp:float list}\n" (array1_to_list ba);
  Stdio.printf !"%{sexp:float list}\n" (array1_to_list ba_narrow);
  [%expect{|
    (0 1 2 3 4)
    (1 2 3) |}]

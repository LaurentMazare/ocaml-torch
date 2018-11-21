open Sexplib.Conv
open Torch

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

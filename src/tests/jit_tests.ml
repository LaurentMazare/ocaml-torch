open Base
open Torch

let%expect_test _ =
  let model = Module.load "../../../../src/tests/foo.pt" in
  let output = Module.forward model [ Tensor.f 42.; Tensor.f 1337. ] in
  Stdio.printf !"%{sexp:float}\n" (Tensor.to_float0_exn output);
  [%expect{|
        1421
      |}]

open Base
open Torch

let%expect_test _ =
  let model = Module.load "../../../../src/tests/foo.pt" in
  let output = Module.forward model [ Tensor.f 42.; Tensor.f 1337. ] in
  Stdio.printf !"%{sexp:float}\n" (Tensor.to_float0_exn output);
  [%expect{|
        1421
      |}]

let%expect_test _ =
  let model = Module.load "../../../../src/tests/foo2.pt" in
  let outputs =
    Module.forward_multi model [ Tensor.f 42.; Tensor.f 1337. ] ~noutputs:2
  in
  let t1, t2 =
    match outputs with
    | [ t1; t2 ] -> t1, t2
    | _ -> assert false
  in
  Stdio.printf !"%{sexp:float} %{sexp:float}\n"
    (Tensor.to_float0_exn t1)
    (Tensor.to_float0_exn t2);
  [%expect{|
        1421 -1295
      |}]

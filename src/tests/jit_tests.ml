open Base
open Torch

let%expect_test _ =
  let model = Module.load "../../../../src/tests/foo.pt" in
  let output = Module.forward model [ Tensor.f 42.; Tensor.f 1337. ] in
  Stdio.printf !"%{sexp:float}\n" (Tensor.to_float0_exn output);
  [%expect {|
        1463
      |}]

let%expect_test _ =
  let model = Module.load "../../../../src/tests/foo2.pt" in
  let ivalue = Module.forward_ model [ Tensor (Tensor.f 42.); Tensor (Tensor.f 1337.) ] in
  Caml.Gc.full_major ();
  let t1, t2 =
    match ivalue with
    | Tuple [ Tensor t1; Tensor t2 ] -> t1, t2
    | _ -> assert false
  in
  Stdio.printf
    !"%{sexp:float} %{sexp:float}\n"
    (Tensor.to_float0_exn t1)
    (Tensor.to_float0_exn t2);
  [%expect {|
        1421 -1295
      |}]

let%expect_test _ =
  let model = Module.load "../../../../src/tests/foo3.pt" in
  let output = Module.forward model [ Tensor.of_float1 [| 1.0; 2.0; 3.0; 4.0; 5.0 |] ] in
  Stdio.printf !"%{sexp:float}\n" (Tensor.to_float0_exn output);
  [%expect {|
        120
      |}]

let%expect_test _ =
  let (ivalues : Ivalue.t list) =
    [ None
    ; Int 42
    ; Double 1.5
    ; Tuple []
    ; Tuple [ Tuple [ None; Int 42 ]; String "foo"; Double 1.5; String "bar" ]
    ; String "foobar"
    ]
  in
  List.iter ivalues ~f:(fun ivalue ->
      Ivalue.to_raw ivalue |> Ivalue.of_raw |> Ivalue.to_string |> Stdio.print_endline);
  [%expect
    {|
        none
        42
        1.5
        ()
        ((none, 42), "foo", 1.5, "bar")
        "foobar"
      |}]

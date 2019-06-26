open Base
open Torch

let write_and_read tensor ~print_tensor ~kind =
  let filename = Caml.Filename.temp_file "torchtest" ".ot" in
  Serialize.save tensor ~filename;
  let y = Serialize.load ~filename in
  let y = Tensor.extract_exn y ~kind in
  let l2 = Tensor.((tensor - y) * (tensor - y)) |> Tensor.sum in
  print_tensor l2;
  Unix.unlink filename

let%expect_test _ =
  let print_tensor tensor = Stdio.printf "%d\n" (Tensor.to_int0_exn tensor) in
  Tensor.randint ~high:42 ~size:[ 3; 1; 4 ] ~options:(T Int64, Cpu)
  |> write_and_read ~print_tensor ~kind:Int64;
  [%expect {|
        0
      |}];
  write_and_read (Tensor.of_int0 1337) ~print_tensor ~kind:Int64;
  [%expect {|
        0
      |}]

let%expect_test _ =
  let print_tensor tensor = Stdio.printf "%f\n" (Tensor.to_float0_exn tensor) in
  write_and_read (Tensor.randn [ 42; 27 ]) ~print_tensor ~kind:Float;
  [%expect {|
        0.000000
      |}];
  write_and_read (Tensor.of_float0 1337.) ~print_tensor ~kind:Float;
  [%expect {|
        0.000000
      |}]

let write_and_read named_tensors =
  let filename = Caml.Filename.temp_file "torchtest" ".ot" in
  Serialize.save_multi ~named_tensors ~filename;
  let ys = Serialize.load_multi ~names:(List.map named_tensors ~f:fst) ~filename in
  List.iter2_exn named_tensors ys ~f:(fun (name, tensor) y ->
      let (Tensor.T y) = y in
      let tensor = Tensor.extract_exn tensor ~kind:(Tensor.kind y) in
      let l2 = Tensor.((tensor - y) * (tensor - y)) |> Tensor.sum in
      match Tensor.kind l2 with
      | Int64 -> Stdio.printf "%s %d\n%!" name (Tensor.to_int0_exn l2)
      | Float -> Stdio.printf "%s %f\n%!" name (Tensor.to_float0_exn l2)
      | _ -> assert false);
  Unix.unlink filename

let%expect_test _ =
  write_and_read
    [ "tensor-1", Tensor.T (Tensor.of_float1 [| 3.; 14.; 15.; 9265.35 |])
    ; "another", Tensor.T (Tensor.of_int0 42)
    ; ( "and yet another"
      , Tensor.T (Tensor.of_int2 [| [| 3; -1; -51234 |]; [| 2718; 2818; 28 |] |]) )
    ];
  [%expect
    {|
        tensor-1 0.000000
        another 0
        and yet another 0
      |}]

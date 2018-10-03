open! Ctypes

module C = Torch_bindings.C(Torch_generated)

module Kind = struct
  type t =
    | Uint8
    | Int8
    | Int16
    | Int
    | Int64
    | Half
    | Float
    | Double
    | ComplexHalf
    | ComplexFloat
    | ComplexDouble

  (* Hardcoded, should match ScalarType.h *)
  let to_int = function
    | Uint8 -> 0
    | Int8 -> 1
    | Int16 -> 2
    | Int -> 3
    | Int64 -> 4
    | Half -> 5
    | Float -> 6
    | Double -> 7
    | ComplexHalf -> 8
    | ComplexFloat -> 9
    | ComplexDouble -> 10

  let of_int_exn = function
    | 0 -> Uint8
    | 1 -> Int8
    | 2 -> Int16
    | 3 -> Int
    | 4 -> Int64
    | 5 -> Half
    | 6 -> Float
    | 7 -> Double
    | 8 -> ComplexHalf
    | 9 -> ComplexFloat
    | 10 -> ComplexDouble
    | d -> failwith (Printf.sprintf "unexpected kind %d" d)
end

module Tensor = struct
  open! C.Tensor
  type nonrec t = t

  let zeros ?(kind=Kind.Float) dims =
    let dim_array = CArray.of_list int dims |> CArray.start in
    let tensor = zeros dim_array (List.length dims) (Kind.to_int kind) in
    Gc.finalise free tensor;
    tensor

  let ones ?(kind=Kind.Float) dims =
    let dim_array = CArray.of_list int dims |> CArray.start in
    let tensor = ones dim_array (List.length dims) (Kind.to_int kind) in
    Gc.finalise free tensor;
    tensor

  let rand dims =
    let dim_array = CArray.of_list int dims |> CArray.start in
    let tensor = rand dim_array (List.length dims) in
    Gc.finalise free tensor;
    tensor

  let reshape t dims =
    let dim_array = CArray.of_list int dims |> CArray.start in
    let tensor = reshape t dim_array (List.length dims) in
    Gc.finalise free tensor;
    tensor

  let shape t =
    let num_dims = num_dims t in
    let carray = CArray.make int num_dims in
    shape t (CArray.start carray);
    CArray.to_list carray

  let kind t = scalar_type t |> Kind.of_int_exn

  let add x y =
    let tensor = add x y in
    Gc.finalise free tensor;
    tensor

  let sub x y =
    let tensor = sub x y in
    Gc.finalise free tensor;
    tensor

  let mul x y =
    let tensor = mul x y in
    Gc.finalise free tensor;
    tensor

  let div x y =
    let tensor = div x y in
    Gc.finalise free tensor;
    tensor

  let pow x y =
    let tensor = pow x y in
    Gc.finalise free tensor;
    tensor

  let matmul x y =
    let tensor = matmul x y in
    Gc.finalise free tensor;
    tensor

  let print = print
end

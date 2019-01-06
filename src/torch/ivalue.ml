open Base
module I = Torch_core.Wrapper.Ivalue

type t =
  | Tensor of Tensor.t
  | Int of int
  | Double of float
  | Tuple of t list

let rec to_wrapper = function
  | Tensor tensor -> I.tensor tensor
  | Int int -> I.int64 (Int64.of_int int)
  | Double double -> I.double double
  | Tuple tuple ->
    I.tuple (List.map ~f:to_wrapper tuple)

let rec of_wrapper ivalue =
  match I.tag ivalue with
  | Tensor -> Tensor (I.to_tensor ivalue)
  | Int -> Int (I.to_int64 ivalue |> Int64.to_int_exn)
  | Double -> Double (I.to_double ivalue)
  | Tuple -> Tuple (I.to_tuple ivalue |> List.map ~f:of_wrapper)

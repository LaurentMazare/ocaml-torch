open Base
module I = Torch_core.Wrapper.Ivalue

type raw = I.t

type t =
  | Tensor of Tensor.packed
  | Int of int
  | Double of float
  | Tuple of t list

let rec to_raw = function
  | Tensor (T tensor) -> I.tensor tensor
  | Int int -> I.int64 (Int64.of_int int)
  | Double double -> I.double double
  | Tuple tuple -> I.tuple (List.map ~f:to_raw tuple)

let rec of_raw ivalue =
  match I.tag ivalue with
  | Tensor -> Tensor (I.to_tensor ivalue)
  | Int -> Int (I.to_int64 ivalue |> Int64.to_int_exn)
  | Double -> Double (I.to_double ivalue)
  | Tuple -> Tuple (I.to_tuple ivalue |> List.map ~f:of_raw)

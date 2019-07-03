open Base

type raw = Torch_core.Wrapper.Ivalue.t

type t =
  | Tensor of Tensor.t
  | Int of int
  | Double of float
  | Tuple of t list

val to_raw : t -> raw
val of_raw : raw -> t

open Base

type t =
  | Tensor of Tensor.t
  | Int of int
  | Double of float
  | Tuple of t list

val to_wrapper : t -> Torch_core.Wrapper.Ivalue.t
val of_wrapper : Torch_core.Wrapper.Ivalue.t -> t

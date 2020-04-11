open Base

type raw = Torch_core.Wrapper.Ivalue.t

type t =
  | None
  | Bool of bool
  | Tensor of Tensor.t
  | Int of int
  | Double of float
  | Tuple of t list
  | String of string

val to_raw : t -> raw
val of_raw : raw -> t
val to_string : t -> string

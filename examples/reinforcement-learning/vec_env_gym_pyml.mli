open Torch

type step =
  { obs : Tensor.t
  ; reward : Tensor.t
  ; is_done : Tensor.t }

type t

val create : string -> num_processes:int -> t
val reset : t -> Tensor.t
val step : t -> actions:int list -> step
val action_space : t -> int

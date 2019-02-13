open Torch

type step =
  { obs : Tensor.t
  ; reward : float
  ; is_done : bool }

type t

val create : string -> action_repeat:int -> t
val reset : t -> Tensor.t
val step : t -> action:int -> step
val actions : t -> string list
val lives : t -> int

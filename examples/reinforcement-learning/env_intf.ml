open Torch

type step =
  { obs : Tensor.t
  ; reward : float
  ; is_done : bool
  }

module type S = sig
  type t

  val create : string -> t
  val reset : t -> Tensor.t
  val step : t -> action:int -> render:bool -> step
end

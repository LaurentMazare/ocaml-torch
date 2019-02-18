open Torch

type actor_critic =
  { actor : Tensor.t
  ; critic : Tensor.t }

type totals =
  { rewards : float
  ; episodes : float }

type rollout =
  { states : Tensor.t
  ; returns : Tensor.t
  ; actions : Tensor.t }

type t

val create : atari_game:string -> num_steps:int -> num_stack:int -> num_procs:int -> t
val action_space : t -> int
val run : t -> model:(Tensor.t -> actor_critic) -> rollout
val get_and_reset_totals : t -> totals

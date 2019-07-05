open Torch

type actor_critic =
  { actor : Tensor.t
  ; critic : Tensor.t
  }

type totals =
  { rewards : float
  ; episodes : float
  }

type rollout =
  { states : Tensor.t
  ; returns : Tensor.t
  ; actions : Tensor.t
  ; values : Tensor.t
  }

type t

(** [create] creates a rollout environment with the specified parameters. *)
val create : atari_game:string -> num_steps:int -> num_stack:int -> num_procs:int -> t

val action_space : t -> int

(** [run t ~model] performs a rollout for [t.num_steps] using the given actor-critic
    model.
    The resulting rollout combines the observed states, returns to the end of the
    episode, performed action, and critic values.
*)
val run : t -> model:(Tensor.t -> actor_critic) -> rollout

(** [get_and_reset_totals t] returns the sum of rewards and the number of finished
    episodes performed during the previous calls to [run t] since the last call
    to [get_and_reset_totals].
*)
val get_and_reset_totals : t -> totals

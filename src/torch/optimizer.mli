type t

val adam : Var_store.t -> learning_rate:float -> t

val sgd
  :  ?momentum:float
  -> ?dampening:float
  -> ?weight_decay:float
  -> ?nesterov:bool
  -> Var_store.t
  -> learning_rate:float
  -> t

val step : ?clip_grad_norm2:float -> t -> unit
val zero_grad : t -> unit
val backward_step : ?clip_grad_norm2:float -> t -> loss:Tensor.t -> unit
val set_learning_rate : t -> learning_rate:float -> unit

module Linear_interpolation : sig
  type t

  (** [create vs] creates a linear interpolation function using [vs] as
      knots. [vs] is a list of pairs [(x, y)], has to be sorted by [x]
      and cannot have two elements with the same [x].
  *)
  val create : (float * float) list -> t

  (** [eval t x] returns the linear interpolation value of [t] at [x].
      If [t] was built from [(x1, y1)] to [(xn, yn)].
      If [x] is below [x1], the returned value is [y1].
      If [x] is above [xn], the returned value is [yn].
      If [x] is between too knots, linear interpolation is used.
  *)
  val eval : t -> float -> float
end

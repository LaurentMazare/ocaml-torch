type t

val adam : Tensor.t list -> learning_rate:float -> t
val step : t -> unit
val zero_grad : t -> unit
val backward_step : t -> loss:Tensor.t -> unit

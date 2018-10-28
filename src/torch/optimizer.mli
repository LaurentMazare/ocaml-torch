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

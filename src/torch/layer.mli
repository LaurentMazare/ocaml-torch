open Base

type t
type t_with_training

val set_training : t_with_training -> is_training:bool -> t
val with_training : t -> t_with_training

type activation =
  | Relu
  | Softmax
  | Log_softmax
  | Tanh
  | Leaky_relu
  | Sigmoid

val linear
  :  Var_store.t
  -> ?activation:activation (* default: no activation *)
  -> ?use_bias:bool (* default: true *)
  -> ?w_init:Var_store.Init.t
  -> input_dim:int
  -> int
  -> t

val conv2d
  :  Var_store.t
  -> ksize:int * int
  -> stride:int * int
  -> ?activation:activation (* default: no activation *)
  -> ?use_bias:bool (* default: true *)
  -> ?w_init:Var_store.Init.t
  -> ?padding:int * int
  -> ?groups:int
  -> input_dim:int
  -> int
  -> t

val conv2d_
  :  Var_store.t
  -> ksize:int
  -> stride:int
  -> ?activation:activation (* default: no activation *)
  -> ?use_bias:bool (* default: true *)
  -> ?w_init:Var_store.Init.t
  -> ?padding:int
  -> ?groups:int
  -> input_dim:int
  -> int
  -> t

val conv_transpose2d
  :  Var_store.t
  -> ksize:int * int
  -> stride:int * int
  -> ?activation:activation (* default: no activation *)
  -> ?use_bias:bool (* default: true *)
  -> ?w_init:Var_store.Init.t
  -> ?padding:int * int
  -> ?output_padding:int * int
  -> ?groups:int
  -> input_dim:int
  -> int
  -> t

val conv_transpose2d_
  :  Var_store.t
  -> ksize:int
  -> stride:int
  -> ?activation:activation (* default: no activation *)
  -> ?use_bias:bool (* default: true *)
  -> ?w_init:Var_store.Init.t
  -> ?padding:int
  -> ?output_padding:int
  -> ?groups:int
  -> input_dim:int
  -> int
  -> t

val batch_norm2d
  :  Var_store.t
  -> ?w_init:Var_store.Init.t
  -> ?cudnn_enabled:bool
  -> ?eps:float
  -> ?momentum:float
  -> int
  -> t_with_training

val id : t
val id_ : t_with_training
val of_fn : (Tensor.t -> Tensor.t) -> t
val of_fn_ : (Tensor.t -> is_training:bool -> Tensor.t) -> t_with_training
val fold : t list -> t
val fold_ : t_with_training list -> t_with_training
val apply : t -> Tensor.t -> Tensor.t
val apply_ : t_with_training -> Tensor.t -> is_training:bool -> Tensor.t

module Lstm : sig
  type t
  type state = Tensor.t * Tensor.t

  val create : Var_store.t -> input_dim:int -> hidden_size:int -> t
  val step : t -> state -> Tensor.t -> state
  val seq : t -> Tensor.t -> Tensor.t * state
  val zero_state : t -> batch_size:int -> state
end

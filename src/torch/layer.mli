open Base

type t

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
  -> input_dim:int
  -> int
  -> t

val conv2d
  :  Var_store.t
  -> ksize:int * int
  -> stride:int * int
  -> ?activation:activation (* default: no activation *)
  -> ?use_bias:bool (* default: true *)
  -> ?padding:int * int
  -> input_dim:int
  -> int
  -> t

val conv2d_
  :  Var_store.t
  -> ksize:int
  -> stride:int
  -> ?activation:activation (* default: no activation *)
  -> ?use_bias:bool (* default: true *)
  -> ?padding:int
  -> input_dim:int
  -> int
  -> t

val conv_transpose2d
  :  Var_store.t
  -> ksize:int * int
  -> stride:int * int
  -> ?activation:activation (* default: no activation *)
  -> ?padding:int * int
  -> ?output_padding:int * int
  -> input_dim:int
  -> int
  -> t

val conv_transpose2d_
  :  Var_store.t
  -> ksize:int
  -> stride:int
  -> ?activation:activation (* default: no activation *)
  -> ?padding:int
  -> ?output_padding:int
  -> input_dim:int
  -> int
  -> t

val batch_norm2d
  :  Var_store.t
  -> ?eps:float
  -> ?momentum:float
  -> int
  -> (Tensor.t -> is_training:bool -> Tensor.t) Staged.t

val id : t
val fold : t list -> t

val apply
  :  t
  -> Tensor.t
  -> Tensor.t

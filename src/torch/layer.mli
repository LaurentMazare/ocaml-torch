module Var_store : sig
  type t
  val create : ?device:Torch_core.Device.t -> name:string -> unit -> t
  val vars : t -> Tensor.t list
  val name : t -> string
  val device : t -> Torch_core.Device.t
end

type t

type activation =
  | Relu
  | Softmax
  | Tanh
  | Leaky_relu
  | Sigmoid

val linear
  :  Var_store.t
  -> ?w_init:float
  -> ?activation:activation (* default: no activation *)
  -> ?use_bias:bool (* default: true *)
  -> input_dim:int
  -> int
  -> t

val conv2d
  :  Var_store.t
  -> ?w_init:float
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
  -> ?w_init:float
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

val apply
  :  t
  -> Tensor.t
  -> Tensor.t

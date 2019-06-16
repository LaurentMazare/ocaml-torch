open Base

(** A layer takes as input a tensor and returns a tensor through the
    [forward] function.
    Layers can hold variables, these are created and registered using
    a [Var_store.t] when creating the layer.
*)
type t

(** A layer of type [t_with_training] is similar to a layer of type [t]
    except that it is also given a boolean argument when applying it to
    a tensor that specifies whether the layer is currently used in
    training or in testing mode.
    This is typically the case for batch normalization or dropout.
*)
type t_with_training

(** [with_training t] returns a layer using the [is_training] argument
    from a standard layer. The [is_training] argument is discarded.
    This is useful when sequencing multiple layers via [fold].
*)
val with_training : t -> t_with_training

(** The different kind of activations supported by the various layers
    below.
*)
type activation =
  | Relu
  | Softmax
  | Log_softmax
  | Tanh
  | Leaky_relu
  | Sigmoid

(** [linear vs ~input_dim output_dim] returns a linear layer. When using
    [forward], the input tensor has to use a shape of [batch_size * input_dim].
    The returned tensor has a shape [batch_size * output_dim].
*)
val linear
  :  Var_store.t
  -> ?activation:activation (* default: no activation *)
  -> ?use_bias:bool (* default: true *)
  -> ?w_init:Var_store.Init.t
  -> input_dim:int
  -> int
  -> t

(** [conv2d vs ~ksize ~stride ~input_dim output_dim] returns a 2 dimension
    convolution layer.  [ksize] specifies the kernel size and [stride] the
    stride.
    When using [forward], the input tensor should have a shape
    [batch_size * input_dim * h * w] and the returned tensor will have a
    shape [batch_size * output_dim * h' * w'].
*)
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

(** [conv2d_] is similar to [conv2d] but uses the same kernel size,
    stride, and padding on both the height and width dimensions, so a
    single integer needs to be specified for these parameters.
*)
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

(** [conv_transpose2d] creates a 2D transposed convolution layer, this
    is sometimes also called 'deconvolution'.
*)
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

(** [conv_transpose2d_] is similar to [conv_transpose2d] but uses a single
    value for the height and width dimension for the kernel size, stride,
    padding and output padding.
*)
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

(** [batch_norm2d vs dim] creates a batch norm 2D layer. This layer
    applies Batch Normalization over a 4D input
    [batch_size * dim * h * w]. The returned tensor has the same shape.
*)
val batch_norm2d
  :  Var_store.t
  -> ?w_init:Var_store.Init.t
  -> ?cudnn_enabled:bool
  -> ?eps:float
  -> ?momentum:float
  -> int
  -> t_with_training

(** The identity layer. [forward id tensor] returns [tensor]. *)
val id : t

(** The identity layer with an [is_training] argument. *)
val id_ : t_with_training

(** [of_fn f] creates a layer based on a function from tensors to tensors.
*)
val of_fn : (Tensor.t -> Tensor.t) -> t

(** [of_fn_ f] creates a layer based on a function from tensors to tensors.
    [f] also has access to the [is_training] flag.
*)
val of_fn_ : (Tensor.t -> is_training:bool -> Tensor.t) -> t_with_training

(** [sequential ts] applies sequentially a list of layers [ts]. *)
val sequential : t list -> t

(** [sequential_ ts] applies sequentially a list of layers [ts]. *)
val sequential_ : t_with_training list -> t_with_training

(** [forward t tensor] applies layer [t] to [tensor]. *)
val forward : t -> Tensor.t -> Tensor.t

(** [forward_ t tensor ~is_training] applies layer [t] to [tensor] with
    the specified [is_training] flag. *)
val forward_ : t_with_training -> Tensor.t -> is_training:bool -> Tensor.t

(** A Long Short Term Memory (LSTM) recurrent neural network. *)
module Lstm : sig
  type t
  type state = Tensor.t * Tensor.t

  (** [create vs ~input_dim ~hidden_size] creates a new LSTM with the
      specified input dimension and hidden size.
  *)
  val create : Var_store.t -> input_dim:int -> hidden_size:int -> t

  (** [step t state input_] applies one step of the LSTM network on the
      given input using the specified state. The updated state is
      returned.
  *)
  val step : t -> state -> Tensor.t -> state

  (** [seq t inputs] applies multiple steps of the LSTM network starting
      from a zero state. The hidden states and the final state are
      returned.
      [inputs] should have shape [batch_size * timesteps * input_dim],
      the returned output tensor then has shape
      [batch_size * timesteps * hidden_size].
  *)
  val seq : t -> Tensor.t -> Tensor.t * state

  (** [zero_state t ~batch_size] returns an initial state to be used for
      a LSTM network.
  *)
  val zero_state : t -> batch_size:int -> state
end

(** The module interface implemented by Recurrent Neural Networks. *)
module type S = sig
  type t
  type state

  (** [create vs ~input_dim ~hidden_size] creates a new RNN with the
      specified input dimension and hidden size.
  *)
  val create : Var_store.t -> input_dim:int -> hidden_size:int -> t

  (** [step t state input_] applies one step of the RNN on the
      given input using the specified state. The updated state is
      returned.
  *)
  val step : t -> state -> Tensor.t -> state

  (** [seq t inputs] applies multiple steps of the RNN starting
      from a zero state. The hidden states and the final state are
      returned.
      [inputs] should have shape [batch_size * timesteps * input_dim],
      the returned output tensor then has shape
      [batch_size * timesteps * hidden_size].
  *)
  val seq : t -> Tensor.t -> Tensor.t * state

  (** [zero_state t ~batch_size] returns an initial state to be used for
      a RNN.
  *)
  val zero_state : t -> batch_size:int -> state
end

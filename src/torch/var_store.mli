type t

val create : ?frozen:bool -> ?device:Torch_core.Device.t -> name:string -> unit -> t

(* The trainable variables are guaranteed to be returned in
   reverse order of addition (the [Optimizer] module relies
   on this). *)
val trainable_vars : t -> Tensor.t list
val all_vars : t -> (string * Tensor.t) list
val name : t -> string
val device : t -> Torch_core.Device.t

module Init : sig
  type t =
    | Zeros
    | Ones
    | Const of float
    | Normal of { mean : float; stdev : float }
    | Uniform of float * float
    | Copy of Tensor.t
end

val new_var
  :  ?trainable:bool (* default: true *)
  -> t
  -> shape:int list
  -> init:Init.t
  -> name:N.t
  -> Tensor.t

val new_var_copy
  :  ?trainable:bool
  -> t
  -> src:Tensor.t
  -> name:N.t
  -> Tensor.t

(** [default_name t name_option str] builds a default name based on [str] when
    [name_option] is [None], otherwise [name_option] is used.
*)
val default_name : t -> N.t option -> string -> N.t

val freeze : t -> unit
val unfreeze : t -> unit

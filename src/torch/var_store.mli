open Base

module Tensor_id : sig
  include Hashable.Key
end

(** A VarStore is used to store all the variables used by a given model.
    The model creates variables by calling [Var_store.new_var] for which
    it has to provide a name.
    [Var_store.sub] creates a sub-directory in the var store which is
    useful to group some variables together.
*)
type t

(** [create ?frozen ?device ~name ()] creates a new variable store
    on the specified device (defaulting to cpu).
*)
val create : ?frozen:bool -> ?device:Device.t -> name:string -> unit -> t

(** [sub t subname] returns a var-store corresponding to path
    [subname] in [t].
*)
val sub : t -> string -> t

(** [subi] is similar to [sub] but uses an integer for the subname. *)
val subi : t -> int -> t

(** Same as [sub]. *)
val ( / ) : t -> string -> t

(** Same as [subi]. *)
val ( // ) : t -> int -> t

(** [num_trainable_vars t] returns the number trainable variables stored in [t]. *)
val num_trainable_vars : t -> int

(** [iter_trainable_vars t ~f] applies f to all trainable variables stored in [t]. *)
val iter_trainable_vars : t -> f:(Tensor_id.t -> Tensor.t -> unit) -> unit

(** [all_vars t] returns all the variables stored in [t]. *)
val all_vars : t -> (string * Tensor.t) list

(** [name t] returns the var-store name. *)
val name : t -> string

(** [device t] returns the device used to store variables hold by [t]. *)
val device : t -> Device.t

module Init : sig
  type t =
    | Zeros
    | Ones
    | Const of float
    | Normal of
        { mean : float
        ; stdev : float
        }
    | Uniform of float * float
    | Copy of Tensor.t
end

(** [new_var ?trainable t ~shape ~init ~name] creates a new variable in [t]    with shape [shape] and the specified initialization.
    The tensor associated with the variable is returned.
*)
val new_var
  :  ?trainable:bool (* default: true *)
  -> t
  -> shape:int list
  -> init:Init.t
  -> name:string
  -> Tensor.t

(** [new_var_copy ?trainable t ~src ~name] creates a new variable in [t]
    by copying tensor [src], so using the same shape, element kind and
    element values.
*)
val new_var_copy : ?trainable:bool -> t -> src:Tensor.t -> name:string -> Tensor.t

(** [freeze t] freezes all variables in the var-store, none of the
    variables are trainable anymore and their gradients are not tracked.
*)
val freeze : t -> unit

(** [unfreeze t] unfreezes the 'trainable' variables from [t]. *)
val unfreeze : t -> unit

val copy : src:t -> dst:t -> unit

(** A VarStore is used to store all the variables used by a given model.
    The model creates variables by calling [Var_store.new_var] for which
    it has to provide a name.
    [Var_store.sub] creates a sub-directory in the var store which is
    useful to group some variables together.
*)
type t

val create : ?frozen:bool -> ?device:Torch_core.Device.t -> name:string -> unit -> t
val sub : t -> string -> t
val ( / ) : t -> string -> t
val trainable_vars : t -> Tensor.t list
val all_vars : t -> (string * Tensor.t) list
val name : t -> string
val device : t -> Torch_core.Device.t

module Init : sig
  type t =
    | Zeros
    | Ones
    | Const of float
    | Normal of {mean : float; stdev : float}
    | Uniform of float * float
    | Copy of Tensor.t
end

val new_var :
     ?trainable:bool (* default: true *)
  -> t
  -> shape:int list
  -> init:Init.t
  -> name:string
  -> Tensor.t

val new_var_copy : ?trainable:bool -> t -> src:Tensor.t -> name:string -> Tensor.t
val freeze : t -> unit
val unfreeze : t -> unit
val copy : src:t -> dst:t -> unit

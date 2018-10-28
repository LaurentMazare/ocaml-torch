type t

val create : ?device:Torch_core.Device.t -> name:string -> unit -> t

(* The trainable variables are guaranteed to be returned in
   reverse order of addition (the [Optimizer] module relies
   on this). *)
val vars : t -> [`trainable | `all ] -> Tensor.t list
val name : t -> string
val device : t -> Torch_core.Device.t
val add_var : t -> var:Tensor.t -> kind:[ `trainable | `non_trainable ] -> unit
val add_vars : t -> vars:Tensor.t list -> kind:[ `trainable | `non_trainable ] -> unit

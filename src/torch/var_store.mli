type t

val create : ?device:Torch_core.Device.t -> name:string -> unit -> t
val vars : t -> Tensor.t list
val name : t -> string
val device : t -> Torch_core.Device.t
val add_var : t -> var:Tensor.t -> unit
val add_vars : t -> vars:Tensor.t list -> unit

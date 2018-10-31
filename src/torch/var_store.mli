type t

val create : ?device:Torch_core.Device.t -> name:string -> unit -> t

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
    | Normal_with_stdev of float
    | Uniform of float * float
end

val new_var
  :  ?trainable:bool (* default: true *)
  -> t
  -> shape:int list
  -> init:Init.t
  -> name:N.t
  -> Tensor.t

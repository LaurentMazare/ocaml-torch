val load : filename:string -> Tensor.t
val save : Tensor.t -> filename:string -> unit

val load_multi
  :  names:string list
  -> filename:string
  -> Tensor.t list

val save_multi
  :  named_tensors:(string * Tensor.t) list
  -> filename:string
  -> unit

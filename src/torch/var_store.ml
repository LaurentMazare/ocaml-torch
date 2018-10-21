open! Base

type t =
  { name : string
  ; mutable tensors : Tensor.t list
  ; device : Torch_core.Device.t
  }

let create ?(device = Torch_core.Device.Cpu) ~name () = { name; tensors = []; device }
let add_var t ~var = t.tensors <- var :: t.tensors
let add_vars t ~vars = t.tensors <- vars @ t.tensors
let vars t = t.tensors
let name t = t.name
let device t = t.device

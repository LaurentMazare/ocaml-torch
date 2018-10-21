open! Base

type t =
  { name : string
  ; mutable trainable_tensors : Tensor.t list
  ; mutable non_trainable_tensors : Tensor.t list
  ; device : Torch_core.Device.t
  }

let create ?(device = Torch_core.Device.Cpu) ~name () =
  { name
  ; trainable_tensors = []
  ; non_trainable_tensors = []
  ; device
  }

let add_var t ~var ~kind =
  match kind with
  | `trainable ->
      t.trainable_tensors <- var :: t.trainable_tensors
  | `non_trainable ->
      t.non_trainable_tensors <- var :: t.non_trainable_tensors

let add_vars t ~vars ~kind =
  match kind with
  | `trainable ->
      t.trainable_tensors <- vars @ t.trainable_tensors
  | `non_trainable ->
      t.non_trainable_tensors <- vars @ t.non_trainable_tensors

let vars t trainable_all =
  match trainable_all with
  | `trainable -> t.trainable_tensors
  | `all -> t.trainable_tensors @ t.non_trainable_tensors

let name t = t.name
let device t = t.device

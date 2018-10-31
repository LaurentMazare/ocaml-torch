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

let vars t trainable_all =
  match trainable_all with
  | `trainable -> t.trainable_tensors
  | `all -> t.trainable_tensors @ t.non_trainable_tensors

let name t = t.name
let device t = t.device

module Init = struct
  type t =
    | Zeros
    | Ones
    | Const of float
    | Normal_with_stdev of float
    | Uniform of float * float
end

let new_var ?(trainable=true) t ~shape ~init ~name =
  ignore name;
  let device = device t in
  let tensor =
    match (init : Init.t) with
    | Zeros -> Tensor.zeros shape ~requires_grad:trainable ~device
    | Ones -> Tensor.ones shape ~requires_grad:trainable ~device
    | Const scale -> Tensor.ones shape ~requires_grad:trainable ~device ~scale
    | Normal_with_stdev stdev -> Tensor.randn shape ~scale:stdev ~requires_grad:trainable ~device
    | Uniform (from, to_) ->
      Tensor.zeros shape ~device
      |> Tensor.uniform_ ~from ~to_
      |> Tensor.set_requires_grad ~r:trainable
  in
  if trainable
  then begin
    t.trainable_tensors <- tensor :: t.trainable_tensors
  end else begin
    t.non_trainable_tensors <- tensor :: t.non_trainable_tensors
  end;
  tensor

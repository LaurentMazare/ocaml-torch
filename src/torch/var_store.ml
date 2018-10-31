open! Base

type t =
  { name : string
  ; mutable trainable_tensors : Tensor.t list
  ; all_tensors_by_name : (string, Tensor.t) Hashtbl.t
  ; device : Torch_core.Device.t
  }

let create ?(device = Torch_core.Device.Cpu) ~name () =
  { name
  ; trainable_tensors = []
  ; all_tensors_by_name = Hashtbl.create (module String)
  ; device
  }

let trainable_vars t = t.trainable_tensors

let all_vars t = Hashtbl.to_alist t.all_tensors_by_name

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
  let name = N.to_string name in
  if Hashtbl.mem t.all_tensors_by_name name
  then begin
    Printf.sprintf "multiple variable with name: %s" name |> failwith
  end;
  Hashtbl.add_exn t.all_tensors_by_name ~key:name ~data:tensor;
  if trainable
  then begin
    t.trainable_tensors <- tensor :: t.trainable_tensors
  end;
  tensor

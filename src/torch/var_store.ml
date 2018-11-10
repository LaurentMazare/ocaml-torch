open! Base

type t =
  { name : string
  ; mutable trainable_tensors : Tensor.t list
  ; all_tensors_by_name : (string, Tensor.t) Hashtbl.t
  ; device : Torch_core.Device.t
  ; mutable name_counter : int
  ; mutable frozen : bool
  }

let create ?(frozen = false) ?(device = Torch_core.Device.Cpu) ~name () =
  { name
  ; trainable_tensors = []
  ; all_tensors_by_name = Hashtbl.create (module String)
  ; device
  ; name_counter = 1
  ; frozen
  }

let default_name t name_option base =
  match name_option with
  | Some t -> t
  | None ->
    t.name_counter <- t.name_counter + 1;
    N.of_list [ Printf.sprintf "%s__%d" base t.name_counter ]

let trainable_vars t = t.trainable_tensors

let freeze t =
  t.frozen <- true;
  List.iter (trainable_vars t) ~f:(fun tensor ->
    ignore (Tensor.set_requires_grad tensor ~r:false : Tensor.t))

let unfreeze t =
  t.frozen <- false;
  List.iter (trainable_vars t) ~f:(fun tensor ->
    ignore (Tensor.set_requires_grad tensor ~r:true : Tensor.t))

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
  let requires_grad = trainable && not t.frozen in
  let tensor =
    match (init : Init.t) with
    | Zeros -> Tensor.zeros shape ~requires_grad ~device
    | Ones -> Tensor.ones shape ~requires_grad ~device
    | Const scale -> Tensor.ones shape ~requires_grad ~device ~scale
    | Normal_with_stdev stdev -> Tensor.randn shape ~scale:stdev ~requires_grad ~device
    | Uniform (from, to_) ->
      Tensor.zeros shape ~device
      |> Tensor.uniform_ ~from ~to_
      |> Tensor.set_requires_grad ~r:requires_grad
  in
  let name = N.to_string name in
  if Hashtbl.mem t.all_tensors_by_name name
  then Printf.failwithf "multiple variable with name: %s" name ();
  Hashtbl.add_exn t.all_tensors_by_name ~key:name ~data:tensor;
  if trainable
  then begin
    t.trainable_tensors <- tensor :: t.trainable_tensors
  end;
  tensor

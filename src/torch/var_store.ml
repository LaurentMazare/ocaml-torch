open Base

(* Maybe we should also store the full path in the var stores ? *)
type t =
  { name : string
  ; mutable trainable_tensors : Tensor.t list
  ; all_tensors_by_name : (string, Tensor.t) Hashtbl.t
  ; subs : (string, t) Hashtbl.t
  ; device : Device.t
  ; mutable frozen : bool
  }

let create ?(frozen = false) ?(device = Device.Cpu) ~name () =
  { name
  ; trainable_tensors = []
  ; subs = Hashtbl.create (module String)
  ; all_tensors_by_name = Hashtbl.create (module String)
  ; device
  ; frozen
  }

let first_free_name name table =
  if Hashtbl.mem table name
  then (
    let rec loop idx =
      let name = Printf.sprintf "%s_%d" name idx in
      if Hashtbl.mem table name then loop (idx + 1) else name
    in
    loop 1)
  else name

let sub t sub_name =
  if String.contains sub_name '.'
  then Printf.failwithf "sub names cannot contain ., %s" sub_name ();
  Hashtbl.find_or_add t.subs sub_name ~default:(fun () ->
      { name = t.name
      ; trainable_tensors = []
      ; subs = Hashtbl.create (module String)
      ; all_tensors_by_name = Hashtbl.create (module String)
      ; device = t.device
      ; frozen = t.frozen
      })

let ( / ) = sub

let rec freeze t =
  t.frozen <- true;
  List.iter t.trainable_tensors ~f:(fun tensor ->
      ignore (Tensor.set_requires_grad tensor ~r:false : Tensor.t));
  Hashtbl.iter t.subs ~f:freeze

let rec unfreeze t =
  t.frozen <- false;
  List.iter t.trainable_tensors ~f:(fun tensor ->
      ignore (Tensor.set_requires_grad tensor ~r:true : Tensor.t));
  Hashtbl.iter t.subs ~f:unfreeze

let rec trainable_vars t =
  let sub_vars = Hashtbl.data t.subs |> List.concat_map ~f:trainable_vars in
  t.trainable_tensors @ sub_vars

let all_vars t =
  let rec walk t ~path =
    let sub_vars =
      Hashtbl.to_alist t.subs
      |> List.concat_map ~f:(fun (key, t) -> walk t ~path:(key :: path))
    in
    let vars =
      Hashtbl.to_alist t.all_tensors_by_name
      |> List.map ~f:(fun (key, tensor) ->
             List.rev (key :: path) |> String.concat ~sep:".", tensor)
    in
    vars @ sub_vars
  in
  walk t ~path:[]

let copy ~src ~dst =
  Tensor.no_grad (fun () ->
      let rec walk ~src ~dst path =
        Hashtbl.iteri dst.all_tensors_by_name ~f:(fun ~key ~data ->
            match Hashtbl.find src.all_tensors_by_name key with
            | Some src -> Tensor.copy_ data ~src
            | None ->
              Printf.failwithf
                "cannot find var %s from var-store %s in %s"
                (List.rev (key :: path) |> String.concat ~sep:".")
                dst.name
                src.name
                ());
        Hashtbl.iteri dst.subs ~f:(fun ~key ~data:dst ->
            match Hashtbl.find src.subs key with
            | Some src -> walk ~src ~dst (key :: path)
            | None ->
              Printf.failwithf
                "cannot find sub %s from var-store %s in %s"
                (List.rev (key :: path) |> String.concat ~sep:".")
                dst.name
                src.name
                ())
      in
      walk ~src ~dst [])

let name t = t.name
let device t = t.device

module Init = struct
  type t =
    | Zeros
    | Ones
    | Const of float
    | Normal of { mean : float; stdev : float }
    | Uniform of float * float
    | Copy of Tensor.t
end

let new_var ?(trainable = true) t ~shape ~init ~name =
  let device = device t in
  let requires_grad = trainable && not t.frozen in
  let tensor =
    match (init : Init.t) with
    | Zeros -> Tensor.zeros shape ~requires_grad ~device
    | Ones -> Tensor.ones shape ~requires_grad ~device
    | Const scale -> Tensor.ones shape ~requires_grad ~device ~scale
    | Normal { mean = 0.; stdev } ->
      Tensor.randn shape ~scale:stdev ~requires_grad ~device
    | Normal { mean; stdev } ->
      Tensor.( + )
        (Tensor.randn shape ~scale:stdev ~requires_grad ~device)
        (Tensor.f mean)
    | Uniform (from, to_) ->
      Tensor.zeros shape ~device
      |> Tensor.uniform_ ~from ~to_
      |> Tensor.set_requires_grad ~r:requires_grad
    | Copy src ->
      Tensor.copy src
      |> Tensor.to_device ~device
      |> Tensor.set_requires_grad ~r:requires_grad
  in
  if String.contains name '.'
  then Printf.failwithf "tensor names cannot contain ., %s" name ();
  let name = first_free_name name t.all_tensors_by_name in
  Hashtbl.add_exn t.all_tensors_by_name ~key:name ~data:tensor;
  if trainable then t.trainable_tensors <- tensor :: t.trainable_tensors;
  tensor

let new_var_copy ?trainable t ~src ~name =
  new_var ?trainable t ~shape:(Tensor.shape src) ~init:(Copy src) ~name

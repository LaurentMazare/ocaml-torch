open Base
include Torch_core.Wrapper.Serialize

let load ~filename = load ~filename |> Tensor.of_ptr

let save t ~filename = save (Tensor.to_ptr t) ~filename

let load_multi ~names ~filename = load_multi ~names ~filename |> List.map ~f:Tensor.of_ptr

let save_multi ~named_tensors ~filename =
  let named_tensors =
    List.map named_tensors ~f:(fun (name, t) -> name, Tensor.to_ptr t)
  in
  save_multi ~named_tensors ~filename

let load_multi_ ~named_tensors ~filename =
  let named_tensors =
    List.map named_tensors ~f:(fun (name, t) -> name, Tensor.to_ptr t)
  in
  load_multi_ ~named_tensors ~filename

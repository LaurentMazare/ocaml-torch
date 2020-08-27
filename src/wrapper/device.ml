type t =
  | Cpu
  | Cuda of int

(* Hardcoded, should match torch_api.cpp *)
let to_int = function
  | Cpu -> -1
  | Cuda i ->
    if i < 0 then Printf.sprintf "negative index for cuda device" |> failwith;
    i
;;

let of_int i = if i < 0 then Cpu else Cuda i

module C = Torch_bindings.C (Torch_generated)

let get_num_threads = C.get_num_threads
let set_num_threads = C.set_num_threads

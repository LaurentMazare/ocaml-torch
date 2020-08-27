type t = Torch_core.Device.t =
  | Cpu
  | Cuda of int

let cuda_if_available () = if Cuda.is_available () then Cuda 0 else Cpu

let is_cuda = function
  | Cpu -> false
  | Cuda _ -> true
;;

let get_num_threads = Torch_core.Device.get_num_threads
let set_num_threads = Torch_core.Device.set_num_threads

type t =
  | Cpu
  (* The int is the gpu device id, it must be non-negative and usually
     starts from 0. *)
  | Cuda of int

val to_int : t -> int
val of_int : int -> t
val get_num_threads : unit -> int
val set_num_threads : int -> unit

(* TODO: also save optimizer states. *)
(* TODO: get the named tensors from (named) variables stores. *)

(** [loop ~start_index ~end_index ~named_tensors ~checkpoint_base f] starts a
    loop ranging from [start_index] to [end_index]. On each iteration [f] is
    called on the current index.

    This loop is checkpointed: regularly the state of all the elements of
    [named_tensors] are saved on disk to file checkpoint_baseXXX.
    If such files already exist when starting the loop, the last one of this
    file is used to restore the variables content.
*)
val loop
  :  start_index:int
  -> end_index:int
  -> named_tensors:(string * Tensor.t) list
  -> checkpoint_base:string
  -> ?checkpoint_every:[ `iters of int | `seconds of float ] (* default : `second 600 *)
  -> (index:int -> unit)
  -> unit

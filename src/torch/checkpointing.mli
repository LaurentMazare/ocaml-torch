(* TODO: also save optimizer states. *)

(** [loop ~start_index ~end_index ~var_stores ~checkpoint_base f] starts a
    loop ranging from [start_index] to [end_index]. On each iteration [f] is
    called on the current index.

    This loop is checkpointed: regularly the state of all the elements of
    [var_stores] are saved on disk to file checkpoint_baseXXX.
    If such files already exist when starting the loop, the last one of this
    file is used to restore the variables content.
*)
val loop
  :  start_index:int
  -> end_index:int
  -> var_stores:Var_store.t list
  -> checkpoint_base:string
  -> ?only_keep:int
  -> ?checkpoint_every:[ `iters of int | `seconds of float ] (* default : `second 600 *)
  -> (index:int -> unit)
  -> unit

open! Base

type t

(** [create ~filename] reads text file [filename] and stores its content in
    [t].
    The tensors returned in this module use compact labels: e.g. if [filename]
    use n different characters, the values used in the tensors will range from
    0 to n-1.
*)
val create : filename:string -> t

(** [iter t ~f ~seq_len ~batch_size] iterates [f] over the whole
    dataset. [f] is given two different tensors [xs] and [ys] which
    shapes are both [seq_len; batch_size; labels]. [ys] is shifted by one
    compared to [xs].
    The dataset is shuffled on each call to [iter].
*)
val iter
  :  ?device:Device.t
  -> t
  -> f:(int -> xs:[ `i64 ] Tensor.t -> ys:[ `i64 ] Tensor.t -> unit)
  -> seq_len:int
  -> batch_size:int
  -> unit

(** [char t ~label] returns the character from the original file that has been
    mapped to [label].
*)
val char : t -> label:int -> char

val total_length : t -> int

(** [labels t] returns the number of different labels, i.e. the number of
    distinct chars in the original file.
*)
val labels : t -> int

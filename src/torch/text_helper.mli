open! Base

type _ t

(** [create ~filename] reads text file [filename] and stores its content in
    [t].
    The tensors returned in this module use compact labels: e.g. if [filename]
    use n different characters, the values used in the tensors will range from
    0 to n-1.
*)
val create : filename:string -> _ t

(** [iter t ~f ~seq_len ~batch_size] iterates [f] over the whole
    dataset. [f] is given two different tensors [xs] and [ys] which
    shapes are both [seq_len; batch_size; labels]. [ys] is shifted by one
    compared to [xs].
    The dataset is shuffled on each call to [iter].
*)
val iter
  :  ?device:Device.t
  -> 'a t
  -> f:(int -> xs:'a Tensor.t -> ys:'a Tensor.t -> unit)
  -> seq_len:int
  -> batch_size:int
  -> unit

(** [char t ~label] returns the character from the original file that has been
    mapped to [label].
*)
val char : _ t -> label:int -> char

val total_length : _ t -> int

(** [labels t] returns the number of different labels, i.e. the number of
    distinct chars in the original file.
*)
val labels : _ t -> int

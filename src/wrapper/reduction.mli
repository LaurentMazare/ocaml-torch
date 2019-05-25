type t =
  | None (** Do not perform any reduction. *)
  | Elementwise_mean
      (** Reduces the tensor to a scalar by taking the mean of the elements. *)
  | Sum (** Reduces the tensor to a scalar by taking the sum of the elements. *)

val to_int : t -> int

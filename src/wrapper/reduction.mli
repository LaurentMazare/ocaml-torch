type t =
  | None
  | Elementwise_mean
  | Sum

val to_int : t -> int

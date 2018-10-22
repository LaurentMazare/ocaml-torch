type t =
  | None
  | Elementwise_mean
  | Sum

let to_int = function
  | None -> 0
  | Elementwise_mean -> 1
  | Sum -> 2

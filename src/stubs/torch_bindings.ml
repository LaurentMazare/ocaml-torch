open Ctypes

module C(F: Cstubs.FOREIGN) = struct
  open F
  module Tensor = struct
    type t = unit ptr
    let t : t typ = ptr void

    let zeros =
      foreign "at_zeros"
        (int            (* data type *)
        @-> ptr int64_t (* dims *)
        @-> int         (* num dims *)
        @-> returning t)

    let ones =
      foreign "at_ones"
        (int            (* data type *)
        @-> ptr int64_t (* dims *)
        @-> int         (* num dims *)
        @-> returning t)

    let add =
      foreign "at_add"
        (t
        @-> t
        @-> returning t)
  end
end

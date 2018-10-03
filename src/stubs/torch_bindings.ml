open Ctypes

module C(F: Cstubs.FOREIGN) = struct
  open F
  module Tensor = struct
    type t = unit ptr
    let t : t typ = ptr void

    let zeros =
      foreign "at_zeros"
        (int
        @-> returning t)

    let ones =
      foreign "at_ones"
        (int
        @-> returning t)

    let add =
      foreign "at_add"
        (t
        @-> t
        @-> returning t)

    let free =
      foreign "at_free"
        (t
        @-> returning void)
  end
end

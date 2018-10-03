open Ctypes

module C(F: Cstubs.FOREIGN) = struct
  open F
  module Tensor = struct
    type t = unit ptr
    let t : t typ = ptr void

    let zeros =
      foreign "at_zeros"
        (   ptr int  (* dims *)
        @-> int      (* num dims *)
        @-> returning t)

    let ones =
      foreign "at_ones"
        (   ptr int  (* dims *)
        @-> int      (* num dims *)
        @-> returning t)

    let rand =
      foreign "at_rand"
        (   ptr int  (* dims *)
        @-> int      (* num dims *)
        @-> returning t)

    let reshape =
      foreign "at_reshape"
        (   t
        @-> ptr int  (* dims *)
        @-> int      (* num dims *)
        @-> returning t)

    let add =
      foreign "at_add" (t @-> t @-> returning t)

    let print =
      foreign "at_print"
        (t
        @-> returning void)

    let free =
      foreign "at_free"
        (t
        @-> returning void)
  end
end

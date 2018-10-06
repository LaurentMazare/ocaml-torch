open Ctypes

module C(F: Cstubs.FOREIGN) = struct
  open F
  module Tensor = struct
    type t = unit ptr
    let t : t typ = ptr void

    let float_vec =
      foreign "at_float_vec"
        (   ptr double (* values *)
        @-> int        (* num values *)
        @-> int        (* kind *)
        @-> returning t)

    let int_vec =
      foreign "at_int_vec"
        (   ptr int64_t (* values *)
        @-> int         (* num values *)
        @-> int         (* kind *)
        @-> returning t)

    let zeros =
      foreign "at_zeros"
        (   ptr int  (* dims *)
        @-> int      (* num dims *)
        @-> int      (* kind *)
        @-> returning t)

    let ones =
      foreign "at_ones"
        (   ptr int  (* dims *)
        @-> int      (* num dims *)
        @-> int      (* kind *)
        @-> returning t)

    let num_dims = foreign "at_dim" (t @-> returning int)

    let shape =
      foreign "at_shape"
        (   t
        @-> ptr int  (* dims *)
        @-> returning void)

    let scalar_type = foreign "at_scalar_type" (t @-> returning int)

    let sum = foreign "at_sum" (t @-> returning t)
    let mean = foreign "at_mean" (t @-> returning t)
    let neg = foreign "at_neg" (t @-> returning t)

    let eq = foreign "at_eq" (t @-> t @-> returning t)

    let sub_assign = foreign "at_sub_assign" (t @-> t @-> returning void)

    let backward = foreign "at_backward" (t @-> returning void)
    let grad = foreign "at_grad" (t @-> returning t)
    let set_requires_grad =
      foreign "at_set_requires_grad" (t @-> int @-> returning t)
    let requires_grad =
      foreign "at_requires_grad" (t @-> returning int)

    let get = foreign "at_get" (t @-> int @-> returning t)
    let select = foreign "at_select" (t @-> int @-> int @-> returning t)
    let double_value = foreign "at_double_value" (t @-> returning float)
    let int64_value = foreign "at_int64_value" (t @-> returning int64_t)
    let fill_double = foreign "at_fill_double" (t @-> float @-> returning void)
    let fill_int64 = foreign "at_fill_int64" (t @-> int64_t @-> returning void)

    let set_double2 =
      foreign "at_set_double2" (t @-> int @-> int @-> float @-> returning void)

    let print = foreign "at_print" (t @-> returning void)
    let save = foreign "at_save" (t @-> string @-> returning void)
    let load = foreign "at_load" (string @-> returning t)

    let free = foreign "at_free" (t @-> returning void)
  end
  module TensorG = Torch_bindings_generated.C(F)
end

include Base
include Torch_core.Wrapper.Tensor

(* TODO: implement these in a single C call rather than chaining [get]. *)
let set_float1 t i value = fill_float (get t i) value
let set_int2 t i j value = fill_int (get (get t i) j) value
let set_int1 t i value = fill_int (get t i) value

let get_float2 t i j = float_value (get (get t i) j)
let get_float1 t i = float_value (get t i)
let get_int2 t i j = int_value (get (get t i) j)
let get_int1 t i = int_value (get t i)

let set_requires_grad t ~b = set_requires_grad t b

let no_grad t ~f =
  if requires_grad t
  then
    let t = set_requires_grad t ~b:false in
    let result = f t in
    ignore (set_requires_grad t ~b:true : t);
    result
  else f t

let zero_grad t =
  let grad = grad t in
  ignore (detach_ grad : t);
  ignore (zero_ grad : t)

let gen ~f ?(requires_grad = false) ?(kind = Torch_core.Kind.Float) ?scale dims =
  let t = f dims kind in
  let t =
    Option.value_map scale
      ~f:(fun scale -> mul t (float_vec [ scale ]))
      ~default:t
  in
  if requires_grad
  then set_requires_grad t ~b:true
  else t

let zeros = gen ~f:zeros
let ones = gen ~f:ones
let rand = gen ~f:rand
let randn = gen ~f:randn

let f v = float_vec [ v ] |> reshape ~dims:[]
let mm = matmul

let (+) = add
let (-) = sub
let ( * ) = mul
let (/) = div

let (~-) = neg
let (-=) t other = ignore (sub_ t other : t)
let (+=) t other = ignore (add_ t other : t)
let (/=) t other = ignore (div_ t other : t)
let ( *=) t other = ignore (mul_ t other : t)
let (=) = eq

let to_type t ~type_ = totype t type_
let to_device t ~device = to1 t device

let narrow t ~dim ~start ~len = narrow t dim start len

let pair_to_list (p1, p2) = [ p1; p2 ]
let conv2d ?(padding=0, 0) ?(dilation=1, 1) ?(groups=1) input weight bias ~stride =
  conv2d
    input
    weight
    bias
    (pair_to_list stride)
    (pair_to_list padding)
    (pair_to_list dilation)
    groups

let conv_transpose2d
    ?(output_padding=0, 0)
    ?(padding=0, 0)
    ?(dilation=1, 1)
    ?(groups=1)
    input
    weight
    bias
    ~stride
  =
  conv_transpose2d
    input
    weight
    bias
    (pair_to_list stride)
    (pair_to_list padding)
    (pair_to_list output_padding)
    groups
    (pair_to_list dilation)

let max_pool2d ?(padding=0, 0) ?(dilation=1, 1) ?(ceil_mode=false) ?stride self ~ksize =
  max_pool2d
    self
    (pair_to_list ksize)
    (Option.value stride ~default:ksize |> pair_to_list)
    (pair_to_list padding)
    (pair_to_list dilation)
    ceil_mode

let dropout t ~keep_probability ~is_training = dropout t keep_probability is_training

let const_batch_norm ?(momentum=0.1) ?(eps=1e-5) input =
  batch_norm input None None None None true momentum eps false

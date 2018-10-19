open! Base

module Var_store = struct
  type t =
    { name : string
    ; mutable tensors : Tensor.t list
    ; device : Torch_core.Device.t
    }

  let create ?(device = Torch_core.Device.Cpu) ~name () = { name; tensors = []; device }
  let add_vars t ~vars = t.tensors <- vars @ t.tensors
  let vars t = t.tensors
  let name t = t.name
  let device t = t.device
end

type t =
  { apply : Tensor.t -> Tensor.t
  }

type activation =
  | Relu
  | Softmax
  | Tanh
  | Leaky_relu
  | Sigmoid

let apply ?activation ys =
  match activation with
  | Some Relu -> Tensor.relu ys
  | Some Softmax -> Tensor.softmax ys
  | Some Tanh -> Tensor.tanh ys
  | Some Sigmoid -> Tensor.sigmoid ys
  | Some Leaky_relu -> Tensor.leaky_relu ys
  | None -> ys

let linear vs ?activation ?(use_bias=true) ~input_dim output_dim =
  let w =
    Tensor.randn [ input_dim; output_dim ]
      ~scale:0.1 ~requires_grad:true ~device:(Var_store.device vs)
  in
  let apply =
    if use_bias
    then begin
      let b =
        Tensor.zeros [ output_dim ] ~requires_grad:true ~device:(Var_store.device vs)
      in
      Var_store.add_vars vs ~vars:[w; b];
      fun xs -> Tensor.(mm xs w + b) |> apply ?activation
    end else begin
      Var_store.add_vars vs ~vars:[w];
      fun xs -> Tensor.(mm xs w) |> apply ?activation
    end
  in
  { apply }

let conv2d vs ?(w_init=0.1) ~ksize:(k1, k2) ~stride ?activation ?(padding=0, 0) ~input_dim output_dim =
  let w =
    Tensor.randn [ output_dim; input_dim; k1; k2 ] ~scale:w_init ~requires_grad:true
      ~device:(Var_store.device vs)
  in
  let b = Tensor.zeros [ output_dim ] ~requires_grad:true ~device:(Var_store.device vs) in
  Var_store.add_vars vs ~vars:[w; b];
  let apply xs =
    Tensor.conv2d xs w b ~padding ~stride |> apply ?activation
  in
  { apply }

let conv2d_ vs ?w_init ~ksize ~stride ?activation ?(padding = 0) ~input_dim output_dim =
  conv2d vs
    ?w_init
    ~ksize:(ksize, ksize)
    ~stride:(stride, stride)
    ?activation
    ~padding:(padding, padding)
    ~input_dim
    output_dim

let conv_transpose2d vs ~ksize:(k1, k2) ~stride ?activation ?(padding=0, 0) ?(output_padding=0, 0) ~input_dim output_dim =
  let w =
    Tensor.randn [ input_dim; output_dim; k1; k2 ]
      ~scale:0.1 ~requires_grad:true ~device:(Var_store.device vs)
  in
  let b = Tensor.zeros [ output_dim ] ~requires_grad:true ~device:(Var_store.device vs) in
  Var_store.add_vars vs ~vars:[w; b];
  let apply xs =
    Tensor.conv_transpose2d xs w b ~output_padding ~padding ~stride |> apply ?activation
  in
  { apply }

let conv_transpose2d_ vs ~ksize ~stride ?activation ?(padding = 0) ?(output_padding = 0) ~input_dim output_dim =
  conv_transpose2d vs
    ~ksize:(ksize, ksize)
    ~stride:(stride, stride)
    ?activation
    ~padding:(padding, padding)
    ~output_padding:(output_padding, output_padding)
    ~input_dim
    output_dim

let apply t xs = t.apply xs

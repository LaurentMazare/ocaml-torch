open! Base

type t =
  { apply : Tensor.t -> Tensor.t
  }

type activation =
  | Relu
  | Softmax
  | Log_softmax
  | Tanh
  | Leaky_relu
  | Sigmoid

let kaiming_uniform vs ~shape ~a =
  let fan_in =
    match shape with
    | [] | [_] -> failwith "unexpected tensor shape"
    (* Weight matrix is transposed for linear layers. *)
    | [ fan_in; _fan_out ] -> fan_in
    | _fan_out :: fan_in :: others ->
      let others = List.fold others ~init:1 ~f:( * ) in
      fan_in * others
  in
  let std = Float.sqrt (2. /. ((1. +. a *. a) *. Float.of_int fan_in)) in
  let bound = Float.sqrt 3. *. std in
  let tensor = Tensor.zeros shape ~device:(Var_store.device vs) in
  let tensor = Tensor.uniform_ tensor ~from:(-. bound) ~to_:bound in
  let tensor = Tensor.set_requires_grad tensor ~r:true in
  Var_store.add_var vs ~var:tensor ~kind:`trainable;
  tensor

let apply ?activation ys =
  match activation with
  | Some Relu -> Tensor.relu ys
  | Some Softmax -> Tensor.softmax ys
  | Some Log_softmax -> Tensor.log_softmax ys ~dim:(-1)
  | Some Tanh -> Tensor.tanh ys
  | Some Sigmoid -> Tensor.sigmoid ys
  | Some Leaky_relu -> Tensor.leaky_relu ys
  | None -> ys

let linear vs ?activation ?(use_bias=true) ~input_dim output_dim =
  let w = kaiming_uniform vs ~shape:[ input_dim; output_dim ] ~a:(Float.sqrt 5.) in
  let apply =
    if use_bias
    then begin
      let b = Tensor.zeros [ output_dim ] ~device:(Var_store.device vs) in
      let bound = 1.0 /. Float.sqrt (Float.of_int input_dim) in
      let b = Tensor.uniform_ b ~from:(-. bound) ~to_:bound in
      let b = Tensor.set_requires_grad b ~r:true in
      Var_store.add_var vs ~var:b ~kind:`trainable;
      fun xs -> Tensor.(mm xs w + b) |> apply ?activation
    end else fun xs -> Tensor.(mm xs w) |> apply ?activation
  in
  { apply }

let conv2d vs ~ksize:(k1, k2) ~stride ?activation ?(use_bias=true) ?(padding=0, 0) ~input_dim output_dim =
  let w = kaiming_uniform vs ~shape:[ output_dim; input_dim; k1; k2 ] ~a:(Float.sqrt 5.) in
  let apply =
    if use_bias
    then begin
      let b = Tensor.zeros [ output_dim ] ~requires_grad:true ~device:(Var_store.device vs) in
      Var_store.add_var vs ~var:b ~kind:`trainable;
      fun xs -> Tensor.conv2d xs w b ~padding ~stride |> apply ?activation
    end else
      let b = Tensor.zeros [ output_dim ] ~device:(Var_store.device vs) in
      fun xs -> Tensor.conv2d xs w b ~padding ~stride |> apply ?activation
  in
  { apply }

let conv2d_ vs ~ksize ~stride ?activation ?use_bias ?(padding = 0) ~input_dim output_dim =
  conv2d vs
    ~ksize:(ksize, ksize)
    ~stride:(stride, stride)
    ?use_bias
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
  Var_store.add_vars vs ~vars:[w; b] ~kind:`trainable;
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

let batch_norm2d vs ?(eps=1e-5) ?(momentum=0.1) output_dim =
  let device = Var_store.device vs in
  let w = Tensor.zeros [ output_dim ] ~device in
  let w = Tensor.uniform_ w ~from:0. ~to_:1. in
  let w = Tensor.set_requires_grad w ~r:true in
  let b = Tensor.zeros [ output_dim ] ~requires_grad:true ~device in
  Var_store.add_vars vs ~vars:[w; b] ~kind:`trainable;
  let running_mean = Tensor.zeros [ output_dim ] ~device in
  let running_var = Tensor.ones [ output_dim ] ~device in
  Var_store.add_vars vs ~vars:[running_mean; running_var] ~kind:`non_trainable;
  Staged.stage (fun xs ~is_training ->
    Tensor.batch_norm xs
      ~weight:(Some w)
      ~bias:(Some b)
      ~running_mean:(Some running_mean)
      ~running_var:(Some running_var)
      ~training:is_training
      ~momentum
      ~eps
      ~cudnn_enabled:false)

let apply t xs = t.apply xs

let id = { apply = Fn.id }
let fold t_list =
  let apply xs =
    List.fold t_list ~init:xs ~f:(fun acc t -> t.apply acc)
  in
  { apply }

module Lstm = struct
  type t =
    { w_ih : Tensor.t
    ; w_hh : Tensor.t
    ; b_ih : Tensor.t
    ; b_hh : Tensor.t
    ; hidden_size : int
    ; device : Torch_core.Device.t
    }

  type state = Tensor.t * Tensor.t

  let create vs ~input_dim ~hidden_size =
    let gate_size = 4 * hidden_size in
    let w_ih = kaiming_uniform vs ~shape:[ gate_size; input_dim ] ~a:(Float.sqrt 5.) in
    let w_hh = kaiming_uniform vs ~shape:[ gate_size; hidden_size ] ~a:(Float.sqrt 5.) in
    let b_ih = Tensor.zeros [ gate_size ] ~requires_grad:true ~device:(Var_store.device vs) in
    let b_hh = Tensor.zeros [ gate_size ] ~requires_grad:true ~device:(Var_store.device vs) in
    Var_store.add_vars vs ~vars:[ b_ih; b_hh ] ~kind:`trainable;
    { w_ih; w_hh; b_ih; b_hh; hidden_size; device = Var_store.device vs }

  let zero_state t ~batch_size =
    let zeros = Tensor.zeros [ batch_size; t.hidden_size ] ~device:t.device in
    zeros, zeros

  let step t (h, c) input_ =
    Tensor.lstm_cell input_
      ~hx:[ h; c ] ~w_ih:t.w_ih ~w_hh:t.w_hh ~b_ih:(Some t.b_ih) ~b_hh:(Some t.b_hh)

  let seq t input_ =
    let batch_size = Tensor.shape input_ |> List.hd_exn in
    let h = Tensor.zeros [ 1; batch_size; t.hidden_size ] ~device:t.device in
    let c = Tensor.zeros [ 1; batch_size; t.hidden_size ] ~device:t.device in
    let output, h, c =
      Tensor.lstm1 input_
        ~hx:[ h; c ]
        ~params:[ t.w_ih; t.w_hh; t.b_ih; t.b_hh ]
        ~has_biases:true
        ~num_layers:1
        ~dropout:0.
        ~train:false
        ~bidirectional:false
        ~batch_first:true
    in
    output, (h, c)
end

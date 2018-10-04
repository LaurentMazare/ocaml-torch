(* The readers implemented here are very inefficient as they read bytes one at a time. *)
open Base
module In_channel = Stdio.In_channel

let image_dim = 28 * 28
let label_count = 10

let read_int32_be in_channel =
  let b1 = Option.value_exn (In_channel.input_byte in_channel) in
  let b2 = Option.value_exn (In_channel.input_byte in_channel) in
  let b3 = Option.value_exn (In_channel.input_byte in_channel) in
  let b4 = Option.value_exn (In_channel.input_byte in_channel) in
  b4 + 256 * (b3 + 256 * (b2 + 256 * b1))

let read_images filename =
  let in_channel = In_channel.create filename in
  let magic_number = read_int32_be in_channel in
  if magic_number <> 2051
  then Printf.failwithf "Incorrect magic number in %s: %d" filename magic_number ();
  let samples = read_int32_be in_channel in
  let rows = read_int32_be in_channel in
  let columns = read_int32_be in_channel in
  let data = Tensor.zeros [samples; rows * columns] in
  for sample = 0 to samples - 1 do
    for idx = 0 to rows * columns - 1 do
      let v = Option.value_exn (In_channel.input_byte in_channel) in
      Tensor.set_float2 data sample idx Float.(of_int v / 255.);
    done;
  done;
  In_channel.close in_channel;
  data

let read_labels filename =
  let in_channel = In_channel.create filename in
  let magic_number = read_int32_be in_channel in
  if magic_number <> 2049
  then Printf.failwithf "Incorrect magic number in %s: %d" filename magic_number ();
  let samples = read_int32_be in_channel in
  let data = Tensor.zeros ~kind:Int [samples] in
  for sample = 0 to samples - 1 do
    let v = Option.value_exn (In_channel.input_byte in_channel) in
    Tensor.set_int1 data sample v;
  done;
  In_channel.close in_channel;
  data

let one_hot labels =
  let nsamples =
    match Tensor.shape labels with
    | [nsamples] -> nsamples
    | [] | _::_::_ -> failwith "unexpected shape"
  in
  let one_hot = Tensor.zeros [nsamples; label_count] in
  for idx = 0 to nsamples - 1 do
    let lbl = Tensor.get_int1 labels idx in
    Tensor.set_float2 one_hot idx lbl 1.
  done;
  one_hot

type t =
  { train_images : Tensor.t
  ; train_labels : Tensor.t
  ; test_images : Tensor.t
  ; test_labels : Tensor.t
  }

let read_files
      ?(train_image_file = "data/train-images-idx3-ubyte")
      ?(train_label_file = "data/train-labels-idx1-ubyte")
      ?(test_image_file = "data/t10k-images-idx3-ubyte")
      ?(test_label_file = "data/t10k-labels-idx1-ubyte")
      ()
  =
  { train_images = read_images train_image_file
  ; train_labels = read_labels train_label_file |> one_hot
  ; test_images = read_images test_image_file
  ; test_labels = read_labels test_label_file |> one_hot
  }

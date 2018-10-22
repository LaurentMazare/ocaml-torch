(* The readers implemented here are very inefficient as they read bytes one at a time. *)
open Base
module In_channel = Stdio.In_channel

let image_w = 28
let image_h = 28
let image_dim = image_w * image_h
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
  let data = Bigarray.Array2.create Float32 C_layout samples (rows * columns) in
  for sample = 0 to samples - 1 do
    for idx = 0 to rows * columns - 1 do
      let v = Option.value_exn (In_channel.input_byte in_channel) in
      data.{ sample, idx } <- Float.(of_int v / 255.)
    done;
  done;
  In_channel.close in_channel;
  Bigarray.genarray_of_array2 data |> Tensor.of_bigarray

let read_labels filename =
  let in_channel = In_channel.create filename in
  let magic_number = read_int32_be in_channel in
  if magic_number <> 2049
  then Printf.failwithf "Incorrect magic number in %s: %d" filename magic_number ();
  let samples = read_int32_be in_channel in
  let data = Bigarray.Array1.create Int64 C_layout samples in
  for sample = 0 to samples - 1 do
    let v = Option.value_exn (In_channel.input_byte in_channel) in
    data.{ sample } <- Int64.of_int v;
  done;
  In_channel.close in_channel;
  Bigarray.genarray_of_array1 data |> Tensor.of_bigarray

let read_files
      ?(train_image_file = "data/train-images-idx3-ubyte")
      ?(train_label_file = "data/train-labels-idx1-ubyte")
      ?(test_image_file = "data/t10k-images-idx3-ubyte")
      ?(test_label_file = "data/t10k-labels-idx1-ubyte")
      ?(with_caching = false)
      ()
  =
  let read () =
    { Dataset_helper.
      train_images = read_images train_image_file
    ; train_labels = read_labels train_label_file
    ; test_images = read_images test_image_file
    ; test_labels = read_labels test_label_file
    }
  in
  if with_caching
  then
    let dirname = Caml.Filename.dirname train_image_file in
    let cache_file = Caml.Filename.concat dirname "mnist-cache.ot" in
    Dataset_helper.read_with_cache ~cache_file ~read
  else read ()

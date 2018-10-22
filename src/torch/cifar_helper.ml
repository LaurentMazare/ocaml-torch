(* The readers implemented here are very inefficient as they read bytes one at a time. *)
open Base
module I = Stdio.In_channel

let image_w = 32
let image_h = 32
let image_c = 3
let image_dim = image_c * image_w * image_h
let label_count = 10

let labels =
  [ "airplane"
  ; "automobile"
  ; "bird"
  ; "cat"
  ; "deer"
  ; "dog"
  ; "frog"
  ; "horse"
  ; "ship"
  ; "truck"
  ]

let samples_per_file = 10_000

let read_byte in_channel =
  Option.value_exn (I.input_byte in_channel)

let read_file filename =
  let in_channel = I.create filename in
  let data =
    Bigarray.Genarray.create Float32 C_layout
      [| samples_per_file; image_c; image_w; image_h |]
  in
  let labels = Bigarray.Array1.create Int64 C_layout samples_per_file in
  for sample = 0 to 9999 do
    labels.{sample} <- read_byte in_channel |> Int64.of_int;
    for c = 0 to image_c - 1 do
      for w = 0 to image_w - 1 do
        for h = 0 to image_h - 1 do
          (Float.of_int (read_byte in_channel) /. 256.)
          |> Bigarray.Genarray.set data [| sample; c; w; h |]
        done
      done
    done
  done;
  I.close in_channel;
  data, Bigarray.genarray_of_array1 labels

let read_files ?(dirname = "data") ?(with_caching = false) () =
  let read () =
    let read_one filename =
      let images, labels = Caml.Filename.concat dirname filename |> read_file in
      Tensor.of_bigarray images, Tensor.of_bigarray labels
    in
    let train_images, train_labels =
      [ "data_batch_1.bin"
      ; "data_batch_2.bin"
      ; "data_batch_3.bin"
      ; "data_batch_4.bin"
      ; "data_batch_5.bin"
      ]
      |> List.map ~f:read_one
      |> List.unzip
    in
    let test_images, test_labels = read_one "test_batch.bin" in
    { Dataset_helper.
      train_images = Tensor.cat train_images ~dim:0
    ; train_labels = Tensor.cat train_labels ~dim:0
    ; test_images
    ; test_labels
    }
  in
  if with_caching
  then
    let cache_file = Caml.Filename.concat dirname "cifar-cache.ot" in
    Dataset_helper.read_with_cache ~cache_file ~read
  else read ()

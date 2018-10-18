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
  let labels = Bigarray.Array1.create Int C_layout samples_per_file in
  for sample = 0 to 9999 do
    labels.{sample} <- read_byte in_channel;
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
  data, labels

type t = Mnist_helper.t =
  { train_images : Tensor.t
  ; train_labels : Tensor.t
  ; test_images : Tensor.t
  ; test_labels : Tensor.t
  }

let read_files ?(dirname = "data") ?(with_caching = false) () =
  let read () =
    let read_one filename =
      let images, labels = Caml.Filename.concat dirname filename |> read_file in
      Tensor.of_bigarray images, Mnist_helper.one_hot labels ~label_count
    in
    let train_images, train_labels =
      [ "data_batch1.bin"
      ; "data_batch2.bin"
      ; "data_batch3.bin"
      ; "data_batch4.bin"
      ; "data_batch5.bin"
      ]
      |> List.map ~f:read_one
      |> List.unzip
    in
    let test_images, test_labels = read_one "test_batch.bin" in
    { train_images = Tensor.cat train_images ~dim:0
    ; train_labels = Tensor.cat train_labels ~dim:0
    ; test_images
    ; test_labels
    }
  in
  if with_caching
  then begin
    let cached_file = Caml.Filename.concat dirname "cifar-cache.ot" in
    try
      Serialize.load_multi
        ~names:["traini"; "trainl"; "testi"; "testl"] ~filename:cached_file
      |> function
      | [ train_images; train_labels; test_images; test_labels ] ->
        { train_images; train_labels; test_images; test_labels }
      | _ -> assert false
    with
    | _ ->
      Stdio.eprintf
        "Cannot read from cached file %s, regenerating...\n%!"
        cached_file;
      let t = read () in
      begin
        try
          Serialize.save_multi ~filename:cached_file
            ~named_tensors:
              [ "traini", t.train_images; "trainl", t.train_labels
              ; "testi", t.test_images; "testl", t.test_labels
              ]
        with
        | _ ->
          Stdio.eprintf "Unable to save cached file %s.\n%!" cached_file
      end;
      t
  end else read ()

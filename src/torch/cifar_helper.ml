(* The readers implemented here are very inefficient as they read bytes one at a time. *)
open Base

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

let read_file filename =
  let content = Dataset_helper.read_char_tensor filename in
  let images =
    Tensor.zeros [ samples_per_file; image_c; image_w; image_h ] ~kind:(T Uint8)
  in
  let labels = Tensor.zeros [ samples_per_file ] ~kind:(T Uint8) in
  for sample = 0 to 9999 do
    let content_offset = 3073 * sample in
    Tensor.copy_
      (Tensor.narrow labels ~dim:0 ~start:sample ~length:1)
      ~src:(Tensor.narrow content ~dim:0 ~start:content_offset ~length:1);
    Tensor.copy_
      (Tensor.narrow images ~dim:0 ~start:sample ~length:1)
      ~src:
        (Tensor.narrow content ~dim:0 ~start:(content_offset + 1) ~length:3072
        |> Tensor.view ~size:[ 1; image_c; image_w; image_h ])
  done;
  ( Tensor.(((to_type images ~type_:Float / f 255.) - f 0.5) * f 4.)
  , Tensor.to_type labels ~type_:Int64 )

let read_files ?(dirname = "data") ?(with_caching = false) () =
  let read_one filename = Caml.Filename.concat dirname filename |> read_file in
  let read () =
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
    { Dataset_helper.train_images = Tensor.cat train_images ~dim:0
    ; train_labels = Tensor.cat train_labels ~dim:0
    ; test_images
    ; test_labels
    }
  in
  if with_caching
  then (
    let cache_file = Caml.Filename.concat dirname "cifar-cache.ot" in
    Dataset_helper.read_with_cache ~cache_file ~read)
  else read ()

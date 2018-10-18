(* Images have shape [ samples; 728 ]. Labels are one-hot encoded with
   shape [ samples; 10 ]. *)
val read_files
  :  ?train_image_file:string
  -> ?train_label_file:string
  -> ?test_image_file:string
  -> ?test_label_file:string
  -> ?with_caching:bool
  -> unit
  -> Dataset_helper.t

val image_w : int
val image_h : int
val image_dim : int
val label_count : int

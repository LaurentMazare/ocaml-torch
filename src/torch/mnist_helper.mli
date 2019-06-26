(* Images have shape [ samples; 728 ]. Labels are one-hot encoded with
   shape [ samples; 10 ]. *)
val read_files : ?prefix:string -> unit -> ([ `f32 ], [ `i64 ]) Dataset_helper.t
val image_w : int
val image_h : int
val image_dim : int
val label_count : int

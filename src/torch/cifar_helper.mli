(* This module uses the binary version of the CIFAR-10 dataset.
   The dataset can be downloaded from https://www.cs.toronto.edu/~kriz/cifar.html.
   Images have shape [ samples; 32; 32; 3 ]. Labels are one-hot encoded with
   shape [ samples; 10 ].
*)
val read_files
  :  ?dirname:string
  -> ?with_caching:bool
  -> unit
  -> ([ `f32 ], [ `i64 ]) Dataset_helper.t

val image_c : int
val image_w : int
val image_h : int
val image_dim : int
val label_count : int
val labels : string list

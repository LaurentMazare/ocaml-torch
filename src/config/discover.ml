open Base
module C = Configurator.V1

let () =
  C.main ~name:"torch-config" (fun _c ->
      let libroot =
        try
          Caml.Sys.getenv "LIBTORCH"
        with
        | _ -> failwith "The LIBTORCH environment variable is not set!"
      in
      let cxx_flags =
        [ "-isystem" ; Printf.sprintf "%s/include" libroot
        ; "-isystem"; Printf.sprintf "%s/include/torch/csrc/api/include" libroot
        ]
      in
      let c_library_flags =
        [ Printf.sprintf "-Wl,-R/%s/lib" libroot
        ; Printf.sprintf "-L/%s/lib" libroot
        ; "-lc10"; "-lcaffe2"; "-ltorch"
        ]
      in
      C.Flags.write_sexp "cxx_flags.sexp" cxx_flags;
      C.Flags.write_sexp "c_library_flags.sexp" c_library_flags)

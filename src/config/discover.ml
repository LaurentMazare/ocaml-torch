open Base
module C = Configurator.V1

let extract_flags c ~package =
  match C.Pkg_config.get c with
  | None -> [], []
  | Some pc ->
    let cuda_cxx_flags, cuda_c_library_flags =
      match C.Pkg_config.query pc ~package with
      | None -> [], []
      | Some deps -> deps.cflags, deps.libs
    in
    cuda_cxx_flags, cuda_c_library_flags

let () =
  C.main ~name:"torch-config" (fun c ->
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
      let cuda_cxx_flags, cuda_c_library_flags = extract_flags c ~package:"cuda" in
      let nvrtc_cxx_flags, nvrtc_c_library_flags = extract_flags c ~package:"nvrtc" in
      C.Flags.write_sexp "cxx_flags.sexp"
        (cxx_flags @ cuda_cxx_flags @ nvrtc_cxx_flags);
      C.Flags.write_sexp "c_library_flags.sexp"
        (c_library_flags @ cuda_c_library_flags @ nvrtc_c_library_flags))

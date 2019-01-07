open Bigarray

type 'kind buffer = ('a, 'b, c_layout) Array1.t
  constraint 'kind = ('a, 'b) kind

type float32 = (float, float32_elt) kind
type int8 = (int, int8_unsigned_elt) kind

external png : string -> w:int -> h:int -> c:int -> int8 buffer -> unit = "ml_stbi_write_png"
external bmp : string -> w:int -> h:int -> c:int -> int8 buffer -> unit = "ml_stbi_write_bmp"
external tga : string -> w:int -> h:int -> c:int -> int8 buffer -> unit = "ml_stbi_write_tga"
external hdr : string -> w:int -> h:int -> c:int -> float32 buffer-> unit = "ml_stbi_write_hdr"
external jpg : string -> w:int -> h:int -> c:int -> quality:int -> int8 buffer-> unit =
  "ml_stbi_write_jpg_bytecode" "ml_stbi_write_jpg_native"

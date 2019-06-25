open Bigarray
open Base

type 'a result = ('a, [ `Msg of string ]) Result.t
type 'kind buffer = ('a, 'b, c_layout) Array1.t constraint 'kind = ('a, 'b) kind

type 'kind t =
  { width : int
  ; height : int
  ; channels : int
  ; offset : int
  ; stride : int
  ; data : 'kind buffer
  }

type float32 = (float, float32_elt) kind
type int8 = (int, int8_unsigned_elt) kind

external load_unmanaged : ?channels:int -> string -> int8 t result = "ml_stbi_load"
external loadf_unmanaged : ?channels:int -> string -> float32 t result = "ml_stbi_loadf"

external decode_unmanaged
  :  ?channels:int
  -> _ buffer
  -> int8 t result
  = "ml_stbi_load_mem"

external decodef_unmanaged
  :  ?channels:int
  -> _ buffer
  -> float32 t result
  = "ml_stbi_loadf_mem"

external ml_stbi_image_free : _ buffer -> unit = "ml_stbi_image_free"

let free_unmanaged image = ml_stbi_image_free image.data

let clone buf =
  let buf' = Array1.create (Array1.kind buf) c_layout (Array1.dim buf) in
  Array1.blit buf buf';
  buf'

let manage f ?channels filename =
  match f ?channels filename with
  | Result.Error _ as err -> err
  | Result.Ok image ->
    let managed = { image with data = clone image.data } in
    free_unmanaged image;
    Result.Ok managed

let load ?channels filename = manage load_unmanaged ?channels filename
let loadf ?channels filename = manage loadf_unmanaged ?channels filename
let decode ?channels filename = manage decode_unmanaged ?channels filename
let decodef ?channels filename = manage decodef_unmanaged ?channels filename

let image ~width ~height ~channels ?(offset = 0) ?(stride = width * channels) data =
  let size = Array1.dim data in
  if width < 0
  then Result.Error (`Msg "width should be positive")
  else if height < 0
  then Result.Error (`Msg "height should be positive")
  else if channels < 0 || channels > 4
  then Result.Error (`Msg "channels should be between 1 and 4")
  else if offset < 0
  then Result.Error (`Msg "offset should be positive")
  else if offset + (stride * height) > size
  then Result.Error (`Msg "image does not fit in buffer")
  else Result.Ok { width; height; channels; offset; stride; data }

let width t = t.width
let height t = t.height
let channels t = t.channels
let data t = t.data

let validate_mipmap t1 t2 =
  if t1.channels <> t2.channels
  then invalid_arg "mipmap: images have different number of channels";
  if t1.width / 2 <> t2.width || t1.height / 2 <> t2.height
  then invalid_arg "mipmap: second image size should exactly be half of first image"

external mipmap : int8 t -> int8 t -> unit = "ml_stbi_mipmap"
external mipmapf : float32 t -> float32 t -> unit = "ml_stbi_mipmapf"

let mipmap t1 t2 =
  validate_mipmap t1 t2;
  mipmap t1 t2

let mipmapf t1 t2 =
  validate_mipmap t1 t2;
  mipmapf t1 t2

external vflip : int8 t -> unit = "ml_stbi_vflip"
external vflipf : float32 t -> unit = "ml_stbi_vflipf"

(** Blur the image *)
external expblur : int8 t -> radius:float -> unit = "ml_stbi_expblur"

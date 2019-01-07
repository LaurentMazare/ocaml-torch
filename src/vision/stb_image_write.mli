(*
  Stb_image_write for OCaml by Frédéric Bour <frederic.bour(_)lakaban.net>
  To the extent possible under law, the person who associated CC0 with
  Stb_image_write for OCaml has waived all copyright and related or neighboring
  rights to Stb_image_write for OCaml.

  You should have received a copy of the CC0 legalcode along with this
  work. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.

  Website: https://github.com/let-def/stb_image_write
  stb_image_write is a public domain library by Sean Barrett,
  http://nothings.org/
  Version 0.1, September 2015
*)
open Bigarray

(*####################*)
(** {1 Image writing} *)

(** [buffer] simply is an alias to a bigarray with c_layout.
    Two kind of pixel buffers are manipulated:
    - int8 for images with 8-bit channels
    - float32 for images with floating point channels

    Content of an image with [c] channels of width [w] and height [h] is
    represented as a contiguous sequence of items such that:
    - channels are interleaved
    - each pixel is made of [c] items
    - each line is made of [w] pixels
    - image is made of [h] lines *)

type 'kind buffer = ('a, 'b, c_layout) Array1.t
  constraint 'kind = ('a, 'b) kind

type float32 = (float, float32_elt) kind
type int8 = (int, int8_unsigned_elt) kind

val png : string -> w:int -> h:int -> c:int -> int8 buffer -> unit
val bmp : string -> w:int -> h:int -> c:int -> int8 buffer -> unit
val tga : string -> w:int -> h:int -> c:int -> int8 buffer -> unit
val hdr : string -> w:int -> h:int -> c:int -> float32 buffer-> unit
val jpg : string -> w:int -> h:int -> c:int -> quality:int -> int8 buffer-> unit

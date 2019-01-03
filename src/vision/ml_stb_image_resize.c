#include <assert.h>
#include <stdio.h>
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/bigarray.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

static int validate_dim(value ba, value w, value h, value nchannels, int byte)
{
  size_t sz = caml_ba_byte_size(Caml_ba_array_val(ba));
  size_t expected = Int_val(w) * Int_val(h) * Int_val(nchannels) * byte;
  return (expected <= sz);
}

CAMLprim value ml_stbir_resize(value in_ba, value in_w, value in_h,
                               value out_ba, value out_w, value out_h,
                               value nchannels)
{
  CAMLparam5(in_ba, in_w, in_h, out_ba, out_w);
  CAMLxparam2(out_h, nchannels);
  int result;

  if (validate_dim(in_ba, in_w, in_h, nchannels, 1)
      && validate_dim(out_ba, out_w, out_h, nchannels, 1))
    result = stbir_resize_uint8(Caml_ba_data_val(in_ba), Int_val(in_w), Int_val(in_h), 0,
                                Caml_ba_data_val(out_ba), Int_val(out_w), Int_val(out_h), 0,
                                Int_val(nchannels));
  else
    result = 0;

  CAMLreturn(Val_int(result));
}

CAMLprim value ml_stbir_resize_bytecode(value * argv, int nargs)
{
  return ml_stbir_resize(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);
}

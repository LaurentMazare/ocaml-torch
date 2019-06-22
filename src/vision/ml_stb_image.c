#include <assert.h>
#include <stdio.h>
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/bigarray.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static int Channels_val(value channel)
{
  CAMLparam1(channel);
  int ret = 0;
  if (channel != Val_unit)
    ret = Long_val(Field(channel, 0));
  CAMLreturn(ret);
}

static value return_image(void *data, int ty, int x, int y, int n)
{
  CAMLparam0();
  CAMLlocal3(ret, tup, ba);

  ba = caml_ba_alloc_dims(ty | CAML_BA_C_LAYOUT, 1, data, x * y * n);

  tup = caml_alloc(6, 0);
  Store_field(tup, 0, Val_long(x));
  Store_field(tup, 1, Val_long(y));
  Store_field(tup, 2, Val_long(n));
  Store_field(tup, 3, Val_long(0));
  Store_field(tup, 4, Val_long(x * n));
  Store_field(tup, 5, ba);

  /* Result.Ok tup */
  ret = caml_alloc(1, 0);
  Store_field(ret, 0, tup);

  CAMLreturn(ret);
}

static value return_failure(void)
{
  CAMLparam0();
  CAMLlocal3(ret, str, err);

  str = caml_copy_string(stbi_failure_reason());

  /* `Msg "str" */
  err = caml_alloc(2, 0);
  Store_field(err, 0, Val_long(3854881));
  Store_field(err, 1, str);

  /* Result.Error (`Msg "str") */
  ret = caml_alloc(1, 1);
  Store_field(ret, 0, err);

  CAMLreturn(ret);
}

CAMLprim value ml_stbi_load(value channels, value filename)
{
  CAMLparam2(channels, filename);
  CAMLlocal1(ret);

  int x, y, n;
	unsigned char* image_data =
    stbi_load(String_val(filename), &x, &y, &n, Channels_val(channels));

  if (image_data)
    ret = return_image(image_data, CAML_BA_UINT8, x, y, n);
  else
    ret = return_failure();

  CAMLreturn(ret);
}

CAMLprim value ml_stbi_loadf(value channels, value filename)
{
  CAMLparam2(channels, filename);
  CAMLlocal1(ret);

  int x, y, n;
	float* image_data =
    stbi_loadf(String_val(filename), &x, &y, &n, Channels_val(channels));

  if (image_data)
    ret = return_image(image_data, CAML_BA_FLOAT32, x, y, n);
  else
    ret = return_failure();

  CAMLreturn(ret);
}

CAMLprim value ml_stbi_load_mem(value channels, value mem)
{
  CAMLparam2(channels, mem);
  CAMLlocal1(ret);

  int x, y, n;
	unsigned char* image_data =
    stbi_load_from_memory(Caml_ba_data_val(mem),
        caml_ba_byte_size(Caml_ba_array_val(mem)),
        &x, &y, &n, Channels_val(channels));

  if (image_data)
    ret = return_image(image_data, CAML_BA_UINT8, x, y, n);
  else
    ret = return_failure();

  CAMLreturn(ret);
}

CAMLprim value ml_stbi_loadf_mem(value channels, value mem)
{
  CAMLparam2(channels, mem);
  CAMLlocal1(ret);

  int x, y, n;
	float* image_data =
    stbi_loadf_from_memory(Caml_ba_data_val(mem),
        caml_ba_byte_size(Caml_ba_array_val(mem)),
        &x, &y, &n, Channels_val(channels));

  if (image_data)
    ret = return_image(image_data, CAML_BA_FLOAT32, x, y, n);
  else
    ret = return_failure();

  CAMLreturn(ret);
}

CAMLprim value ml_stbi_image_free(value ba)
{
  CAMLparam1(ba);
  void *data = Caml_ba_data_val(ba);

  assert (data);
  stbi_image_free(data);
  Caml_ba_data_val(ba) = NULL;

  CAMLreturn(Val_unit);
}

#define POUT(x,n) pout[x] = (pin[x] + pin[n + x] + pin[w * n + n] + pin[w * n + x]) / 4
#define POUTf(x,n) pout[x] = (pin[x] + pin[n + x] + pin[w * n + n] + pin[w * n + x]) / 4.0f

#define LOOP(w,h,n) \
  for (unsigned int y = 0, w2 = (w) / 2, h2 = (h) / 2; \
       y < h2; ++y, pin0 += sin, pin = pin0, pout0 += sout, pout = pout0) \
    for (unsigned int x = 0; x < w2; ++x, pin += 2 * n, pout += n)

CAMLprim value ml_stbi_mipmap(value img_in, value img_out)
{
  CAMLparam2(img_in, img_out);
  unsigned char *pin, *pout,
    *pin0 = Caml_ba_data_val(Field(img_in, 5)),
    *pout0 = Caml_ba_data_val(Field(img_out, 5));
  assert (pin0 && pout0);

  pin0 += Long_val(Field(img_in, 3));
  pout0 += Long_val(Field(img_out, 3));

  unsigned int
    sin = Long_val(Field(img_in, 4)),
    sout = Long_val(Field(img_out, 4)),
    w = Long_val(Field(img_in, 0)),
    h = Long_val(Field(img_in, 1));

  switch (Long_val(Field(img_in, 2))) {
    case 1:
      LOOP(w, h, 1) { POUT(0, 1); }
      break;
    case 2:
      LOOP(w, h, 2) { POUT(0, 2); POUT(1, 2); }
      break;
    case 3:
      LOOP(w, h, 3) { POUT(0, 3); POUT(1, 3); POUT(2, 3); }
      break;
    case 4:
      LOOP(w, h, 4) { POUT(0, 4); POUT(1, 4); POUT(2, 4); POUT(3, 4); }
      break;
  }

  CAMLreturn(Val_unit);
}

CAMLprim value ml_stbi_mipmapf(value img_in, value img_out)
{
  CAMLparam2(img_in, img_out);
  float *pin, *pout,
    *pin0 = Caml_ba_data_val(Field(img_in, 5)),
    *pout0 = Caml_ba_data_val(Field(img_out, 5));
  assert (pin0 && pout0);

  pin0 += Long_val(Field(img_in, 3));
  pout0 += Long_val(Field(img_out, 3));

  unsigned int
    sin = Long_val(Field(img_in, 4)),
    sout = Long_val(Field(img_out, 4)),
    w = Long_val(Field(img_in, 0)),
    h = Long_val(Field(img_in, 1));

  switch (Long_val(Field(img_in, 2))) {
    case 1:
      LOOP(w, h, 1) { POUTf(0, 1); }
      break;
    case 2:
      LOOP(w, h, 2) { POUTf(0, 2); POUTf(1, 2); }
      break;
    case 3:
      LOOP(w, h, 3) { POUTf(0, 3); POUTf(1, 3); POUTf(2, 3); }
      break;
    case 4:
      LOOP(w, h, 4) { POUTf(0, 4); POUTf(1, 4); POUTf(2, 4); POUTf(3, 4); }
      break;
  }

  CAMLreturn(Val_unit);
}

static void memswap(void *i0, void *i1, size_t count)
{
  unsigned char *p0 = i0, *p1 = i1;
  for (size_t i = 0; i < count; ++i)
  {
    unsigned char tmp = p0[i];
    p0[i] = p1[i];
    p1[i] = tmp;
  }
}

CAMLprim value ml_stbi_vflip(value img)
{
  CAMLparam1(img);
  unsigned char *ptop = Caml_ba_data_val(Field(img, 5));
  assert (ptop);
  ptop += Long_val(Field(img, 3));

  unsigned int
    w = Long_val(Field(img, 0)),
    h = Long_val(Field(img, 1)),
    n = Long_val(Field(img, 2)),
    stride = Long_val(Field(img, 4)),
    row = w * n;

  unsigned char *pbot = ptop + (stride * h - stride);
  w = w * n;

  for (unsigned int y = 0; y < h; y++)
  {
    memswap(ptop, pbot, row);
    ptop += stride;
    pbot -= stride;
  }

  CAMLreturn(Val_unit);
}

CAMLprim value ml_stbi_vflipf(value img)
{
  CAMLparam1(img);
  float *ptop = Caml_ba_data_val(Field(img, 5));
  assert (ptop);
  ptop += Long_val(Field(img, 3));

  unsigned int
    w = Long_val(Field(img, 0)),
    h = Long_val(Field(img, 1)),
    n = Long_val(Field(img, 2)),
    stride = Long_val(Field(img, 4)),
    row = w * n * sizeof(float);

  float *pbot = ptop + (stride * h - stride);
  w = w * n;

  for (unsigned int y = 0; y < h; y++)
  {
    memswap(ptop, pbot, row);
    ptop += stride;
    pbot -= stride;
  }

  CAMLreturn(Val_unit);
}

// Based on Exponential blur, Jani Huhtanen, 2006
// and [https://github.com/memononen/fontstash](fontstash), Mikko Mononen, 2014

#define APREC 16
#define ZPREC 7

#define APPROX(alpha, reg, acc) \
  ((alpha * (((int)(reg) << ZPREC) - acc)) >> APREC)

#define BLUR0(reg, acc) int acc = (int)(reg) << ZPREC

#define BLUR(reg, acc) \
  do { \
    acc += APPROX(alpha, reg, acc); \
    reg = (unsigned char)(acc >> ZPREC); \
  } while (0)

#define OUTERLOOP(var, ptr, bound, stride) \
  for (unsigned char *_limit = ptr + bound * stride, *var = ptr; var < _limit; var += stride)

#define INNERLOOP(var, bound, stride, BODY) \
  do { \
    int var; \
    for (var = stride; var < bound * stride; var += stride) BODY; \
    for (var = (bound - 2) * stride; var >= 0; var -= stride) BODY; \
    for (var = stride; var < bound * stride; var += stride) BODY; \
    for (var = (bound - 2) * stride; var >= 0; var -= stride) BODY; \
  } while (0)

static void expblur4(unsigned char* ptr, int w, int h, int stride, int alpha)
{
  OUTERLOOP(dst, ptr, h, stride)
  {
    BLUR0(dst[0], acc0);
    BLUR0(dst[1], acc1);
    BLUR0(dst[2], acc2);
    BLUR0(dst[3], acc3);
    INNERLOOP(x, w, 4,
        {
         BLUR(dst[x+0], acc0);
         BLUR(dst[x+1], acc1);
         BLUR(dst[x+2], acc2);
         BLUR(dst[x+3], acc3);
        });
  }

  OUTERLOOP(dst, ptr, w, 4)
  {
    BLUR0(dst[0], acc0);
    BLUR0(dst[1], acc1);
    BLUR0(dst[2], acc2);
    BLUR0(dst[3], acc3);
    INNERLOOP(y, h, stride,
        {
         BLUR(dst[y+0], acc0);
         BLUR(dst[y+1], acc1);
         BLUR(dst[y+2], acc2);
         BLUR(dst[y+3], acc3);
        });
  }
}

static void expblur3(unsigned char* ptr, int w, int h, int stride, int alpha)
{
  OUTERLOOP(dst, ptr, h, stride)
  {
    BLUR0(dst[0], acc0);
    BLUR0(dst[1], acc1);
    BLUR0(dst[2], acc2);
    INNERLOOP(x, w, 3,
        {
         BLUR(dst[x+0], acc0);
         BLUR(dst[x+1], acc1);
         BLUR(dst[x+2], acc2);
        });
  }

  OUTERLOOP(dst, ptr, w, 3)
  {
    BLUR0(dst[0], acc0);
    BLUR0(dst[1], acc1);
    BLUR0(dst[2], acc2);
    INNERLOOP(y, h, stride,
        {
         BLUR(dst[y+0], acc0);
         BLUR(dst[y+1], acc1);
         BLUR(dst[y+2], acc2);
        });
  }
}

static void expblur2(unsigned char* ptr, int w, int h, int stride, int alpha)
{
  OUTERLOOP(dst, ptr, h, stride)
  {
    BLUR0(dst[0], acc0);
    BLUR0(dst[1], acc1);
    INNERLOOP(x, w, 2,
        {
         BLUR(dst[x+0], acc0);
         BLUR(dst[x+1], acc1);
        });
  }

  OUTERLOOP(dst, ptr, w, 2)
  {
    BLUR0(dst[0], acc0);
    BLUR0(dst[1], acc1);
    INNERLOOP(y, h, stride,
        {
         BLUR(dst[y+0], acc0);
         BLUR(dst[y+1], acc1);
        });
  }
}

static void expblur1(unsigned char* ptr, int w, int h, int stride, int alpha)
{
  OUTERLOOP(dst, ptr, h, stride)
  {
    BLUR0(dst[0], acc0);
    INNERLOOP(x, w, 1,
        {
         BLUR(dst[x+0], acc0);
        });
  }

  OUTERLOOP(dst, ptr, w, 1)
  {
    BLUR0(dst[0], acc0);
    INNERLOOP(y, h, stride,
        {
         BLUR(dst[y+0], acc0);
        });
  }
}

static void expblur(unsigned char* ptr, int w, int h, int channels, int stride, float radius)
{
	int i, alpha;
	float sigma;

  if (radius < 0.01) return;

  // Calculate the alpha such that 90% of the kernel is within the radius.
  // (Kernel extends to infinity)
	sigma = radius * 0.57735f; // 1 / sqrt(3)

  // Improve blur quality by doing two pass
  // blur(sigma1) o blur(sigma2) = blur(sqrt(sqr(sigma1)*sqr(sigma2)))
  sigma = sigma * 0.707106f; // 1 / sqrt(2)

	alpha = (int)((1<<APREC) * (1.0f - expf(-2.3f / (sigma + 1.0f))));

  switch (channels)
  {
    case 1: expblur1(ptr, w, h, stride, alpha); break;
    case 2: expblur2(ptr, w, h, stride, alpha); break;
    case 3: expblur3(ptr, w, h, stride, alpha); break;
    case 4: expblur4(ptr, w, h, stride, alpha); break;
    default: abort();
  }
}

CAMLprim value ml_stbi_expblur(value img, value radius)
{
  CAMLparam2(img, radius);

  unsigned char *ptr = Caml_ba_data_val(Field(img, 5));
  assert (ptr);
  ptr += Long_val(Field(img, 3));

  unsigned int
    w = Long_val(Field(img, 0)),
    h = Long_val(Field(img, 1)),
    n = Long_val(Field(img, 2)),
    stride = Long_val(Field(img, 4));

  expblur(ptr, w, h, n, stride, Double_val(radius));
  CAMLreturn(Val_unit);
}

#ifndef __TORCH_API_H__
#define __TORCH_API_H__
#include<stdint.h>

#ifdef __cplusplus
extern "C" {
typedef torch::Tensor *tensor;
typedef torch::Scalar *scalar;
typedef torch::optim::Optimizer *optimizer;
#define PROTECT(x) \
  try { \
    x \
  } catch (const exception& e) { \
    caml_failwith(strdup(e.what())); \
  }
#else
typedef void *tensor;
typedef void *optimizer;
typedef void *scalar;
#endif

tensor at_tensor_of_data(void *vs, long int *dims, int ndims, int element_size_in_bytes, int type);
void at_copy_data(tensor tensor, void *vs, int64_t numel, int element_size_in_bytes);
tensor at_float_vec(double *values, int value_len, int type);
tensor at_int_vec(int64_t *values, int value_len, int type);

int at_defined(tensor);
int at_dim(tensor);
void at_shape(tensor, int *);
int at_scalar_type(tensor);

void at_backward(tensor);
int at_requires_grad(tensor);

tensor at_get(tensor, int index);
double at_double_value(tensor);
int64_t at_int64_value(tensor);
void at_fill_double(tensor, double);
void at_fill_int64(tensor, int64_t);
void at_copy_(tensor dst, tensor src);

void at_print(tensor);
void at_save(tensor, char *filename);
tensor at_load(char *filename);

void at_save_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename);
/* [at_load_multi] takes as input an array of nullptr for [tensors]. */
void at_load_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename);
/* [at_load_multi_] takes as input an array of allocation [tensors]. */
void at_load_multi_(tensor *tensors, char **tensor_names, int ntensors, char *filename);

void at_free(tensor);

tensor at_nll_loss(tensor t, tensor targets, int reduction);

optimizer ato_adam(tensor *, int ntensors, double learning_rate);
optimizer ato_sgd(tensor *,
                  int ntensors,
                  double learning_rate,
                  double momentum,
                  double dampening,
                  double weight_decay,
                  int nesterov);
void ato_set_learning_rate(optimizer, double learning_rate);
void ato_zero_grad(optimizer);
void ato_step(optimizer);
void ato_free(optimizer);

scalar ats_int(int64_t);
scalar ats_float(double);
void ats_free(scalar);

int atc_cuda_device_count();
int atc_cuda_is_available();
int atc_cudnn_is_available();

#include "torch_api_generated.h"

#ifdef __cplusplus
};
#endif

#endif

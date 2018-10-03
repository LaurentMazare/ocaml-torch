#ifndef __TORCH_API_H__
#define __TORCH_API_H__
#ifdef __cplusplus
extern "C" {
#endif

typedef void *tensor;

tensor at_zeros();
tensor at_ones();
tensor at_add(tensor, tensor);
void at_free(tensor);

#ifdef __cplusplus
};
#endif
#endif

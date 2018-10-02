#ifdef __cplusplus
extern "C" {
#endif

typedef void *tensor;

tensor at_zeros();
tensor at_ones();
tensor add(tensor, tensor);
void free(tensor);

#ifdef __cplusplus
};
#endif

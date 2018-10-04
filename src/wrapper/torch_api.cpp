#include <torch/torch.h>
#include<vector>
#include<caml/fail.h>
#include "torch_api.h"

#define PROTECT(x) \
  try { \
    x \
  } catch (const exception& e) { \
    caml_failwith(strdup(e.what())); \
  }

using namespace std;

vector<long int> of_carray(int *vs, int len) {
  vector<long int> result;
  for (int i = 0; i < len; ++i) result.push_back(vs[i]);
  return result;
}

tensor at_float_vec(double *vs, int len, int type) {
  PROTECT(
    torch::Tensor tensor = torch::empty({len}, torch::ScalarType(type)); 
    for (int i = 0; i < len; ++i) tensor[i] = vs[i];
    return new torch::Tensor(tensor);
  )
}

tensor at_int_vec(int64_t *vs, int len, int type) {
  PROTECT(
    torch::Tensor tensor = torch::empty({len}, torch::ScalarType(type)); 
    for (int i = 0; i < len; ++i) tensor[i] = vs[i];
    return new torch::Tensor(tensor);
  )
}

tensor at_zeros(int *dim_list, int dim_len, int type) {
  PROTECT(
    return new torch::Tensor(torch::zeros(of_carray(dim_list, dim_len), torch::ScalarType(type)));
  )
}

tensor at_ones(int *dim_list, int dim_len, int type) {
  PROTECT(
    return new torch::Tensor(torch::ones(of_carray(dim_list, dim_len), torch::ScalarType(type)));
  )
}

tensor at_rand(int *dim_list, int dim_len) {
  PROTECT(
    return new torch::Tensor(torch::rand(of_carray(dim_list, dim_len)));
  )
}

tensor at_reshape(tensor t, int *dim_list, int dim_len) {
  PROTECT(
    return new torch::Tensor(torch::reshape(*t, of_carray(dim_list, dim_len)));
  )
}

int at_dim(tensor t) {
  PROTECT(return t->dim();)
}

void at_shape(tensor t, int *dims) {
  PROTECT(
    int i = 0;
    for (int dim : t->sizes()) dims[i++] = dim;
  )
}

int at_scalar_type(tensor t) {
  PROTECT(
    return static_cast<int>(t->scalar_type());
  )
}

tensor at_add(tensor t1, tensor t2) {
  PROTECT(
    return new torch::Tensor(add(*t1, *t2));
  )
}

tensor at_sub(tensor t1, tensor t2) {
  PROTECT(
    return new torch::Tensor(sub(*t1, *t2));
  )
}

tensor at_mul(tensor t1, tensor t2) {
  PROTECT(
    return new torch::Tensor(mul(*t1, *t2));
  )
}

tensor at_div(tensor t1, tensor t2) {
  PROTECT(
    return new torch::Tensor(div(*t1, *t2));
  )
}

tensor at_pow(tensor t1, tensor t2) {
  PROTECT(
    return new torch::Tensor(pow(*t1, *t2));
  )
}

tensor at_matmul(tensor t1, tensor t2) {
  PROTECT(
    return new torch::Tensor(matmul(*t1, *t2));
  )
}

void at_backward(tensor t) {
  PROTECT(t->backward();)
}

tensor at_grad(tensor t) {
  PROTECT(return new torch::Tensor(t->grad());)
}

tensor at_set_requires_grad(tensor t, int b) {
  PROTECT(return new torch::Tensor(t->set_requires_grad(b));)
}

tensor at_get(tensor t, int index) {
  PROTECT(return new torch::Tensor((*t)[index]);)
}

tensor at_select(tensor t, int dim, int index) {
  PROTECT(return new torch::Tensor(select(*t, dim, index));)
}

double at_double_value(tensor t) {
  PROTECT(return t->item<double>();)
}

int64_t at_int64_value(tensor t) {
  PROTECT(return t->item<int64_t>();)
}

void at_fill_double(tensor t, double v) {
  PROTECT(t->fill_(v);)
}

void at_fill_int64(tensor t, int64_t v) {
  PROTECT(t->fill_(v);)
}

void at_set_double2(tensor t, int dim1, int dim2, double v) {
  PROTECT(
     auto dtype = t->dtype();
     if (dtype == torch::ScalarType::Float) {
       auto accessor = (t->accessor<float, 2>());
       accessor[dim1][dim2] = v;
     }
     else if (dtype == torch::ScalarType::Double) {
       auto accessor = (t->accessor<double, 2>());
       accessor[dim1][dim2] = v;
     }
     else
       caml_failwith("unexpected tensor type");
  )
}

void at_print(tensor t) {
  PROTECT(
    torch::Tensor *tensor = (torch::Tensor*)t;
    cout << *tensor << endl;
  )
}

void at_save(tensor t, char *filename) {
  PROTECT(torch::save(*t, filename);)
}

tensor at_load(char *filename) {
  PROTECT(return new torch::Tensor(torch::load(filename));)
}

void at_free(tensor t) {
  free(t);
}

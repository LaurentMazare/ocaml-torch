#include <torch/torch.h>
#include<vector>
#include<caml/fail.h>
#include "torch_api.h"

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

tensor at_sum(tensor t) {
  PROTECT(
    return new torch::Tensor(sum(*t));
  )
}

tensor at_mean(tensor t) {
  PROTECT(
    return new torch::Tensor(mean(*t));
  )
}

tensor at_softmax(tensor t) {
  PROTECT(
    return new torch::Tensor(softmax(*t, -1));
  )
}

tensor at_neg(tensor t) {
  PROTECT(
    return new torch::Tensor(neg(*t));
  )
}

void at_sub_assign(tensor t1, tensor t2) {
  PROTECT(
    *t1 -= *t2;
  )
}

tensor at_eq(tensor t1, tensor t2) {
  PROTECT(
    return new torch::Tensor(eq(*t1, *t2));
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

int at_requires_grad(tensor t) {
  PROTECT(return t->requires_grad();)
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
  delete(t);
}

#include "torch_api_generated.cpp.h"

#include<torch/csrc/autograd/engine.h>
#include<torch/torch.h>
#include<ATen/autocast_mode.h>
#include<torch/script.h>
#include<torch/csrc/jit/mobile/import_data.h>
#include<stdexcept>
#include<vector>
#include<caml/fail.h>
#undef invalid_argument
#include "torch_api.h"

using namespace std;

void at_manual_seed(int64_t seed) {
  torch::manual_seed(seed);
}

vector<torch::Tensor> of_carray_tensor(torch::Tensor **vs, int len) {
  vector<torch::Tensor> result;
  for (int i = 0; i < len; ++i) result.push_back(*(vs[i]));
  return result;
}

c10::List<c10::optional<torch::Tensor>> of_carray_tensor_opt(torch::Tensor **vs, int len) {
  vector<c10::optional<torch::Tensor>> result;
  for (int i = 0; i < len; ++i) {
    result.push_back(vs[i] != nullptr ? c10::optional<torch::Tensor>(*(vs[i])) : c10::nullopt);
  }
  return c10::List<c10::optional<torch::Tensor>>(result);
}

at::Device device_of_int(int d) {
    if (d < 0) return at::Device(at::kCPU);
    return at::Device(at::kCUDA, /*index=*/d);
}

tensor at_new_tensor() {
  PROTECT(
    return new torch::Tensor();
  )
  return nullptr;
}

tensor at_tensor_of_data(void *vs, int64_t *dims, int ndims, int element_size_in_bytes, int type) {
  PROTECT(
    torch::Tensor tensor = torch::zeros(torch::IntArrayRef(dims, ndims), torch::ScalarType(type));
    if ((int64_t)element_size_in_bytes != tensor.element_size())
      throw std::invalid_argument("incoherent element sizes in bytes");
    void *tensor_data = tensor.data_ptr();
    memcpy(tensor_data, vs, tensor.numel() * element_size_in_bytes);
    return new torch::Tensor(tensor);
  )
  return nullptr;
}

void at_copy_data(tensor tensor, void *vs, int64_t numel, int elt_size_in_bytes) {
  PROTECT(
    if ((int64_t)elt_size_in_bytes != tensor->element_size())
      throw std::invalid_argument("incoherent element sizes in bytes");
    if ((int64_t)numel > tensor->numel())
      throw std::invalid_argument("target numel is larger than tensor numel");
    if (tensor->device().type() != at::kCPU) {
      torch::Tensor tmp_tensor = tensor->to(at::kCPU).contiguous();
      void *tensor_data = tmp_tensor.data_ptr();
      memcpy(vs, tensor_data, numel * elt_size_in_bytes);
    }
    else {
      auto tmp_tensor = tensor->contiguous();
      void *tensor_data = tmp_tensor.data_ptr();
      memcpy(vs, tensor_data, numel * elt_size_in_bytes);
    }
  )
}

tensor at_float_vec(double *vs, int len, int type) {
  PROTECT(
    torch::Tensor tensor = torch::empty({len}, torch::ScalarType(type));
    for (int i = 0; i < len; ++i) tensor[i] = vs[i];
    return new torch::Tensor(tensor);
  )
  return nullptr;
}

tensor at_int_vec(int64_t *vs, int len, int type) {
  PROTECT(
    torch::Tensor tensor = torch::empty({len}, torch::ScalarType(type));
    for (int i = 0; i < len; ++i) tensor[i] = vs[i];
    return new torch::Tensor(tensor);
  )
  return nullptr;
}

int at_defined(tensor t) {
  PROTECT(return t->defined();)
  return -1;
}

int at_is_sparse(tensor t) {
  PROTECT(return t->is_sparse();)
  return -1;
}

int at_dim(tensor t) {
  PROTECT(return t->dim();)
  return -1;
}

void at_shape(tensor t, int *dims) {
  PROTECT(
    int i = 0;
    for (int dim : t->sizes()) dims[i++] = dim;
  )
}

void at_stride(tensor t, int64_t *dims) {
  PROTECT(
    int i = 0;
    for (int64_t dim: t->strides()) dims[i++] = dim;
  )
}

int at_scalar_type(tensor t) {
  PROTECT(
    return static_cast<int>(t->scalar_type());
  )
}

void at_autocast_clear_cache() {
  at::autocast::clear_cache();
}

int at_autocast_decrement_nesting() {
  PROTECT(
    return at::autocast::decrement_nesting();
  )
  return -1;
}

int at_autocast_increment_nesting() {
  PROTECT(
    return at::autocast::increment_nesting();
  )
  return -1;
}

int at_autocast_is_enabled() {
  PROTECT(
    return at::autocast::is_enabled();
  )
  return -1;
}

int at_autocast_set_enabled(int b) {
  PROTECT(
    bool is_enabled = at::autocast::is_enabled();
    at::autocast::set_enabled(b);
    return is_enabled;
  )
  return -1;
}

int at_device(tensor tensor) {
  PROTECT (
    auto device = tensor->device();
    if (device.is_cpu()) return -1;
    return device.index();
  )
}

void at_backward(tensor t, int keep_graph, int create_graph) {
  PROTECT(t->backward({}, keep_graph, create_graph);)
}

int at_requires_grad(tensor t) {
  PROTECT(return t->requires_grad();)
  return -1;
}

int at_grad_set_enabled(int b) {
  PROTECT(
    bool is_enabled = torch::autograd::GradMode::is_enabled();
    torch::autograd::GradMode::set_enabled(b);
    return is_enabled;
  )
  return -1;
}

tensor at_get(tensor t, int index) {
  PROTECT(return new torch::Tensor((*t)[index]);)
  return nullptr;
}

template<typename T>
T at_value_at_indexes(tensor t, int *indexes, int indexes_len) {
  PROTECT(
    torch::Tensor tensor = *t;
    for (int i = 0; i < indexes_len; ++i) {
      tensor = tensor[indexes[i]];
    }
    return tensor.item<T>();
  )
  return T();
}

double at_double_value_at_indexes(tensor t, int *indexes, int indexes_len) {
  return at_value_at_indexes<double>(t, indexes, indexes_len);
}

int64_t at_int64_value_at_indexes(tensor t, int *indexes, int indexes_len) {
  return at_value_at_indexes<int64_t>(t, indexes, indexes_len);
}

template<typename T>
void at_set_value_at_indexes(tensor t, int *indexes, int indexes_len, T v) {
  PROTECT(
    torch::Tensor tensor = *t;
    for (int i = 0; i < indexes_len; ++i) {
      tensor = tensor[indexes[i]];
    }
    tensor.fill_(v);
  )
}

void at_set_double_value_at_indexes(tensor t, int *indexes, int indexes_len, double v) {
  at_set_value_at_indexes<double>(t, indexes, indexes_len, v);
}

void at_set_int64_value_at_indexes(tensor t, int *indexes, int indexes_len, int64_t v) {
  at_set_value_at_indexes<int64_t>(t, indexes, indexes_len, v);
}

void at_fill_double(tensor t, double v) {
  PROTECT(t->fill_(v);)
}

void at_fill_int64(tensor t, int64_t v) {
  PROTECT(t->fill_(v);)
}

void at_print(tensor t) {
  PROTECT(
    torch::Tensor *tensor = (torch::Tensor*)t;
    cout << *tensor << endl;
  )
}

char *at_to_string(tensor t, int line_size) {
  PROTECT(
    std::ostringstream oss;
    torch::print(oss, *t, line_size);
    return strdup(oss.str().c_str());
  )
  return nullptr;
}

void at_copy_(tensor dst, tensor src) {
  PROTECT(
    dst->copy_(*src);
  )
}

void at_save(tensor t, char *filename) {
  PROTECT(torch::save(*t, filename);)
}

void at_save_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename) {
  PROTECT(
    torch::serialize::OutputArchive archive;
    for (int i = 0; i < ntensors; ++i)
      archive.write(std::string(tensor_names[i]), *(tensors[i]), /* buffer=*/ false);
    archive.save_to(filename);
  )
}

void at_load_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename) {
  PROTECT(
    torch::serialize::InputArchive archive;
    archive.load_from(std::string(filename));
    vector<torch::Tensor> ts(ntensors);
    for (int i = 0; i < ntensors; ++i)
      archive.read(std::string(tensor_names[i]), ts[i]);
    // Only allocate the new tensor now so that if there is an exception raised during
    // [read], no memory has to be freed.
    for (int i = 0; i < ntensors; ++i)
      tensors[i] = new torch::Tensor(ts[i]);
  )
}

void at_loadz_callback(char *filename, void (*f)(char *, tensor)) {
  PROTECT(
    auto params = torch::jit::_load_parameters(filename);
    for (const auto &p : params) {
      f((char*)p.first.c_str(), new torch::Tensor(p.second));
    }
  )
}

void at_loadz_callback_with_device(char *filename, void (*f)(char *, tensor), int device_id) {
  PROTECT(
    auto params = torch::jit::_load_parameters(filename, device_of_int(device_id));
    for (const auto &p : params) {
      f((char*)p.first.c_str(), new torch::Tensor(p.second));
    }
  )
}

void at_load_callback(char *filename, void (*f)(char *, tensor)) {
  PROTECT(
    auto module = torch::jit::load(filename);
    for (const auto &p : module.named_parameters()) {
      auto v = p.value;
      f((char*)p.name.c_str(), new torch::Tensor(v));
    }
  )
}

void at_load_multi_(tensor *tensors, char **tensor_names, int ntensors, char *filename) {
  PROTECT(
    torch::NoGradGuard no_grad;
    torch::serialize::InputArchive archive;
    archive.load_from(std::string(filename));
    for (int i = 0; i < ntensors; ++i) {
      if (tensors[i]->device().type() == at::kCPU)
        archive.read(std::string(tensor_names[i]), *(tensors[i]));
      else {
        torch::Tensor tmp_tensor = torch::empty_like(*(tensors[i]), at::device(at::kCPU));
        archive.read(std::string(tensor_names[i]), tmp_tensor);
        tensors[i]->copy_(tmp_tensor);
      }
    }
  )
}

tensor at_load(char *filename) {
  PROTECT(
    torch::Tensor tensor;
    torch::load(tensor, filename);
    return new torch::Tensor(tensor);
  )
  return nullptr;
}

int at_get_num_interop_threads() {
  PROTECT(return at::get_num_interop_threads();)
  return -1;
}

int at_get_num_threads() {
  PROTECT(return at::get_num_threads();)
  return -1;
}

void at_set_num_interop_threads(int n_threads) {
  PROTECT(at::set_num_interop_threads(n_threads);)
}

void at_set_num_threads(int n_threads) {
  PROTECT(at::set_num_threads(n_threads);)
}

void at_free(tensor t) {
  delete(t);
}

void at_run_backward(tensor *tensors,
                     int ntensors,
                     tensor *inputs,
                     int ninputs,
                     tensor *outputs,
                     int keep_graph,
                     int create_graph) {
  PROTECT(
    vector<torch::autograd::Edge> roots;
    for (int i = 0; i < ntensors; ++i)
      roots.push_back(torch::autograd::impl::gradient_edge(*tensors[i]));

    vector<torch::autograd::Edge> inputs_;
    for (int i = 0; i < ninputs; ++i) {
      if (!inputs[i]->requires_grad())
        throw std::invalid_argument("one of the input tensor does not use set_requires_grad");
      inputs_.push_back(torch::autograd::impl::gradient_edge(*inputs[i]));
    }

    vector<torch::autograd::Variable> grads;
    for (int i = 0; i < ntensors; ++i)
      grads.push_back(torch::ones_like(*tensors[i]));

    auto vl = torch::autograd::Engine::get_default_engine().execute(roots, grads, keep_graph, create_graph, false, inputs_);
    for (int i = 0; i < ninputs; ++i) {
      outputs[i] = static_cast<tensor>(new torch::autograd::Variable(vl[i]));
    }
  )
}

optimizer ato_adam(double learning_rate,
                   double beta1,
                   double beta2,
                   double weight_decay,
                   double eps) {
  PROTECT(
    auto options =
      torch::optim::AdamOptions(learning_rate)
        .betas(std::tuple<double, double>(beta1, beta2))
        .weight_decay(weight_decay)
        .eps(eps);
    return new torch::optim::Adam(vector<torch::Tensor>(), options);
  )
  return nullptr;
}

optimizer ato_rmsprop(double learning_rate,
                      double alpha,
                      double eps,
                      double weight_decay,
                      double momentum,
                      int centered) {
  PROTECT(
    auto options =
      torch::optim::RMSpropOptions(learning_rate)
        .alpha(alpha)
        .eps(eps)
        .weight_decay(weight_decay)
        .momentum(momentum)
        .centered(centered != 0);
      return new torch::optim::RMSprop(vector<torch::Tensor>(), options);
    )
  return nullptr;
}

optimizer ato_sgd(double learning_rate,
                  double momentum,
                  double dampening,
                  double weight_decay,
                  int nesterov) {
  PROTECT(
    auto options =
      torch::optim::SGDOptions(learning_rate)
      .momentum(momentum)
      .dampening(dampening)
      .weight_decay(weight_decay)
      .nesterov(nesterov);
    return new torch::optim::SGD(vector<torch::Tensor>(), options);
  )
  return nullptr;
}

void ato_add_parameters(optimizer t, tensor *tensors, int ntensors) {
  PROTECT(
    for (int i = 0; i < ntensors; ++i)
      t->param_groups()[0].params().push_back(*(tensors[i]));
  )
}

template <class T>
void set_lr(optimizer t, double learning_rate) {
  torch::optim::OptimizerOptions* d = &(t->defaults());
  if (auto p = dynamic_cast<T*>(d)) {
    p->lr(learning_rate);
    for (auto &param_group: t->param_groups()) {
      torch::optim::OptimizerOptions* d = &(param_group.options());
      if (auto p2 = dynamic_cast<T*>(d)) {
        p2->lr(learning_rate);
      }
      else throw std::invalid_argument("unexpected param group type");
    }
  }
}

void ato_set_learning_rate(optimizer t, double learning_rate) {
  PROTECT(
    set_lr<torch::optim::AdamOptions>(t, learning_rate);
    set_lr<torch::optim::AdamWOptions>(t, learning_rate);
    set_lr<torch::optim::RMSpropOptions>(t, learning_rate);
    set_lr<torch::optim::SGDOptions>(t, learning_rate);
  )
}

template <class T>
void set_lr_group(optimizer t, size_t group, double learning_rate) {
  auto &param_group = t->param_groups().at(group);
  torch::optim::OptimizerOptions* d = &(param_group.options());
  if (auto p = dynamic_cast<T*>(d)) {
    p->lr(learning_rate);
  }
}

void ato_set_learning_rate_group(optimizer t, size_t group, double learning_rate) {
  PROTECT(
    set_lr_group<torch::optim::AdamOptions>(t, group, learning_rate);
    set_lr_group<torch::optim::AdamWOptions>(t, group, learning_rate);
    set_lr_group<torch::optim::RMSpropOptions>(t, group, learning_rate);
    set_lr_group<torch::optim::SGDOptions>(t, group, learning_rate);
  )
}

void ato_set_momentum(optimizer t, double momentum) {
  PROTECT(
    torch::optim::OptimizerOptions* d = &(t->defaults());
    if (auto adam = dynamic_cast<torch::optim::AdamOptions*>(d)) {
      auto betas = adam->betas();
      adam->betas(std::tuple<double, double>(momentum, get<1>(betas)));
      for (auto &param_group: t->param_groups()) {
          torch::optim::OptimizerOptions* d = &(param_group.options());
          if (auto adam2 = dynamic_cast<torch::optim::AdamOptions*>(d)) {
              adam2->betas(std::tuple<double, double>(momentum, get<1>(betas)));
          }
          else throw std::invalid_argument("unexpected param group type");
      }
    }
    else if (auto adamw = dynamic_cast<torch::optim::AdamWOptions*>(d)) {
        auto betas = adamw->betas();
        adamw->betas(std::tuple<double, double>(momentum, get<1>(betas)));
        for (auto &param_group: t->param_groups()) {
            torch::optim::OptimizerOptions* d = &(param_group.options());
            if (auto adamw2 = dynamic_cast<torch::optim::AdamWOptions*>(d)) {
                adamw2->betas(std::tuple<double, double>(momentum, get<1>(betas)));
            }
            else throw std::invalid_argument("unexpected param group type");
        }
    }
    else if (auto rms = dynamic_cast<torch::optim::RMSpropOptions*>(d)) {
      rms->momentum(momentum);
      for (auto &param_group: t->param_groups()) {
          torch::optim::OptimizerOptions* d = &(param_group.options());
          if (auto rms2 = dynamic_cast<torch::optim::RMSpropOptions*>(d)) {
              rms2->momentum(momentum);
          }
          else throw std::invalid_argument("unexpected param group type");
      }
    }
    else if (auto sgd = dynamic_cast<torch::optim::SGDOptions*>(d)) {
      sgd->momentum(momentum);
      for (auto &param_group: t->param_groups()) {
          torch::optim::OptimizerOptions* d = &(param_group.options());
          if (auto sgd2 = dynamic_cast<torch::optim::SGDOptions*>(d)) {
              sgd2->momentum(momentum);
          }
          else throw std::invalid_argument("unexpected param group type");
      }
    }
    else
     throw std::invalid_argument("unexpected optimizer");
  )
}

void ato_set_momentum_group(optimizer t, size_t group, double momentum) {
  PROTECT(
    auto &param_group = t->param_groups().at(group);
    torch::optim::OptimizerOptions* d = &(param_group.options());

    if (auto adam = dynamic_cast<torch::optim::AdamOptions*>(d)) {
        auto betas = adam->betas();
        adam->betas(std::tuple<double, double>(momentum, get<1>(betas)));
    }
    else if (auto adamw = dynamic_cast<torch::optim::AdamWOptions*>(d)) {
        auto betas = adamw->betas();
        adamw->betas(std::tuple<double, double>(momentum, get<1>(betas)));
    }
    else if (auto rms = dynamic_cast<torch::optim::RMSpropOptions*>(d)) {
        rms->momentum(momentum);
    }
    if (auto sgd = dynamic_cast<torch::optim::SGDOptions*>(d)) {
        sgd->momentum(momentum);
    }
    else
        throw std::invalid_argument("unexpected optimizer");
  )
}

template <class T>
void set_weight_decay(optimizer t, double weight_decay) {
  torch::optim::OptimizerOptions* d = &(t->defaults());
  if (auto p = dynamic_cast<T*>(d)) {
    p->weight_decay(weight_decay);
    for (auto &param_group: t->param_groups()) {
      torch::optim::OptimizerOptions* d = &(param_group.options());
      if (auto p2 = dynamic_cast<T*>(d)) {
        p2->weight_decay(weight_decay);
      }
      else throw std::invalid_argument("unexpected param group type");
    }
  }
}

void ato_set_weight_decay(optimizer t, double weight_decay) {
  PROTECT(
    set_weight_decay<torch::optim::AdamOptions>(t, weight_decay);
    set_weight_decay<torch::optim::AdamWOptions>(t, weight_decay);
    set_weight_decay<torch::optim::RMSpropOptions>(t, weight_decay);
    set_weight_decay<torch::optim::SGDOptions>(t, weight_decay);
  )
}

template <class T>
void set_weight_decay_group(optimizer t, size_t group, double weight_decay) {
  auto &param_group = t->param_groups().at(group);
  torch::optim::OptimizerOptions* d = &(param_group.options());
  if (auto p = dynamic_cast<T*>(d)) {
    p->weight_decay(weight_decay);
  }
}

void ato_set_weight_decay_group(optimizer t, size_t group, double weight_decay) {
  PROTECT(
    set_weight_decay_group<torch::optim::AdamOptions>(t, group, weight_decay);
    set_weight_decay_group<torch::optim::AdamWOptions>(t, group, weight_decay);
    set_weight_decay_group<torch::optim::RMSpropOptions>(t, group, weight_decay);
    set_weight_decay_group<torch::optim::SGDOptions>(t, group, weight_decay);
  )
}

void ato_zero_grad(optimizer t) {
  PROTECT(t->zero_grad();)
}

void ato_step(optimizer t) {
  PROTECT(t->step();)
}

void ato_free(optimizer t) {
  delete(t);
}

scalar ats_int(int64_t v) {
  PROTECT(return new torch::Scalar(v);)
  return nullptr;
}

scalar ats_float(double v) {
  PROTECT(return new torch::Scalar(v);)
  return nullptr;
}

int64_t ats_to_int(scalar s) {
  PROTECT(return s->toLong();)
  return -1;
}

double ats_to_float(scalar s) {
  PROTECT(return s->toDouble();)
  return 0.;
}

char *ats_to_string(scalar s) {
  PROTECT(
    using namespace at;
    std::ostringstream oss;
    oss << (*s);
    return strdup(oss.str().c_str());
  )
  return nullptr;
}

void ats_free(scalar s) {
  delete(s);
}

int atc_cuda_device_count() {
  PROTECT(return torch::cuda::device_count();)
  return -1;
}

int atc_cuda_is_available() {
  PROTECT(return torch::cuda::is_available();)
  return -1;
}

int atc_cudnn_is_available() {
  PROTECT(return torch::cuda::cudnn_is_available();)
  return -1;
}

void atc_set_benchmark_cudnn(int b) {
  at::globalContext().setBenchmarkCuDNN(b);
}

module atm_load(char *filename) {
  PROTECT(
    return new torch::jit::script::Module(torch::jit::load(filename));
  )
  return nullptr;
}

module atm_load_str(char *data, size_t sz) {
  PROTECT(
    std::istringstream stream(std::string(data, sz));
    return new torch::jit::script::Module(torch::jit::load(stream));
  )
  return nullptr;
}

tensor atm_forward(module m, tensor *tensors, int ntensors) {
  PROTECT(
    std::vector<torch::jit::IValue> inputs;
    for (int i = 0; i < ntensors; ++i)
      inputs.push_back(*(tensors[i]));
    torch::jit::IValue output = m->forward(std::move(inputs));
    if (!output.isTensor())
      throw std::invalid_argument("forward did not return a tensor");
    return new torch::Tensor(output.toTensor());
  )
  return nullptr;
}

ivalue atm_forward_(module m,
                    ivalue *ivalues,
                    int nivalues) {
  PROTECT(
    std::vector<torch::jit::IValue> inputs;
    for (int i = 0; i < nivalues; ++i)
      inputs.push_back(*(ivalues[i]));
    torch::jit::IValue output = m->forward(inputs);
    return new torch::jit::IValue(output);
  )
  return nullptr;
}

void atm_free(module m) {
  delete(m);
}

void atm_to(module m, int device, int dtype, bool non_blocking) {
  PROTECT(
    m->to(device_of_int(device), at::ScalarType(dtype), non_blocking);
  )
}

ivalue ati_tensor(tensor t) {
  PROTECT(
    return new torch::jit::IValue(*t);
  )
  return nullptr;
}

ivalue ati_int(int64_t i) {
  PROTECT(
    return new torch::jit::IValue(i);
  )
  return nullptr;
}

ivalue ati_double(double d) {
  PROTECT(
    return new torch::jit::IValue(d);
  )
  return nullptr;
}

ivalue ati_bool(int i) {
  PROTECT(
    return new torch::jit::IValue((bool)i);
  )
  return nullptr;
}

ivalue ati_string(char *s) {
  PROTECT(
    string str(s);
    return new torch::jit::IValue(str);
  )
  return nullptr;
}

ivalue ati_none() {
  PROTECT(
    return new torch::jit::IValue();
  )
  return nullptr;
}

ivalue ati_tuple(ivalue *is, int nvalues) {
  PROTECT(
    vector<torch::jit::IValue> vec;
    for (int i = 0; i < nvalues; ++i) vec.push_back(*(is[i]));
    return new torch::jit::IValue(torch::ivalue::Tuple::create(vec));
  )
  return nullptr;
}

ivalue ati_generic_list(ivalue *is, int nvalues) {
  PROTECT(
    c10::List<torch::jit::IValue> vec(c10::AnyType::get());
    for (int i = 0; i < nvalues; ++i) vec.push_back(*(is[i]));
    return new torch::jit::IValue(c10::List<torch::jit::IValue>(vec));
  )
  return nullptr;
}

ivalue ati_generic_dict(ivalue *is, int nvalues) {
  c10::Dict<torch::jit::IValue, torch::jit::IValue> dict(c10::AnyType::get(), c10::AnyType::get());
  PROTECT(
    for (int i = 0; i < nvalues; ++i) dict.insert(*(is[2*i]), *(is[2*i+1]));
    return new torch::jit::IValue(dict);
  )
  return nullptr;
}

ivalue ati_int_list(int64_t *is, int nvalues) {
  PROTECT(
    c10::List<int64_t> vec;
    for (int i = 0; i < nvalues; ++i) vec.push_back(is[i]);
    return new torch::jit::IValue(vec);
  )
  return nullptr;
}

ivalue ati_double_list(double *is, int nvalues) {
  PROTECT(
    c10::List<double> vec;
    for (int i = 0; i < nvalues; ++i) vec.push_back(is[i]);
    return new torch::jit::IValue(vec);
  )
  return nullptr;
}

ivalue ati_bool_list(char *is, int nvalues) {
  PROTECT(
    c10::List<bool> vec;
    for (int i = 0; i < nvalues; ++i) vec.push_back(is[i] != 0);
    return new torch::jit::IValue(vec);
  )
  return nullptr;
}

ivalue ati_string_list(char **is, int nvalues) {
  PROTECT(
    c10::List<string> vec;
    for (int i = 0; i < nvalues; ++i) vec.push_back(string(is[i]));
    return new torch::jit::IValue(vec);
  )
  return nullptr;
}

ivalue ati_tensor_list(tensor *is, int nvalues) {
  PROTECT(
    c10::List<at::Tensor> vec;
    for (int i = 0; i < nvalues; ++i) vec.push_back(*(is[i]));
    return new torch::jit::IValue(vec);
  )
  return nullptr;
}

int ati_tag(ivalue i) {
  PROTECT(
    if (i->isNone()) return 0;
    else if (i->isTensor()) return 1;
    else if (i->isDouble()) return 2;
    else if (i->isInt()) return 3;
    else if (i->isBool()) return 4;
    else if (i->isTuple()) return 5;
    else if (i->isIntList()) return 6;
    else if (i->isDoubleList()) return 7;
    else if (i->isBoolList()) return 8;
    else if (i->isString()) return 9;
    else if (i->isTensorList()) return 10;
    else if (i->isList()) return 12;
    else if (i->isGenericDict()) return 13;
    else if (i->isObject()) return 14;
    throw std::invalid_argument(("unsupported tag " + i->tagKind()).c_str());
    return -1;
  )
  return -1;
}

int64_t ati_to_int(ivalue i) {
  PROTECT(
    return i->toInt();
  )
  return -1;
}

double ati_to_double(ivalue i) {
  PROTECT(
    return i->toDouble();
  )
  return 0;
}

int ati_to_bool(ivalue i) {
  PROTECT(
    return i->toBool();
  )
  return -1;
}

char *ati_to_string(ivalue i) {
  PROTECT(
    auto str = i->toStringRef();
    return strdup(str.c_str());
  )
  return nullptr;
}

tensor ati_to_tensor(ivalue i) {
  PROTECT(
    return new torch::Tensor(i->toTensor());
  )
  return nullptr;
}

int ati_length(ivalue i) {
  PROTECT(
    if (i->isTuple()) return i->toTuple()->elements().size();
    else if (i->isIntList()) return i->toIntList().size();
    else if (i->isDoubleList()) return i->toDoubleList().size();
    else if (i->isBoolList()) return i->toBoolList().size();
    else if (i->isString()) return i->toStringRef().size();
    else if (i->isTensorList()) return i->toTensorList().size();
    else if (i->isList()) return i->toList().size();
    else if (i->isGenericDict()) return i->toGenericDict().size();
    throw std::invalid_argument(("unsupported tag for length " + i->tagKind()).c_str());
    return -1;
  )
  return -1;
}

int ati_tuple_length(ivalue i) {
  PROTECT(
    return i->toTuple()->elements().size();
  )
  return -1;
}

void ati_to_tuple(ivalue i,
                  ivalue *outputs,
                  int noutputs) {
  PROTECT(
    auto vec = i->toTuple()->elements();
    if (vec.size() != noutputs) {
      throw std::invalid_argument("unexpected tuple size");
    }
    for (int i = 0; i < noutputs; ++i)
      outputs[i] = new torch::jit::IValue(vec[i]);
  )
}

void ati_to_generic_list(ivalue i,
                         ivalue *outputs,
                         int noutputs) {
  PROTECT(
    auto vec = i->toList();
    if (vec.size() != noutputs) {
      throw std::invalid_argument("unexpected list size");
    }
    for (int i = 0; i < noutputs; ++i)
      outputs[i] = new torch::jit::IValue(vec[i]);
  )
}

void ati_to_generic_dict(ivalue i,
                         ivalue *outputs,
                         int noutputs) {
  PROTECT(
    auto dict = i->toGenericDict();
    if (dict.size() != noutputs) {
      throw std::invalid_argument("unexpected dict size");
    }
    int k = 0;
    for (auto it = dict.begin(); it != dict.end(); ++it) {
      outputs[k++] = new torch::jit::IValue(it->key());
      outputs[k++] = new torch::jit::IValue(it->value());
    }
  )
}

void ati_to_int_list(ivalue i,
                  int64_t *outputs,
                  int noutputs) {
  PROTECT(
    auto vec = i->toIntList();
    if (vec.size() != noutputs) {
      throw std::invalid_argument("unexpected list<int> size");
    }
    for (int i = 0; i < noutputs; ++i)
      outputs[i] = vec[i];
  )
}

void ati_to_double_list(ivalue i,
                        double *outputs,
                        int noutputs) {
  PROTECT(
    auto vec = i->toDoubleList();
    if (vec.size() != noutputs) {
      throw std::invalid_argument("unexpected list<double> size");
    }
    for (int i = 0; i < noutputs; ++i)
      outputs[i] = vec[i];
  )
}

void ati_to_bool_list(ivalue i,
                      char *outputs,
                      int noutputs) {
  PROTECT(
    auto vec = i->toBoolList();
    if (vec.size() != noutputs) {
      throw std::invalid_argument("unexpected list<bool> size");
    }
    for (int i = 0; i < noutputs; ++i)
      outputs[i] = vec[i];
  )
}

void ati_to_tensor_list(ivalue i,
                        tensor *outputs,
                        int noutputs) {
  PROTECT(
    auto vec = i->toTensorList();
    if (vec.size() != noutputs) {
      throw std::invalid_argument("unexpected list<tensor> size");
    }
    for (int i = 0; i < noutputs; ++i)
      outputs[i] = new torch::Tensor(vec[i]);
  )
}

ivalue ati_object_method_(ivalue i, char *method_name, ivalue *ivalues, int nivalues) {
  PROTECT(
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(*i); // self parameter
    for (int j = 0; j < nivalues; ++j)
      inputs.push_back(*(ivalues[j]));
    torch::jit::IValue output = i->toObjectRef().type()->getMethod(method_name)(std::move(inputs));
    return new torch::jit::IValue(output);
  )
  return nullptr;
}

ivalue ati_object_getattr_(ivalue i, char *attr_name) {
  PROTECT(
    torch::jit::IValue output  = i->toObjectRef().getAttr(attr_name);
    return new torch::jit::IValue(output);
  )
  return nullptr;
}

ivalue ati_clone(ivalue i) {
  PROTECT(
    return new torch::jit::IValue(*i);
  )
  return nullptr;
}

void ati_free(ivalue i) {
  delete(i);
}

void at_set_graph_executor_optimize(bool o) {
  torch::jit::setGraphExecutorOptimize(o);
}

#include "torch_api_generated.cpp.h"

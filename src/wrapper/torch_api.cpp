#include<torch/csrc/autograd/engine.h>
#include<torch/torch.h>
#include<torch/script.h>
#include<vector>
#include<caml/fail.h>
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

tensor at_new_tensor() {
  PROTECT(
    return new torch::Tensor();
  )
  return nullptr;
}

tensor at_tensor_of_data(void *vs, int64_t *dims, int ndims, int element_size_in_bytes, int type) {
  PROTECT(
    torch::Tensor tensor = torch::zeros(torch::IntArrayRef(dims, ndims), torch::ScalarType(type));
    if (element_size_in_bytes != tensor.element_size())
      caml_failwith("incoherent element sizes in bytes");
    void *tensor_data = tensor.data_ptr();
    memcpy(tensor_data, vs, tensor.numel() * element_size_in_bytes);
    return new torch::Tensor(tensor);
  )
  return nullptr;
}

void at_copy_data(tensor tensor, void *vs, int64_t numel, int elt_size_in_bytes) {
  PROTECT(
    if (elt_size_in_bytes != tensor->element_size())
      caml_failwith("incoherent element sizes in bytes");
    if (numel != tensor->numel())
      caml_failwith("incoherent number of elements");
    if (tensor->device().type() != at::kCPU) {
      torch::Tensor tmp_tensor = tensor->to(at::kCPU).contiguous();
      void *tensor_data = tmp_tensor.data_ptr();
      memcpy(vs, tensor_data, numel * elt_size_in_bytes);
    }
    else {
      torch::Tensor tmp_tensor = tensor->contiguous();
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

int at_scalar_type(tensor t) {
  PROTECT(
    return static_cast<int>(t->scalar_type());
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
    torch::autograd::Engine engine;
    vector<torch::autograd::Edge> roots;
    for (int i = 0; i < ntensors; ++i)
      roots.push_back(torch::autograd::impl::gradient_edge(torch::autograd::as_variable_ref(*tensors[i])));

    vector<torch::autograd::Edge> inputs_;
    for (int i = 0; i < ninputs; ++i) {
      if (!inputs[i]->requires_grad())
        caml_invalid_argument("one of the input tensor does not use set_requires_grad");
      inputs_.push_back(torch::autograd::impl::gradient_edge(torch::autograd::as_variable_ref(*inputs[i])));
    }

    vector<torch::autograd::Variable> grads;
    for (int i = 0; i < ntensors; ++i)
      grads.push_back(torch::ones_like(*tensors[i]));

    auto vl = torch::autograd::Engine::get_default_engine().execute(roots, grads, keep_graph, create_graph, inputs_);
    for (int i = 0; i < ninputs; ++i) {
      outputs[i] = static_cast<tensor>(new torch::autograd::Variable(vl[i]));
    }
  )
}

optimizer ato_adam(double learning_rate,
                   double beta1,
                   double beta2,
                   double weight_decay) {
  PROTECT(
    auto options =
      torch::optim::AdamOptions(learning_rate)
        .beta1(beta1)
        .beta2(beta2)
        .weight_decay(weight_decay);
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
    t->add_parameters(of_carray_tensor(tensors, ntensors));
  )
}

void ato_set_learning_rate(optimizer t, double learning_rate) {
  PROTECT(
    if (auto adam = dynamic_cast<torch::optim::Adam*>(t))
      adam->options.learning_rate(learning_rate);
    else if (auto rms = dynamic_cast<torch::optim::RMSprop*>(t))
      rms->options.learning_rate(learning_rate);
    else if (auto sgd = dynamic_cast<torch::optim::SGD*>(t))
      sgd->options.learning_rate(learning_rate);
    else
     caml_invalid_argument("unexpected optimizer");
  )
}

void ato_set_momentum(optimizer t, double momentum) {
  PROTECT(
    if (auto adam = dynamic_cast<torch::optim::Adam*>(t))
      adam->options.beta1(momentum);
    else if (auto rms = dynamic_cast<torch::optim::RMSprop*>(t))
      rms->options.momentum(momentum);
    else if (auto sgd = dynamic_cast<torch::optim::SGD*>(t))
      sgd->options.momentum(momentum);
    else
     caml_invalid_argument("unexpected optimizer");
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

tensor atm_forward(module m, tensor *tensors, int ntensors) {
  PROTECT(
    std::vector<torch::jit::IValue> inputs;
    for (int i = 0; i < ntensors; ++i)
      inputs.push_back(*(tensors[i]));
    torch::jit::IValue output = m->forward(inputs);
    if (!output.isTensor())
      caml_failwith("forward did not return a tensor");
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

ivalue ati_tuple(ivalue *is, int nvalues) {
  PROTECT(
    vector<torch::jit::IValue> vec;
    for (int i = 0; i < nvalues; ++i) vec.push_back(*(is[i]));
    return new torch::jit::IValue(torch::ivalue::Tuple::create(vec));
  )
  return nullptr;
}

int ati_tag(ivalue i) {
  PROTECT(
    if (i->isTensor()) return 0;
    else if (i->isInt()) return 1;
    else if (i->isDouble()) return 2;
    else if (i->isTuple()) return 3;
    caml_failwith(("unsupported tag" + i->tagKind()).c_str());
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
  return 0.;
}

tensor ati_to_tensor(ivalue i) {
  PROTECT(
    return new torch::Tensor(i->toTensor());
  )
  return nullptr;
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
      caml_failwith("unexpected tuple size");
    }
    for (int i = 0; i < noutputs; ++i)
      outputs[i] = new torch::jit::IValue(vec[i]);
  )
}


void ati_free(ivalue i) {
  delete(i);
}

at::Device device_of_int(int d) {
    if (d < 0) return at::Device(at::kCPU);
    return at::Device(at::kCUDA, /*index=*/d);
}

#include "torch_api_generated.cpp.h"

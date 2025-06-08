#include <torch/extension.h>

std::string hello() {
    return "Hello World!";
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("hello", torch::wrap_pybind_function(hello), "hello");
}
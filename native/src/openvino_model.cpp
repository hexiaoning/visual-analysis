#include "openvino_model.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string_view>

#include <openvino/openvino.hpp>

namespace visual {
namespace {

ov::Core& core() {
    static ov::Core runtime;
    return runtime;
}

ov::element::Type inputType(ONNXTensorElementDataType element) {
    if(element==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return ov::element::f32;
    if(element==ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) return ov::element::i64;
    throw std::runtime_error("Unsupported OpenVINO input tensor type.");
}

struct Ports {
    std::vector<std::string> names;
    std::vector<size_t> indices;
};

Ports bindPorts(const std::vector<ov::Output<const ov::Node>>& ports) {
    Ports result;
    for(const auto& port:ports) result.names.push_back(port.get_any_name());
    result.indices.resize(ports.size());
    std::iota(result.indices.begin(),result.indices.end(),size_t{});
    return result;
}

}

struct OpenVinoModel::State {
    explicit State(const std::filesystem::path& path)
        : model(compile(path)),
          request(model.create_infer_request()) {
        const auto devices=model.get_property(ov::execution_devices);
        if(devices.empty() || !std::ranges::all_of(devices,[](std::string_view name){return name.starts_with("GPU");})) {
            throw std::runtime_error("OpenVINO did not select an Intel GPU.");
        }
        inputs=bindPorts(model.inputs());
        outputs=bindPorts(model.outputs());
    }

    static ov::CompiledModel compile(const std::filesystem::path& path) {
        const auto latency=ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY);
        if(path.stem()==L"rtdetr_v2_r50vd") {
            auto model=core().read_model(path.wstring());
            model->reshape(ov::PartialShape{1,3,640,640});
            return core().compile_model(model,"GPU",latency,ov::hint::inference_precision(ov::element::f32));
        }
        return core().compile_model(path.wstring(),"GPU",latency);
    }

    ov::CompiledModel model;
    ov::InferRequest request;
    Ports inputs,outputs;
};

OpenVinoModel::OpenVinoModel(const std::filesystem::path& path) : state_(std::make_unique<State>(path)) {}
OpenVinoModel::~OpenVinoModel()=default;

std::vector<std::string> OpenVinoModel::inputNames() const { return state_->inputs.names; }
std::vector<std::string> OpenVinoModel::outputNames() const { return state_->outputs.names; }

std::vector<int64_t> OpenVinoModel::inputShape(size_t index) const {
    const auto shape=state_->model.input(state_->inputs.indices.at(index)).get_partial_shape();
    if(shape.rank().is_dynamic()) return {};
    std::vector<int64_t> result;
    for(int64_t dimension=0;dimension<shape.rank().get_length();++dimension) {
        const auto& item=shape[dimension];
        result.push_back(item.is_dynamic()?-1:item.get_length());
    }
    return result;
}

std::vector<Ort::Value> OpenVinoModel::run(std::span<const Ort::Value> inputs) {
    for(size_t index=0;index<inputs.size();++index) {
        const auto properties=inputs[index].GetTensorTypeAndShapeInfo();
        const auto dimensions=properties.GetShape();
        const ov::Shape shape(dimensions.begin(),dimensions.end());
        state_->request.set_input_tensor(state_->inputs.indices.at(index),ov::Tensor(inputType(properties.GetElementType()),shape,
            const_cast<void*>(inputs[index].GetTensorRawData())));
    }
    state_->request.infer();
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<Ort::Value> outputs;
    for(const auto index:state_->outputs.indices) {
        const auto tensor=state_->request.get_output_tensor(index);
        if(tensor.get_element_type()!=ov::element::f32) throw std::runtime_error("Expected an FP32 OpenVINO result.");
        const auto shape=tensor.get_shape();
        const std::vector<int64_t> dimensions(shape.begin(),shape.end());
        auto output=Ort::Value::CreateTensor(allocator,dimensions.data(),dimensions.size(),
                                              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        std::memcpy(output.GetTensorMutableRawData(),tensor.data<const float>(),tensor.get_byte_size());
        outputs.push_back(std::move(output));
    }
    return outputs;
}

}

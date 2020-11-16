#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("DummyPairwiseHamming")
	.Input("x: float32")
	.Output("a: int32")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
   c->set_output(0, c->input(0));
   //c->set_output(1, c->input(1));
   //c->set_output(2, c->input(1));
   return Status::OK();
});

class PairwiseHammingOp : public OpKernel {
 public:
  explicit PairwiseHammingOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float_t>();

    std::cout << "dick" << std::endl;
    // Create an output tensor
    Tensor* output_tensor = NULL;
    int b = input_tensor.shape().dim_size(0);
    int n = input_tensor.shape().dim_size(1);

    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,n,n},
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output_flat(0) = input(0);
  }
};

REGISTER_KERNEL_BUILDER(Name("DummyPairwiseHamming").Device(DEVICE_CPU), PairwiseHammingOp);


REGISTER_OP("BPairwiseHammingOp")
	.Input("x: float32")
	.Output("a: int32")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
   c->set_output(0, c->input(0));
   //c->set_output(1, c->input(1));
   //c->set_output(2, c->input(1));
   return Status::OK();
});

class BPairwiseHammingOp : public OpKernel {
 public:
  explicit BPairwiseHammingOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float_t>();

    std::cout << "dick" << std::endl;
    // Create an output tensor
    Tensor* output_tensor = NULL;
    int b = input_tensor.shape().dim_size(0);
    int n = input_tensor.shape().dim_size(1);

    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,n,n},
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output_flat(0) = input(0);
  }
};

REGISTER_KERNEL_BUILDER(Name("BPairwiseHammingOp").Device(DEVICE_CPU), BPairwiseHammingOp);

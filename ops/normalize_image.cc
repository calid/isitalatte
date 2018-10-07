#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("NormalizeImage")
    .Input("imagedata: float")
    .Output("normalized: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class NormalizeImageOp : public OpKernel {
    public:
        explicit NormalizeImageOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            const Tensor& input_tensor = context->input(0);
            auto input = input_tensor.flat<float>();

            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                        &output_tensor));

            auto output_flat = output_tensor->flat<float>();

            const int N = input.size();
            for (int i = 0; i < N; i++) {
                output_flat(i) = input(i) / 255;
            }
        }
};

REGISTER_KERNEL_BUILDER(Name("NormalizeImage").Device(DEVICE_CPU), NormalizeImageOp);

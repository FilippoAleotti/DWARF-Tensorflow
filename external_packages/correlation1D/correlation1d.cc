#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("Correlation1d")
    .Input("input_a: float32")
    .Input("input_b: float32")
    .Attr("kernel_size: int")
    .Attr("max_displacement: int")
    .Attr("stride_1: int")
    .Attr("stride_2: int")
    .Attr("pad: int")
    .Output("output: float32")
    .SetShapeFn([](InferenceContext *c) {
      ShapeHandle input_a, input_b, input;

      // Get shapes of both inputs and verify they are rank 4
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_a));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &input_b));

      // Verify inputs are same dimensions
      TF_RETURN_IF_ERROR(c->Merge(input_a, input_b, &input));

      // Get the attributes
      int kernel_size, max_displacement, stride_1, stride_2, pad;
      TF_RETURN_IF_ERROR(c->GetAttr("kernel_size", &kernel_size));
      TF_RETURN_IF_ERROR(c->GetAttr("max_displacement", &max_displacement));
      TF_RETURN_IF_ERROR(c->GetAttr("stride_1", &stride_1));
      TF_RETURN_IF_ERROR(c->GetAttr("stride_2", &stride_2));
      TF_RETURN_IF_ERROR(c->GetAttr("pad", &pad));

      // Get dimensions of input (already padded)
      int64 batch = c->Value(c->Dim(input, 0));
      int64 input_height = c->Value(c->Dim(input, 1));
      int64 input_width = c->Value(c->Dim(input, 2));
      int64 padded_width = input_width + 2 * pad;
      int64 padded_height = input_height;

      // The size of unreachable border region on each side
      int kernel_radius = (kernel_size - 1) / 2;
      int border_size = max_displacement + kernel_radius;

      // Calculate the output dimensions
      int64 output_height = padded_height;
      int64 output_width = (int64)ceil((float)(padded_width - border_size * 2) / (float)stride_1);

      // TODO: Verify output size >= 1

      int neighborhood_grid_radius = max_displacement / stride_2;
      int neighborhood_grid_width = neighborhood_grid_radius * 2 + 1;
      int64 output_channels = neighborhood_grid_width;

      // Set output shape
      c->set_output(0, c->MakeShape({batch, output_height, output_width, output_channels}));
      return Status::OK();
    });

REGISTER_OP("Correlation1dGrad")
    .Input("gradients: float32")
    .Input("input_a: float32")
    .Input("input_b: float32")
    .Attr("kernel_size: int")
    .Attr("max_displacement: int")
    .Attr("stride_1: int")
    .Attr("stride_2: int")
    .Attr("pad: int")
    .Output("backpros_a: float32")
    .Output("backpros_b: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Merge(c->input(1), c->input(2), &out));
      c->set_output(0, out);
      c->set_output(1, out);
      return Status::OK();
    });

void Correlation1d(const GPUDevice& device,
                 const float     *input_a,
                 const float     *input_b,
                 const int        batch_size,
                 const int        out_height,
                 const int        out_width,
                 const int        out_channels,
                 const int        out_count,
                 const int        in_height_padded,
                 const int        in_width_padded,
                 const int        in_channels,
                 int              max_displacement,
                 int              x_shift,
                 int              neighborhood_grid_width,
                 int              kernel_radius,
                 int              kernel_size,
                 int              stride_1,
                 int              stride_2,
                 float           *output);


void Correlation1dGradA(const GPUDevice& device,
                      const int        batch_size,
                      const int        out_width,
                      const int        out_height,
                      const int        out_channels,
                      const int        max_displacement,
                      const int        x_shift,
                      const int        neighborhood_grid_width,
                      const int        kernel_radius,
                      const int        stride_1,
                      const int        stride_2,
                      const int        in_width,
                      const int        in_height,
                      const int        padded_in_width,
                      const int        padded_in_height,
                      const int        in_channels,
                      const int        in_count_per_sample,
                      const int        pad,
                      const float     *input_b,
                      const float     *gradient,
                      float           *output_a_gradient);

void Correlation1dGradB(const GPUDevice& device,
                      const int        batch_size,
                      const int        out_width,
                      const int        out_height,
                      const int        out_channels,
                      const int        max_displacement,
                      const int        x_shift,
                      const int        neighborhood_grid_width,
                      const int        kernel_radius,
                      const int        stride_1,
                      const int        stride_2,
                      const int        in_width,
                      const int        in_height,
                      const int        padded_in_width,
                      const int        padded_in_height,
                      const int        in_channels,
                      const int        in_count_per_sample,
                      const int        pad,
                      const float     *input_a,
                      const float     *gradient,
                      float           *output_b_gradient);

void Pad(const GPUDevice& device,
         const float     *input,
         int              batch_size,
         int              input_height,
         int              input_width,
         int              input_channels,
         int              output_height,
         int              output_width,
         float           *output);

template<typename Device>
class Correlation1dKernel : public OpKernel {
  public:
    explicit Correlation1dKernel(OpKernelConstruction *ctx) : OpKernel(ctx) {
      // Get the attributes
      OP_REQUIRES_OK(ctx, ctx->GetAttr("kernel_size", &kernel_size));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("max_displacement", &max_displacement));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("stride_1", &stride_1));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("stride_2", &stride_2));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("pad", &pad));

      OP_REQUIRES(ctx, kernel_size % 2 != 0, errors::InvalidArgument("kernel_size must be odd"));
    }

    void Compute(OpKernelContext *ctx) override {
      // Get the input images and transforms and verify their dimensions
      const Tensor& input_a_t = ctx->input(0);
      const Tensor& input_b_t = ctx->input(1);

      OP_REQUIRES(ctx, input_a_t.dims() == 4, errors::InvalidArgument("input_a must have rank 4"));
      OP_REQUIRES(ctx, input_b_t.dims() == 4, errors::InvalidArgument("input_b must have rank 4"));

      // Get dimensions of input (already padded)
      int batch_size     = input_a_t.dim_size(0);
      int input_height   = input_a_t.dim_size(1);
      int input_width    = input_a_t.dim_size(2);
      int input_channels = input_a_t.dim_size(3);
      int padded_height  = input_height;
      int padded_width   = input_width + 2 * pad;

      // The size of unreachable border region on each side
      int kernel_radius = (kernel_size - 1) / 2;
      int border_size   = max_displacement + kernel_radius;

      // Calculate the output dimensions
      int output_height = ceil((float)(padded_height - kernel_radius * 2) / (float)stride_1);
      int output_width  = ceil((float)(padded_width - border_size * 2) / (float)stride_1);

      OP_REQUIRES(ctx, output_height >= 1,
                  errors::InvalidArgument("Neighborhood and kernel don't fit in input height."));
      OP_REQUIRES(ctx, output_width >= 1,
                  errors::InvalidArgument("Neighborhood and kernel don't fit in input width."));

      int neighborhood_grid_radius = max_displacement / stride_2;
      int neighborhood_grid_width  = neighborhood_grid_radius * 2 + 1;
      int x_shift = -neighborhood_grid_radius;
      int output_channels          = neighborhood_grid_width;

      // Allocate the memory for the output
      Tensor *output_t;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(
                       0,
                       TensorShape({ batch_size, output_height, output_width, output_channels }),
                       &output_t));

      // Get the tensors
      auto input_a = input_a_t.tensor<float, 4>();
      auto input_b = input_b_t.tensor<float, 4>();
      auto output  = output_t->tensor<float, 4>();

      // Create temporary tensors for padded inputs
      Tensor padded_input_a_t, padded_input_b_t;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DataTypeToEnum<float>::value,
                                        TensorShape({ batch_size, padded_height, padded_width, input_channels }),
                                        &padded_input_a_t));
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DataTypeToEnum<float>::value,
                                        TensorShape({ batch_size, padded_height, padded_width, input_channels }),
                                        &padded_input_b_t));
      auto padded_input_a = padded_input_a_t.tensor<float, 4>();
      auto padded_input_b = padded_input_b_t.tensor<float, 4>();

      // Pad the inputs
      Pad(ctx->eigen_device<Device>(),
          input_a.data(),
          batch_size,
          input_height,
          input_width,
          input_channels,
          padded_height,
          padded_width,
          padded_input_a.data());
      Pad(ctx->eigen_device<Device>(),
          input_b.data(),
          batch_size,
          input_height,
          input_width,
          input_channels,
          padded_height,
          padded_width,
          padded_input_b.data());

      // Perform cross correlation
      Correlation1d(ctx->eigen_device<Device>(),
                  padded_input_a.data(),
                  padded_input_b.data(),
                  batch_size,
                  output_height,
                  output_width,
                  output_channels,
                  output_height * output_width * output_channels,
                  padded_height,
                  padded_width,
                  input_channels,
                  max_displacement,
                  x_shift,
                  neighborhood_grid_width,
                  kernel_radius,
                  kernel_size,
                  stride_1,
                  stride_2,
                  output.data());
    }

  private:
    int kernel_size;
    int max_displacement;
    int stride_1;
    int stride_2;
    int pad;
};

REGISTER_KERNEL_BUILDER(Name("Correlation1d")
                        .Device(DEVICE_GPU),
                        Correlation1dKernel<GPUDevice>)

template<typename Device>
class Correlation1dGradKernel : public OpKernel {
  public:
    explicit Correlation1dGradKernel(OpKernelConstruction *ctx) : OpKernel(ctx) {
      // Get the attributes
      OP_REQUIRES_OK(ctx, ctx->GetAttr("kernel_size", &kernel_size));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("max_displacement", &max_displacement));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("stride_1", &stride_1));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("stride_2", &stride_2));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("pad", &pad));

      OP_REQUIRES(ctx, kernel_size % 2 != 0, errors::InvalidArgument("kernel_size must be odd"));
    }

    void Compute(OpKernelContext *ctx) override {
      // Get the input images and verify their dimensions
      const Tensor& gradients_t = ctx->input(0);
      const Tensor& input_a_t   = ctx->input(1);
      const Tensor& input_b_t   = ctx->input(2);

      OP_REQUIRES(ctx, input_a_t.dims() == 4, errors::InvalidArgument("input_a must have rank 4"));
      OP_REQUIRES(ctx, input_b_t.dims() == 4, errors::InvalidArgument("input_b must have rank 4"));

      // Get dimensions of input
      const int batch_size          = input_a_t.dim_size(0);
      const int in_height           = input_a_t.dim_size(1);
      const int in_width            = input_a_t.dim_size(2);
      const int in_channels         = input_a_t.dim_size(3);
      const int in_count_per_sample = in_height * in_width * in_channels;
      const int padded_height       = in_height;
      const int padded_width        = in_width + 2 * pad;

      // The size of unreachable border region on each side
      const int kernel_radius = (kernel_size - 1) / 2;
      const int border_size   = max_displacement + kernel_radius;

      // Calculate the output dimensions
      const int out_height =  ceil((float)(padded_height - kernel_radius * 2) / (float)stride_1);;
      const int out_width  = ceil((float)(padded_width - border_size * 2) / (float)stride_1);

      const int neighborhood_grid_radius = max_displacement / stride_2;
      const int neighborhood_grid_width  = neighborhood_grid_radius * 2 + 1;
      int x_shift = -neighborhood_grid_radius;
      const int out_channels             =  neighborhood_grid_width;

      // Allocate the memory for the outputs
      Tensor *output_a_gradient_t;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_a_t.shape(), &output_a_gradient_t));
      Tensor *output_b_gradient_t;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(1, input_b_t.shape(), &output_b_gradient_t));

      // Get the tensors
      auto gradients         = gradients_t.tensor<float, 4>();
      auto input_a           = input_a_t.tensor<float, 4>();
      auto input_b           = input_b_t.tensor<float, 4>();
      auto output_a_gradient = output_a_gradient_t->tensor<float, 4>();
      auto output_b_gradient = output_b_gradient_t->tensor<float, 4>();

      // Create temporary tensors for padded inputs
      Tensor padded_input_a_t, padded_input_b_t;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DataTypeToEnum<float>::value,
                                        TensorShape({ batch_size, padded_height, padded_width, in_channels }),
                                        &padded_input_a_t));
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DataTypeToEnum<float>::value,
                                        TensorShape({ batch_size, padded_height, padded_width, in_channels }),
                                        &padded_input_b_t));
      auto padded_input_a = padded_input_a_t.tensor<float, 4>();
      auto padded_input_b = padded_input_b_t.tensor<float, 4>();

      // Pad the inputs
      Pad(ctx->eigen_device<Device>(),
          input_a.data(),
          batch_size,
          in_height,
          in_width,
          in_channels,
          padded_height,
          padded_width,
          padded_input_a.data());
      Pad(ctx->eigen_device<Device>(),
          input_b.data(),
          batch_size,
          in_height,
          in_width,
          in_channels,
          padded_height,
          padded_width,
          padded_input_b.data());

      Correlation1dGradA(ctx->eigen_gpu_device(),
                       batch_size,
                       out_width,
                       out_height,
                       out_channels,
                       max_displacement,
                       x_shift,
                       neighborhood_grid_width,
                       kernel_radius,
                       stride_1,
                       stride_2,
                       in_width,
                       in_height,
                       padded_width,
                       padded_height,
                       in_channels,
                       in_count_per_sample,
                       pad,
                       padded_input_b.data(),
                       gradients.data(),
                       output_a_gradient.data());

      Correlation1dGradB(ctx->eigen_gpu_device(),
                       batch_size,
                       out_width,
                       out_height,
                       out_channels,
                       max_displacement,
                       x_shift,
                       neighborhood_grid_width,
                       kernel_radius,
                       stride_1,
                       stride_2,
                       in_width,
                       in_height,
                       padded_width,
                       padded_height,
                       in_channels,
                       in_count_per_sample,
                       pad,
                       padded_input_a.data(),
                       gradients.data(),
                       output_b_gradient.data());
    }

  private:
    int kernel_size;
    int max_displacement;
    int stride_1;
    int stride_2;
    int pad;
};

REGISTER_KERNEL_BUILDER(Name("Correlation1dGrad")
                        .Device(DEVICE_GPU),
                        Correlation1dGradKernel<GPUDevice>)

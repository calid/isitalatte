The op in this directory was built using bazel and the TensorFlow source.  To
build it yourself do the following:

0. Clone the TensorFlow repo at https://github.com/tensorflow/tensorflow.
0. Copy the op `.cc` and `BUILD` files into
   `tensorflow/tensorflow/core/user_ops`.
0. From that directory run `bazel build -c opt
   //tensorflow/core/user_ops:normalize_image.so`. If you are using gcc>=5
   you will also need to add `--cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0` to the
   bazel command line.
0. Once the build finishes copy
   `tensorflow/bazel-bin/tensorflow/core/user_ops/normalize_image.so` back
   into this directory.

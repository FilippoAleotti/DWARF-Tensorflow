#!/usr/bin/env bash
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
CUDA_DIR=/usr/local
nvcc -std=c++11 -c -o correlation1d.cu.o correlation1d.cu.cc ${TF_CFLAGS[@]} -I $CUDA_DIR -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr
#nvcc -std=c++11 -c -o shift_corr.cu.o shift_corr.cu.cc -I $TF_INC -I $CUDA_DIR -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared -o correlation1d.so correlation1d.cc correlation1d.cu.o ${TF_CFLAGS[@]} -L /usr/local/cuda/lib64 -fPIC -lcudart ${TF_LFLAGS[@]}
#g++ -std=c++11 -shared -o shift_corr.so shift_corr.cc shift_corr.cu.o -fPIC -lcudart ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2

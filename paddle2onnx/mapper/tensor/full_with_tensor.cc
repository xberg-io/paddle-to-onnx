// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle2onnx/mapper/tensor/full_with_tensor.h"
#include "paddle2onnx/proto/p2o_paddle.pb.h"
#include <iostream>
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_PIR_MAPPER(full_with_tensor, FullWithTensorMapper)

int32_t FullWithTensorMapper::GetMinOpsetVersion(bool verbose) { return 8; }

void FullWithTensorMapper::Opset8() {
  auto value_info = GetInput("value");
  auto shape_info = GetInput("shape");
  auto output_info = GetOutput("out");

  auto shape = helper_->AutoCast(shape_info[0].name, shape_info[0].dtype,
                                 P2ODataType::INT64);

  if (shape_info[0].Rank() == 0) {
    shape = helper_->Reshape(shape, {1});
  } else if (shape_info[0].Rank() > 1) {
    shape = helper_->Reshape(shape, {-1});
  }

  auto expand_node = helper_->MakeNode("Expand", {value_info[0].name, shape});

  helper_->AutoCast(expand_node->output(0), output_info[0].name,
                    value_info[0].dtype, output_info[0].dtype);
}

} // namespace paddle2onnx

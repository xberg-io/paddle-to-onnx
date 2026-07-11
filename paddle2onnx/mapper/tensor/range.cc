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

#include "paddle2onnx/mapper/tensor/range.h"

namespace paddle2onnx {
REGISTER_MAPPER(range, RangeMapper)
REGISTER_PIR_MAPPER(range, RangeMapper)

void RangeMapper::Opset11() {
  auto start_info = GetInput("Start");
  auto end_info = GetInput("End");
  auto step_info = GetInput("Step");
  auto out_info = GetOutput("Out");
  int32_t out_dtype = -1;
  // ~keep TODO(jiangjiajun): remove eliminable Cast for constants.
  std::vector<std::string> aligned_inputs = helper_->DtypeAlignment(
      {start_info[0], end_info[0], step_info[0]}, &out_dtype);
  std::vector<int64_t> empty_axes;

  // ~keep TODO(jiangjiajun): remove eliminable Squeeze for constants.
  if (start_info[0].shape.size() > 0) {
    aligned_inputs[0] = helper_->Squeeze(aligned_inputs[0], empty_axes);
  }
  if (end_info[0].shape.size() > 0) {
    aligned_inputs[1] = helper_->Squeeze(aligned_inputs[1], empty_axes);
  }
  if (step_info[0].shape.size() > 0) {
    aligned_inputs[2] = helper_->Squeeze(aligned_inputs[2], empty_axes);
  }
  auto out = helper_->MakeNode("Range", aligned_inputs)->output(0);
  helper_->AutoCast(out, out_info[0].name, out_dtype, out_info[0].dtype);
}

} // namespace paddle2onnx

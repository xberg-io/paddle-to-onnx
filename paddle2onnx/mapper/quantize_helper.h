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

#pragma once
#include <onnx/onnx_pb.h>

#include <cmath>
#include <fstream>
#include <iomanip>

#include "paddle2onnx/mapper/mapper.h"
#include "paddle2onnx/parser/parser.h"
namespace paddle2onnx {

struct QuantizeModelProcessor {
public:
  std::vector<QuantizeInfo> quantize_info;
  const PaddleParser *parser_;
  OnnxHelper *helper_;

  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> *parameters_;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> *inputs_;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> *outputs_;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> *nodes_;
  std::vector<std::string> supported_quantize_type_;

  std::map<std::string, std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>>
      name2node_dict_;
  std::vector<std::string> tensors_to_be_quantize;
  std::vector<std::string> only_dequantize_tensors;
  void ProcessQuantizeModel(
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> *parameters,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> *inputs,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> *outputs,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> *nodes,
      OnnxHelper *helper, const std::string &deploy_backend,
      const PaddleParser &parser, std::string *calibration_cache = nullptr);

  void RemoveAllQuantizeOps();

  bool CanBeQuantize(const std::vector<std::string> &tensor_names,
                     const std::vector<int64_t> &output_index = {-1});
  void AppendQuantizeTensor(const std::string &tensor,
                            const bool &only_dequantize = false);

  void AddQDQForORT();

  bool ConnectToOutput(const std::string &output_name);

  void GenerateCache(std::string *calibration_cache);

  void AddTrtQDQ();

  void AddQDQForRKNN();

  void RemoveIdentityOp();

  void AddQDQInModel(const std::vector<std::string> &tensors_to_be_quantize);

  void QuantizeInfoBroadcast();

  void MergeConvAdd();

  void MergeConvBN();

  bool IsGraphOutput(const std::string &name);

  void SortNodes();

  bool GetTensorShape(const std::string &name, std::vector<int64_t> *shape);

  template <typename T>
  bool GetTensorByName(const std::string &name, std::vector<T> *value);

  void GetTensorWiseQuantizeInfo(const std::vector<float> &tensor,
                                 std::vector<float> *scale,
                                 std::vector<int64_t> *zero);

  void GetChannelWiseQuantizeInfo(const std::vector<float> &tensor,
                                  const std::vector<int64_t> &shape,
                                  const int64_t &quant_axis,
                                  std::vector<float> *scale,
                                  std::vector<int64_t> *zero);

  void UpdateInputNameToNodes();

  void RemoveNodeByName(const std::string &name, const bool &update_io = true);

  void ReplaceInputOfAllNodes(
      const std::string &old_name, const std::string &new_name,
      const std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>
          &except_nodes = {});
};
} // namespace paddle2onnx

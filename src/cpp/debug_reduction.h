#pragma once

#include "label.h"
#include "prediction.h"
#include "vw/core/input_parser.h"
#include "vw/core/learner.h"
#include "vw/core/prediction_type.h"
#include "vw/core/reduction_stack.h"

#include <chrono>
#include <memory>
#include <stack>
#include <string>
#include <vector>

namespace vwpy
{

struct debug_node
{
  std::string name;
  std::string function;
  bool is_multiline;
  size_t num_examples;
  prediction_t output_prediction;
  std::variant<label_variant_t, std::vector<label_variant_t>> input_labels;
  std::variant<std::vector<std::string>, std::vector<std::vector<std::string>>> interactions;
  std::variant<float, std::vector<float>> weight;
  std::variant<float, std::vector<float>> partial_prediction;
  std::variant<float, std::vector<float>> updated_prediction;
  std::variant<uint64_t, std::vector<uint64_t>> offset;

  // These durations measure the time spent below this node, that means it includes the time spent in debug code in all
  // reductions below it. It seems possible to filter out the debug code time by also measuring the time spent in debug
  // code in this reduction and collecting these times below you.
  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point end_time;

  std::chrono::high_resolution_clock::time_point overall_start_time;
  std::chrono::high_resolution_clock::time_point overall_end_time;

  std::chrono::high_resolution_clock::duration calc_self_time() const
  {
    std::chrono::high_resolution_clock::duration self_time = end_time - start_time;
    for (const auto& child : children) { self_time -= (child->overall_end_time - child->overall_start_time); }
    return self_time;
  }

  std::chrono::high_resolution_clock::duration calc_overall_time() const
  {
    return (overall_end_time - overall_start_time) - calc_debug_time_recursive();
  }

  std::chrono::high_resolution_clock::duration calc_debug_time_recursive() const
  {
    std::chrono::high_resolution_clock::duration self_time = end_time - start_time;
    std::chrono::high_resolution_clock::duration debug_time = (overall_end_time - overall_start_time) - self_time;
    for (const auto& child : children) { self_time += child->calc_debug_time_recursive(); }
    return debug_time;
  }

  std::vector<std::shared_ptr<debug_node>> children;
};

struct debug_data
{
  std::stack<std::shared_ptr<debug_node>> active;
  std::shared_ptr<debug_node> root;
};

struct debug_data_stash : public VW::details::input_parser
{
  debug_data_stash() : VW::details::input_parser("debug_data_stash")
  {
    shared_debug_state = std::make_shared<debug_data>();
  }

  bool next(VW::workspace& workspace_instance, VW::io_buf& buffer, VW::multi_ex& output_examples) override
  {
    return false;
  }

  std::shared_ptr<debug_data> shared_debug_state;
  std::vector<std::shared_ptr<void>> kept_around_reduction_state;
};

std::shared_ptr<VW::LEARNER::learner> debug_reduction_setup(VW::setup_base_i& stack_builder);

struct debug_stack_builder : public VW::default_reduction_stack_setup
{
  debug_stack_builder() : VW::default_reduction_stack_setup()
  {
    // Insert debug reduction between every existing reduction

    auto it = _reduction_stack.begin() + 1;
    auto tuple_to_insert = std::make_tuple("debug", debug_reduction_setup);
    // Insert the tuple in between each existing element
    while (it <= _reduction_stack.end())
    {
      it = _reduction_stack.insert(it, tuple_to_insert);
      it += 2;
    }

    _setup_name_map[debug_reduction_setup] = "debug";
  }
};
}  // namespace vwpy

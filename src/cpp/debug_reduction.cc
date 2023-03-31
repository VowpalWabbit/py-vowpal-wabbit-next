#include "debug_reduction.h"

#include "label.h"
#include "vw/core/debug_print.h"
#include "vw/core/label_type.h"

#include <memory>
#include <stack>
#include <string>
#include <vector>

namespace
{

vwpy::prediction_t get_prediction(const VW::example& ex, const VW::prediction_type_t type)
{
  return vwpy::to_prediction(ex.pred, type);
}

vwpy::prediction_t get_prediction(const VW::multi_ex& ex, const VW::prediction_type_t type)
{
  return vwpy::to_prediction(ex[0]->pred, type);
}

std::variant<vwpy::label_variant_t, std::vector<vwpy::label_variant_t>> get_labels(
    const VW::example& ex, const VW::label_type_t type)
{
  return vwpy::to_label_variant(ex, type);
}

std::variant<vwpy::label_variant_t, std::vector<vwpy::label_variant_t>> get_labels(
    const VW::multi_ex& ex, const VW::label_type_t type)
{
  std::vector<vwpy::label_variant_t> labels;
  for (auto& example : ex) { labels.push_back(vwpy::to_label_variant(*example, type)); }
  return labels;
}

std::variant<float, std::vector<float>> get_weight(const VW::example& ex) { return ex.weight; }

std::variant<float, std::vector<float>> get_weight(const VW::multi_ex& ex)
{
  std::vector<float> weights;
  for (auto& example : ex) { weights.push_back(example->weight); }
  return weights;
}

std::variant<float, std::vector<float>> get_partial_prediction(const VW::example& ex) { return ex.partial_prediction; }

std::variant<float, std::vector<float>> get_partial_prediction(const VW::multi_ex& ex)
{
  std::vector<float> partial_predictions;
  for (auto& example : ex) { partial_predictions.push_back(example->partial_prediction); }
  return partial_predictions;
}

std::variant<float, std::vector<float>> get_updated_prediction(const VW::example& ex) { return ex.updated_prediction; }

std::variant<float, std::vector<float>> get_updated_prediction(const VW::multi_ex& ex)
{
  std::vector<float> updated_predictions;
  for (auto& example : ex) { updated_predictions.push_back(example->updated_prediction); }
  return updated_predictions;
}

std::variant<uint64_t, std::vector<uint64_t>> get_offset(const VW::example& ex) { return ex.ft_offset; }

std::variant<uint64_t, std::vector<uint64_t>> get_offset(const VW::multi_ex& ex)
{
  std::vector<uint64_t> offsets;
  for (auto& example : ex) { offsets.push_back(example->ft_offset); }
  return offsets;
}

bool is_printable(unsigned char c) { return (c >= 32 && c <= 126); }

std::string interaction_to_string(const std::vector<VW::namespace_index>& inter_list)
{
  std::stringstream ss;
  for (auto inter : inter_list)
  {
    if (is_printable(inter)) { ss << static_cast<char>(inter); }
    else { ss << fmt::format("\\x{:x}", std::byte(inter)); }
  }
  return ss.str();
}

std::vector<std::string> get_interactions(const VW::example& ex)
{
  std::vector<std::string> interactions;
  for (auto& inter_list : *ex.interactions) { interactions.push_back(interaction_to_string(inter_list)); }
  return interactions;
}

std::vector<std::vector<std::string>> get_interactions(const VW::multi_ex& ex)
{
  std::vector<std::vector<std::string>> interactions;
  for (auto& example : ex) { interactions.push_back(get_interactions(*example)); }
  return interactions;
}

struct debug_data_holder
{
  debug_data_holder(std::shared_ptr<vwpy::debug_data> shared_debug_state) : shared_debug_state(shared_debug_state) {}
  std::shared_ptr<vwpy::debug_data> shared_debug_state;
};

template <typename ExampleT, bool is_learn>
static void debug_transform(debug_data_holder& data, VW::LEARNER::learner& base, ExampleT& ex)
{
  auto overall_start_time = std::chrono::high_resolution_clock::now();
  auto self = std::make_shared<vwpy::debug_node>();
  self->overall_start_time = overall_start_time;
  self->name = base.get_name();
  self->input_labels = get_labels(ex, base.get_input_label_type());
  self->interactions = get_interactions(ex);
  self->weight = get_weight(ex);
  self->offset = get_offset(ex);

  // Test if ExampleT is VW::example
  if constexpr (std::is_same_v<ExampleT, VW::example>)
  {
    self->is_multiline = false;
    self->num_examples = 1;
  }
  else if constexpr (std::is_same_v<ExampleT, VW::multi_ex>)
  {
    self->is_multiline = true;
    self->num_examples = ex.size();
  }
  else
  {
    static_assert(std::is_same_v<ExampleT, VW::example> || std::is_same_v<ExampleT, VW::multi_ex>,
        "ExampleT must be VW::example or VW::multi_ex");
  }

  if (!data.shared_debug_state->active.empty()) { data.shared_debug_state->active.top()->children.push_back(self); }
  else { data.shared_debug_state->root = self; }

  // TODO: make this more robust.
  // If we are entering into the same reduction which currently resides at the top of the stack we do not push a new
  // node. We should probably do this more robustly using offsets though...
  bool pushed = false;
  if (data.shared_debug_state->active.empty() || data.shared_debug_state->active.top()->name != base.get_name())
  {
    pushed = true;
    data.shared_debug_state->active.push(self);
  }
  self->start_time = std::chrono::high_resolution_clock::now();
  if constexpr (is_learn)
  {
    self->function = "learn";
    base.learn(ex);
  }
  else
  {
    self->function = "predict";
    base.predict(ex);
  }
  self->end_time = std::chrono::high_resolution_clock::now();
  self->output_prediction = get_prediction(ex, base.get_output_prediction_type());
  self->updated_prediction = get_updated_prediction(ex);
  self->partial_prediction = get_partial_prediction(ex);
  if (pushed) { data.shared_debug_state->active.pop(); }
  self->overall_end_time = std::chrono::high_resolution_clock::now();
}
}  // namespace

std::shared_ptr<VW::LEARNER::learner> vwpy::debug_reduction_setup(VW::setup_base_i& stack_builder)
{
  auto base = stack_builder.setup_base_learner();
  if (base == nullptr) { return nullptr; }

  // We only want to insert the debug interceptor if the next reduction is not the debug reduction.
  if (base.get()->get_name().find("debug") != std::string::npos) { return base; }

  auto* all = stack_builder.get_all_pointer();

  // This is a huge hack, we need a way to keep global state between all instantiations of the debug reduction.
  // We use this abstract class field that we assume is not in use.
  if (all->parser_runtime.custom_parser == nullptr)
  {
    all->parser_runtime.custom_parser = VW::make_unique<vwpy::debug_data_stash>();
  }
  auto* cast_stash = dynamic_cast<vwpy::debug_data_stash*>(all->parser_runtime.custom_parser.get());
  if (cast_stash == nullptr)
  {
    throw std::runtime_error(
        "Unable to cast custom_parser to debug_data_stash. Currently the debug infrastructure cannot co-exist with a "
        "custom parser such as CSV.");
  }

  auto data = std::make_unique<debug_data_holder>(cast_stash->shared_debug_state);

  auto reduction_name = fmt::format("{}-debug", base->get_name());
  std::shared_ptr<VW::LEARNER::learner> learner = nullptr;
  if (base->is_multiline())
  {
    learner = VW::LEARNER::make_reduction_learner(std::move(data), base, debug_transform<VW::multi_ex, true>,
        debug_transform<VW::multi_ex, false>, reduction_name)
                  .set_learn_returns_prediction(base->learn_returns_prediction)
                  .set_input_label_type(base->get_input_label_type())
                  .set_output_label_type(base->get_input_label_type())
                  .set_input_prediction_type(base->get_output_prediction_type())
                  .set_output_prediction_type(base->get_output_prediction_type())
                  .build();
  }
  else
  {
    learner = VW::LEARNER::make_reduction_learner(
        std::move(data), base, debug_transform<VW::example, true>, debug_transform<VW::example, false>, reduction_name)
                  .set_learn_returns_prediction(base->learn_returns_prediction)
                  .set_input_label_type(base->get_input_label_type())
                  .set_output_label_type(base->get_input_label_type())
                  .set_input_prediction_type(base->get_output_prediction_type())
                  .set_output_prediction_type(base->get_output_prediction_type())
                  .build();
  }
  cast_stash->kept_around_reduction_state.push_back(
      learner->get_internal_type_erased_data_pointer_test_use_only_shared());
  learner->set_internal_type_erased_data_pointer_does_not_override_funcs(
      base->get_internal_type_erased_data_pointer_test_use_only_shared());
  return learner;
}

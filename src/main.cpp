#include "vw/common/text_utils.h"
#include "vw/config/options_cli.h"
#include "vw/core/array_parameters_dense.h"
#include "vw/core/cache.h"
#include "vw/core/cb.h"
#include "vw/core/constant.h"
#include "vw/core/cost_sensitive.h"
#include "vw/core/example.h"
#include "vw/core/global_data.h"
#include "vw/core/guard.h"
#include "vw/core/io_buf.h"
#include "vw/core/label_type.h"
#include "vw/core/learner.h"
#include "vw/core/loss_functions.h"
#include "vw/core/memory.h"
#include "vw/core/merge.h"
#include "vw/core/metric_sink.h"
#include "vw/core/multiclass.h"
#include "vw/core/object_pool.h"
#include "vw/core/parse_example.h"
#include "vw/core/parse_regressor.h"
#include "vw/core/prediction_type.h"
#include "vw/core/scope_exit.h"
#include "vw/core/simple_label.h"
#include "vw/core/version.h"
#include "vw/core/vw.h"
#include "vw/io/io_adapter.h"
#include "vw/io/logger.h"
#include "vw/json_parser/decision_service_utils.h"
#include "vw/json_parser/parse_example_json.h"

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <sys/types.h>

#include <iostream>
#include <memory>
#include <optional>
#include <variant>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

// helper type for the visitor #4
template <class... Ts>
struct overloaded : Ts...
{
  using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

namespace py = pybind11;
namespace
{

class python_reader : public VW::io::reader
{
public:
  python_reader(py::object file) : VW::io::reader(false), _file(file) {}

  ssize_t read(char* buffer, size_t num_bytes) override
  {
    auto read_func = _file.attr("read");
    auto res = read_func(num_bytes);
    auto bytes = res.cast<py::bytes>();
    std::string_view bytes_view = bytes;
    if (bytes_view.size() > 0) { std::memcpy(buffer, bytes_view.data(), bytes_view.size()); }
    return bytes_view.size();
  }

private:
  py::object _file;
};

class python_writer : public VW::io::writer
{
public:
  python_writer(py::object file) : _file(file) {}

  ssize_t write(const char* buffer, size_t num_bytes) override
  {
    auto res = _file.attr("write")(py::bytes(buffer, num_bytes));
    return res.cast<ssize_t>();
  }

  void flush() override { _file.attr("flush")(); }

private:
  py::object _file;
};

// This is a global object pool for examples.
static VW::object_pool<VW::example> SHARED_EXAMPLE_POOL;

void clean_example(VW::example& ec)
{
  for (auto& fs : ec) { fs.clear(); }

  ec.pred = VW::polyprediction{};
  ec.l = VW::polylabel{};
  ec.ex_reduction_features.clear();
  ec.indices.clear();
  ec.tag.clear();
  ec.sorted = false;
  ec.end_pass = false;
  ec.is_newline = false;
  ec.ex_reduction_features.clear();
  ec.num_features_from_interactions = 0;
}

std::shared_ptr<VW::example> get_example_from_pool()
{
  return std::shared_ptr<VW::example>(SHARED_EXAMPLE_POOL.get_object().release(),
      [](VW::example* ptr)
      {
        clean_example(*ptr);
        SHARED_EXAMPLE_POOL.return_object(ptr);
      });
}

using scalar_pred_t = float;
using scalars_pred_t = std::vector<float>;
using action_scores_pred_t = std::vector<std::tuple<uint32_t, float>>;
using decision_scores_pred_t = std::vector<std::vector<std::tuple<uint32_t, float>>>;
using multiclass_pred_t = uint32_t;
using multilabels_pred_t = std::vector<uint32_t>;
using prob_density_func_pred_t = std::vector<std::tuple<float, float, float>>;
using prob_density_func_value_pred_t = std::tuple<float, float>;
using active_multiclass_pred_t = std::tuple<uint32_t, std::vector<uint32_t>>;

using prediction_t = std::variant<scalar_pred_t, scalars_pred_t, action_scores_pred_t, decision_scores_pred_t,
    multiclass_pred_t, multilabels_pred_t, prob_density_func_pred_t, prob_density_func_value_pred_t,
    active_multiclass_pred_t, py::none>;

// TODO: consider a more efficient way of exposing these values
prediction_t to_prediction(const VW::polyprediction& polypred, VW::prediction_type_t type)
{
  assert(type != VW::prediction_type_t::MULTICLASS_PROBS && "MULTICLASS_PROBS exists but is not in use.");
  switch (type)
  {
    case VW::prediction_type_t::SCALAR:
      return polypred.scalar;
    case VW::prediction_type_t::SCALARS:
      return std::vector<float>(polypred.scalars.begin(), polypred.scalars.end());
    case VW::prediction_type_t::ACTION_SCORES:
    case VW::prediction_type_t::ACTION_PROBS:
    {
      std::vector<std::tuple<uint32_t, float>> result;
      for (const auto& action_score : polypred.a_s) { result.emplace_back(action_score.action, action_score.score); }
      return result;
    }
    case VW::prediction_type_t::PDF:
    {
      std::vector<std::tuple<float, float, float>> result;
      for (const auto& pdf : polypred.pdf) { result.emplace_back(pdf.left, pdf.right, pdf.pdf_value); }
      return result;
    }
    case VW::prediction_type_t::MULTICLASS:
      return polypred.multiclass;
    case VW::prediction_type_t::MULTILABELS:
      return std::vector<uint32_t>(polypred.multilabels.label_v.begin(), polypred.multilabels.label_v.end());
    case VW::prediction_type_t::PROB:
      return polypred.prob;
    case VW::prediction_type_t::DECISION_PROBS:
    {
      std::vector<std::vector<std::tuple<uint32_t, float>>> result;
      result.reserve(polypred.decision_scores.size());
      for (const auto& decision_scores : polypred.decision_scores)
      {
        std::vector<std::tuple<uint32_t, float>> decision_scores_result;
        decision_scores_result.reserve(decision_scores.size());
        for (const auto& action_score : decision_scores)
        {
          decision_scores_result.emplace_back(action_score.action, action_score.score);
        }
        result.emplace_back(decision_scores_result);
      }
      return result;
    }
    case VW::prediction_type_t::ACTION_PDF_VALUE:
      return std::tuple<float, float>(polypred.pdf_value.action, polypred.pdf_value.pdf_value);
    case VW::prediction_type_t::ACTIVE_MULTICLASS:
      return std::make_tuple(polypred.active_multiclass.predicted_class,
          std::vector<uint32_t>(polypred.active_multiclass.more_info_required_for_classes.begin(),
              polypred.active_multiclass.more_info_required_for_classes.end()));
    case VW::prediction_type_t::NOPRED:
    default:
      return py::none();
  }
}

struct cache_reader
{
  cache_reader(std::shared_ptr<VW::workspace> workspace, py::object file) : _workspace(workspace), _file(file)
  {
    auto reader = VW::make_unique<python_reader>(file);
    read_cache_header(*reader);
    _buffer.add_file(std::move(reader));
  }

  std::shared_ptr<VW::example> read_cache_example()
  {
    VW::multi_ex examples;
    auto return_value = get_example_from_pool();
    examples.push_back(return_value.get());

    auto bytes_read = VW::parsers::cache::read_example_from_cache(_workspace.get(), _buffer, examples);
    if (bytes_read == 0) { return nullptr; }

    return return_value;
  }

private:
  // Impl from VW
  void read_cache_header(VW::io::reader& cache_reader)
  {
    size_t version_buffer_length;
    if (static_cast<size_t>(cache_reader.read(reinterpret_cast<char*>(&version_buffer_length),
            sizeof(version_buffer_length))) < sizeof(version_buffer_length))
    {
      THROW("failed to read: version_buffer_length");
    }

    if (version_buffer_length > 61) THROW("cache version too long, cache file is probably invalid");
    if (version_buffer_length == 0) THROW("cache version too short, cache file is probably invalid");

    std::vector<char> version_buffer(version_buffer_length);
    if (static_cast<size_t>(cache_reader.read(version_buffer.data(), version_buffer_length)) < version_buffer_length)
    {
      THROW("failed to read: version buffer");
    }
    VW::version_struct cache_version(version_buffer.data());
    if (cache_version != VW::VERSION)
    {
      auto msg = fmt::format(
          "Cache file version does not match current VW version. Cache files must be produced by the version consuming "
          "them. Cache version: {} VW version: {}",
          cache_version.to_string(), VW::VERSION.to_string());
      THROW(msg);
    }

    char marker;
    if (static_cast<size_t>(cache_reader.read(&marker, sizeof(marker))) < sizeof(marker)) { THROW("failed to read"); }

    if (marker != 'c') THROW("data file is not a cache file");

    uint32_t cache_numbits;
    if (static_cast<size_t>(cache_reader.read(reinterpret_cast<char*>(&cache_numbits), sizeof(cache_numbits))) <
        sizeof(cache_numbits))
    {
      THROW("failed to read");
    }

    // TODO: consider validating the number of bits
  }

  py::object _file;
  VW::io_buf _buffer;
  std::shared_ptr<VW::workspace> _workspace;
};

struct logger_context
{
  py::object driver_logger;
  py::object log_logger;
};

struct workspace_with_logger_contexts
{
  std::unique_ptr<logger_context> logger_context_ptr;
  std::shared_ptr<VW::workspace> workspace_ptr;
};

// TODO capture audit logs and send to their own log stream
void driver_log(void* context, const std::string& message)
{
  // We don't need to take the GIL here because all C++ should be driven by
  // Python and not a background thread.
  py::object& driver_logger = static_cast<logger_context*>(context)->driver_logger;
  driver_logger.attr("info")(message);
}

void log_log(void* context, VW::io::log_level level, const std::string& message)
{
  // We don't need to take the GIL here because all C++ should be driven by
  // Python and not a background thread.
  py::object& log_logger = static_cast<logger_context*>(context)->log_logger;
  switch (level)
  {
    case VW::io::log_level::TRACE_LEVEL:
      log_logger.attr("debug")(message);
      break;
    case VW::io::log_level::DEBUG_LEVEL:
      log_logger.attr("debug")(message);
      break;
    case VW::io::log_level::INFO_LEVEL:
      log_logger.attr("info")(message);
      break;
    case VW::io::log_level::WARN_LEVEL:
      log_logger.attr("warning")(message);
      break;
    case VW::io::log_level::ERROR_LEVEL:
      log_logger.attr("error")(message);
      break;
    case VW::io::log_level::CRITICAL_LEVEL:
      log_logger.attr("critical")(message);
      break;
    case VW::io::log_level::OFF_LEVEL:
      break;
  }
}

std::shared_ptr<VW::example> parse_text_line(workspace_with_logger_contexts& workspace, std::string_view line)
{
  auto ex = get_example_from_pool();
  VW::parsers::text::read_line(*workspace.workspace_ptr, ex.get(), line);
  return ex;
}

std::vector<std::shared_ptr<VW::example>> parse_dsjson_line(
    workspace_with_logger_contexts& workspace, std::string_view line)
{
  auto ex = get_example_from_pool();

  VW::multi_ex examples;
  examples.push_back(SHARED_EXAMPLE_POOL.get_object().release());

  auto example_factory = []() -> VW::example& { return *SHARED_EXAMPLE_POOL.get_object().release(); };

  VW::parsers::json::decision_service_interaction interaction;
  try
  {
    std::vector<char> owned_str;
    owned_str.resize(line.size() + 1);
    std::memcpy(owned_str.data(), line.data(), line.size());
    owned_str[line.size()] = '\0';

    // Not using the copy_line param as there were parse issues caused. It is possible they are due to the fact the line
    // input does not necessarily have a null terminator.

    bool result;
    if (workspace.workspace_ptr->audit || workspace.workspace_ptr->hash_inv)
    {
      result = VW::parsers::json::read_line_decision_service_json<true>(
          *workspace.workspace_ptr, examples, owned_str.data(), owned_str.size(), false, example_factory, &interaction);
    }
    else
    {
      result = VW::parsers::json::read_line_decision_service_json<false>(
          *workspace.workspace_ptr, examples, owned_str.data(), owned_str.size(), false, example_factory, &interaction);
    }

    // Since we are using strict parse any errors should be surfaced via an exception.
    assert(result);
  }
  catch (const VW::vw_exception& ex)
  {
    for (auto* ex : examples)
    {
      clean_example(*ex);
      SHARED_EXAMPLE_POOL.return_object(ex);
    }
    throw;
  }

  std::vector<std::shared_ptr<VW::example>> result;
  result.reserve(examples.size());
  for (auto* ex : examples)
  {
    result.emplace_back(ex,
        [](VW::example* ex)
        {
          clean_example(*ex);
          SHARED_EXAMPLE_POOL.return_object(ex);
        });
  }

  return result;
}

std::vector<std::shared_ptr<VW::example>> parse_json_line(
    workspace_with_logger_contexts& workspace, std::string_view line)
{
  auto ex = get_example_from_pool();

  VW::multi_ex examples;
  examples.push_back(SHARED_EXAMPLE_POOL.get_object().release());

  auto example_factory = []() -> VW::example& { return *SHARED_EXAMPLE_POOL.get_object().release(); };

  VW::parsers::json::decision_service_interaction interaction;
  try
  {
    // Must copy as the input is destructively parsed.
    std::vector<char> owned_str;
    owned_str.resize(line.size() + 1);
    std::memcpy(owned_str.data(), line.data(), line.size());
    owned_str[line.size()] = '\0';

    if (workspace.workspace_ptr->audit || workspace.workspace_ptr->hash_inv)
    {
      VW::parsers::json::template read_line_json<true>(
          *workspace.workspace_ptr, examples, owned_str.data(), owned_str.size(), example_factory);
    }
    else
    {
      VW::parsers::json::template read_line_json<false>(
          *workspace.workspace_ptr, examples, owned_str.data(), owned_str.size(), example_factory);
    }
  }
  catch (const VW::vw_exception& ex)
  {
    for (auto* ex : examples)
    {
      clean_example(*ex);
      SHARED_EXAMPLE_POOL.return_object(ex);
    }
    throw;
  }

  std::vector<std::shared_ptr<VW::example>> result;
  result.reserve(examples.size());
  for (auto* ex : examples)
  {
    result.emplace_back(ex,
        [](VW::example* ex)
        {
          clean_example(*ex);
          SHARED_EXAMPLE_POOL.return_object(ex);
        });
  }

  return result;
}

// Impl from VW
void write_cache_header(workspace_with_logger_contexts& workspace, py::object file)
{
  python_writer writer(file);
  size_t v_length = static_cast<uint64_t>(VW::VERSION.to_string().length()) + 1;

  writer.write(reinterpret_cast<const char*>(&v_length), sizeof(v_length));
  writer.write(VW::VERSION.to_string().c_str(), v_length);
  writer.write("c", 1);
  writer.write(
      reinterpret_cast<const char*>(&workspace.workspace_ptr->num_bits), sizeof(workspace.workspace_ptr->num_bits));
}

void write_cache_example(workspace_with_logger_contexts& workspace, VW::example& ex, py::object file)
{
  VW::parsers::cache::details::cache_temp_buffer temp_buffer;
  VW::io_buf output;
  output.add_file(VW::make_unique<python_writer>(file));
  VW::parsers::cache::write_example_to_cache(output, &ex, workspace.workspace_ptr->example_parser->lbl_parser,
      workspace.workspace_ptr->parse_mask, temp_buffer);
  output.flush();
}

std::unique_ptr<VW::model_delta> merge_deltas(const std::vector<const VW::model_delta*>& deltas_to_merge)
{
  auto result = VW::merge_deltas(deltas_to_merge);
  return std::make_unique<VW::model_delta>(std::move(result));
}

std::unique_ptr<VW::model_delta> calculate_delta(
    const workspace_with_logger_contexts& base_workspace, const workspace_with_logger_contexts& derived_workspace)
{
  auto delta = *derived_workspace.workspace_ptr - *base_workspace.workspace_ptr;
  return std::make_unique<VW::model_delta>(std::move(delta));
}

std::unique_ptr<workspace_with_logger_contexts> apply_delta(
    const workspace_with_logger_contexts& base_workspace, const VW::model_delta& delta)
{
  auto applied = *base_workspace.workspace_ptr + delta;
  return std::make_unique<workspace_with_logger_contexts>(
      workspace_with_logger_contexts{std::make_unique<logger_context>(*base_workspace.logger_context_ptr),
          std::shared_ptr<VW::workspace>(std::move(applied))});
}

template <typename LearnerT, typename ExampleT>
void update_stats_recursive(VW::workspace& workspace, LearnerT& learner, ExampleT& example)
{
  if (learner.has_update_stats())
  {
    learner.update_stats(workspace, example);
    return;
  }

  const auto has_at_least_one_new_style_func = learner.has_update_stats() || learner.has_output_example_prediction() ||
      learner.has_print_update() || learner.has_cleanup_example();

  // Finish example used to utilize the copy forwarding semantics.
  // Traverse until first hit to mimic this but with greater type safety.
  auto* base = learner.get_base_learner();
  if (base != nullptr)
  {
    if (learner.is_multiline() != base->is_multiline())
    {
      THROW("Cannot forward update_stats call across multiline/singleline boundary.");
    }
    base->finish_example(workspace, example);
  }
  else { THROW("No update_stats functions were registered in the stack."); }
}

void py_setup_example(VW::workspace& ws, VW::example& ex)
{
  ex.partial_prediction = 0.;
  ex.num_features = 0;
  ex.reset_total_sum_feat_sq();
  ex.loss = 0.;
  ex.debug_current_reduction_depth = 0;
  // TODO: workout if this is necessary or how to set it from a non-friend function
  // ex._use_permutations = all.permutations;

  ex.weight = ws.example_parser->lbl_parser.get_weight(ex.l, ex.ex_reduction_features);

  if (ws.add_constant)
  {
    // TODO make workspace a const arg here.
    VW::add_constant_feature(ws, &ex);
  }

  uint64_t multiplier = static_cast<uint64_t>(ws.wpp) << ws.weights.stride_shift();

  if (multiplier != 1)
  {  // make room for per-feature information.
    for (auto& fs : ex)
    {
      for (auto& j : fs.indices) { j *= multiplier; }
    }
  }
  ex.num_features = 0;
  for (const auto& fs : ex) { ex.num_features += fs.size(); }

  if (ex.interactions != nullptr)
  {
    THROW("Example has either already been setup, or was never unsetup. This should never happen and is a bug.")
  }

  // Set the interactions for this example to the global set.
  ex.interactions = &ws.interactions;
  ex.extent_interactions = &ws.extent_interactions;
}

void py_setup_example(VW::workspace& ws, std::vector<VW::example*>& ex)
{
  for (auto& example : ex) { py_setup_example(ws, *example); }
}

void py_unsetup_example(VW::workspace& ws, VW::example& ex)
{
  // Reset these to avoid reuse issues, but make sure keep the label that was passed in.
  // This is wasteful from a memory perspective but important for correctness at
  // the moment.
  VW::polylabel replacement{};
  switch (ws.l->get_input_label_type())
  {
    case VW::label_type_t::SIMPLE:
      replacement.simple = std::move(ex.l.simple);
      break;
    case VW::label_type_t::CB:
      replacement.cb = std::move(ex.l.cb);
      break;
    case VW::label_type_t::CB_EVAL:
      replacement.cb_eval = std::move(ex.l.cb_eval);
      break;
    case VW::label_type_t::CS:
      replacement.cs = std::move(ex.l.cs);
      break;
    case VW::label_type_t::MULTILABEL:
      replacement.multilabels = std::move(ex.l.multilabels);
      break;
    case VW::label_type_t::MULTICLASS:
      replacement.multi = std::move(ex.l.multi);
      break;
    case VW::label_type_t::CCB:
      replacement.conditional_contextual_bandit = std::move(ex.l.conditional_contextual_bandit);
      break;
    case VW::label_type_t::SLATES:
      replacement.slates = std::move(ex.l.slates);
      break;
    case VW::label_type_t::NOLABEL:
      break;
    case VW::label_type_t::CONTINUOUS:
      replacement.cb_cont = std::move(ex.l.cb_cont);
      break;
    default:
      THROW("Unknown label type encountered in py_unsetup_example");
  }
  ex.l = std::move(replacement);
  ex.pred = VW::polyprediction{};

  if (ws.add_constant)
  {
    if (ex.feature_space[VW::details::CONSTANT_NAMESPACE].size() != 1)
    {
      THROW("Constant feature not found. This should not happen.");
    }
    ex.feature_space[VW::details::CONSTANT_NAMESPACE].clear();
    auto num_times = std::count(ex.indices.begin(), ex.indices.end(), VW::details::CONSTANT_NAMESPACE);
    if (num_times != 1) { THROW("Constant index not found. This should not happen."); }
    auto it = std::find(ex.indices.begin(), ex.indices.end(), VW::details::CONSTANT_NAMESPACE);
    ex.indices.erase(it);
  }

  uint32_t multiplier = ws.wpp << ws.weights.stride_shift();
  if (multiplier != 1)
  {
    for (auto ns : ex.indices)
    {
      for (auto& idx : ex.feature_space[ns].indices) { idx /= multiplier; }
    }
  }

  if (ex.interactions == nullptr)
  {
    THROW("Example has either already been unsetup, or was never setup. This should never happen and is a bug.")
  }

  ex.interactions = nullptr;
  ex.extent_interactions = nullptr;
}

void py_unsetup_example(VW::workspace& ws, std::vector<VW::example*>& ex)
{
  for (auto& example : ex) { py_unsetup_example(ws, *example); }
}

// return type is an optional error information (nullopt if success), driver output, list of log messages
// stdin is not supported
std::tuple<std::optional<std::string>, std::string, std::vector<std::string>> run_cli_driver(
    const std::vector<std::string>& args, bool onethread)
{
  auto args_copy = args;
  args_copy.push_back("--no_stdin");
  auto options = VW::make_unique<VW::config::options_cli>(args_copy);
  std::stringstream driver_log;
  std::vector<std::string> log_log;

  auto logger = VW::io::create_custom_sink_logger(&log_log,
      [](void* context, VW::io::log_level /* unused */, const std::string& message)
      {
        auto* log_log = static_cast<std::vector<std::string>*>(context);
        log_log->push_back(message);
      });

  auto driver_logger = [](void* context, const std::string& message)
  {
    auto* driver_log = static_cast<std::stringstream*>(context);
    *driver_log << message;
  };

  try
  {
    auto all = VW::initialize(std::move(options), nullptr, driver_logger, &driver_log, &logger);
    all->vw_is_main = true;

    if (onethread) { VW::LEARNER::generic_driver_onethread(*all); }
    else
    {
      VW::start_parser(*all);
      VW::LEARNER::generic_driver(*all);
      VW::end_parser(*all);
    }

    if (all->example_parser->exc_ptr) { std::rethrow_exception(all->example_parser->exc_ptr); }
    VW::sync_stats(*all);
    all->finish();
  }
  catch (const std::exception& ex)
  {
    return std::make_tuple(ex.what(), driver_log.str(), log_log);
  }
  catch (...)
  {
    return std::make_tuple("Unknown exception occurred", driver_log.str(), log_log);
  }

  return std::make_tuple(std::nullopt, driver_log.str(), log_log);
}

struct dense_weight_holder
{
  dense_weight_holder(VW::dense_parameters* weights, size_t wpp, std::shared_ptr<VW::workspace> ws)
      : weights(weights), wpp(wpp), ws(ws)
  {
  }

  VW::dense_parameters* weights;
  size_t wpp;
  std::shared_ptr<VW::workspace> ws;
};

struct python_dict_writer : VW::metric_sink_visitor
{
  python_dict_writer(py::dict& dest_dict) : _dest_dict(dest_dict) {}
  void int_metric(const std::string& key, uint64_t value) override { _dest_dict[key.c_str()] = value; }
  void float_metric(const std::string& key, float value) override { _dest_dict[key.c_str()] = value; }
  void string_metric(const std::string& key, const std::string& value) override { _dest_dict[key.c_str()] = value; }
  void bool_metric(const std::string& key, bool value) override { _dest_dict[key.c_str()] = value; }
  void sink_metric(const std::string& key, const VW::metric_sink& value) override
  {
    py::dict nested;
    auto nested_py = python_dict_writer(nested);
    value.visit(nested_py);
    _dest_dict[key.c_str()] = nested;
  }

private:
  py::dict& _dest_dict;
};

py::dict convert_metrics_to_dict(const VW::metric_sink& metrics)
{
  py::dict dictionary;
  python_dict_writer writer(dictionary);
  metrics.visit(writer);
  return dictionary;
}

// Labels

struct py_simple_label
{
  float label;
  float weight;
  float initial;
};

bool is_shared(const VW::cs_label& label)
{
  const auto& costs = label.costs;
  if (costs.size() != 1) { return false; }
  if (costs[0].class_index != 0) { return false; }
  if (costs[0].x != -FLT_MAX) { return false; }
  return true;
}

}  // namespace

PYBIND11_MODULE(_core, m)
{
  py::options options;
  options.disable_enum_members_docstring();

  py::class_<dense_weight_holder>(m, "DenseParameters", py::buffer_protocol())
      .def_buffer(
          [](dense_weight_holder& m) -> py::buffer_info
          {
            auto length = (m.weights->mask() + 1) >> m.weights->stride_shift();
            return py::buffer_info(m.weights->first(),  /* Pointer to buffer */
                sizeof(float),                          /* Size of one scalar */
                py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
                3,                                      /* Number of dimensions */
                {static_cast<ssize_t>(length), static_cast<ssize_t>(m.wpp),
                    static_cast<ssize_t>(m.weights->stride())}, /* Buffer dimensions */
                {sizeof(float) * static_cast<ssize_t>(m.wpp) * static_cast<ssize_t>(m.weights->stride()),
                    sizeof(float) * static_cast<ssize_t>(m.weights->stride()), sizeof(float)}
                /* Strides (in bytes) for each index */
            );
          });

  py::enum_<VW::label_type_t>(m, "LabelType")
      .value("Simple", VW::label_type_t::SIMPLE)
      .value("CB", VW::label_type_t::CB)
      .value("CBEval", VW::label_type_t::CB_EVAL)
      .value("CS", VW::label_type_t::CS)
      .value("Multilabel", VW::label_type_t::MULTILABEL)
      .value("Multiclass", VW::label_type_t::MULTICLASS)
      .value("CCB", VW::label_type_t::CCB)
      .value("Slates", VW::label_type_t::SLATES)
      .value("NoLabel", VW::label_type_t::NOLABEL)
      .value("Continuous", VW::label_type_t::CONTINUOUS);

  py::enum_<VW::prediction_type_t>(m, "PredictionType")
      .value("Scalar", VW::prediction_type_t::SCALAR)
      .value("Scalars", VW::prediction_type_t::SCALARS)
      .value("ActionScores", VW::prediction_type_t::ACTION_SCORES)
      .value("Pdf", VW::prediction_type_t::PDF)
      .value("ActionProbs", VW::prediction_type_t::ACTION_PROBS)
      .value("Multiclass", VW::prediction_type_t::MULTICLASS)
      .value("Multilabels", VW::prediction_type_t::MULTILABELS)
      .value("Prob", VW::prediction_type_t::PROB)
      .value("MulticlassProbs", VW::prediction_type_t::MULTICLASS_PROBS)
      .value("DecisionProbs", VW::prediction_type_t::DECISION_PROBS)
      .value("ActionPdfValue", VW::prediction_type_t::ACTION_PDF_VALUE)
      .value("ActiveMulticlass", VW::prediction_type_t::ACTIVE_MULTICLASS)
      .value("NoPred", VW::prediction_type_t::NOPRED);

  py::class_<py_simple_label>(m, "SimpleLabel")
      .def(py::init(
               [](float label, float weight, float initial)
               {
                 py_simple_label l;
                 l.label = label;
                 l.weight = weight;
                 l.initial = initial;
                 return l;
               }),
          R"docstring(
    A label representing a simple regression problem.

    Args:
      label (float): The label.
      weight (float): The weight of the example.
      initial (float): The initial value of the prediction.
)docstring",
          py::arg("label"), py::arg("weight") = 1.f, py::arg("initial") = 0.f)
      .def_readwrite("label", &py_simple_label::label, R"docstring(
    The label.
)docstring")
      .def_readwrite("weight", &py_simple_label::weight, R"docstring(
    The weight of this label.
)docstring")
      .def_readwrite("initial", &py_simple_label::initial, R"docstring(
    The initial value of the prediction.
)docstring");

  py::class_<VW::multiclass_label>(m, "MulticlassLabel")
      .def(py::init(
               [](uint32_t label, float weight)
               {
                 VW::multiclass_label l;
                 l.label = label;
                 l.weight = weight;
                 return l;
               }),
          R"docstring(
    A label representing a multiclass classification problem.

    Args:
      label (int): The label.
      weight (float): The weight of the example.
)docstring",
          py::arg("label"), py::arg("weight") = 1.f)
      .def_readwrite("label", &VW::multiclass_label::label, R"docstring(
    The class of this label.
)docstring")
      .def_readwrite("weight", &VW::multiclass_label::weight, R"docstring(
    The weight of this label.
)docstring");

  py::class_<VW::cb_label>(m, "CBLabel")
      .def(py::init(
               [](std::optional<std::variant<std::tuple<float, float>, std::tuple<uint32_t, float, float>>> label_value,
                   float weight, bool is_shared)
               {
                 auto label = std::make_unique<VW::cb_label>();
                 label->weight = weight;
                 if (is_shared)
                 {
                   if (label_value.has_value())
                   {
                     throw std::invalid_argument("Shared examples cannot have action, cost, or probability.");
                   }
                   // Shared examples have essentially a sentinel value as prob.
                   label->costs.emplace_back(VW::cb_class(0.f, 0, -1.f));
                   return label;
                 }

                 if (label_value.has_value())
                 {
                   std::visit(
                       overloaded{
                           [&](std::tuple<float, float> value)
                           {
                             auto [cost, probability] = value;
                             label->costs.emplace_back(VW::cb_class(cost, 0, probability));
                           },
                           [&](std::tuple<uint32_t, float, float> value)
                           {
                             auto [action, cost, probability] = value;
                             label->costs.emplace_back(VW::cb_class(cost, action, probability));
                           },
                       },
                       *label_value);
                 }

                 return label;
               }),
          py::kw_only(), py::arg("label") = py::none(), py::arg("weight") = 1.f, py::arg("shared") = false, R"docstring(
    A label representing a contextual bandit problem.

    .. note::
      Currently the label can only contain 1 or 0 cb costs. There is a mode in vw for CB (non-adf) that allows for multiple cb_classes per example, but it is not currently accessible via direct label access. If creating examples/labels from parsed input it should still work as expected. If you need this feature, please open an issue on the github repo.

    Args:
      label (Optional[Union[Tuple[float, float], Tuple[int, float, float]]): This is (action, cost, probability). The same rules as VW apply for if the action can be left out of the tuple.
      weight (float): The weight of the example.
      shared (bool): Whether the example is shared. This is only used for ADF examples and must be the first example. There can only be one shared example per ADF example list.
)docstring")
      .def_property_readonly(
          "shared", [](VW::cb_label& l) -> bool { return l.costs.size() == 1 && l.costs[0].probability == -1.f; },
          R"docstring(
    Whether the example is shared. This is only used for ADF examples and must be the first example. There can only be one shared example per ADF example list.
)docstring")
      .def_property(
          "label",
          [](VW::cb_label& l) -> std::optional<std::tuple<uint32_t, float, float>>
          {
            if (l.costs.size() == 0) { return std::nullopt; }
            if (l.costs.size() == 1 && l.costs[0].probability == -1.f) { return std::nullopt; }
            return std::make_tuple(l.costs[0].action, l.costs[0].cost, l.costs[0].probability);
          },
          [](VW::cb_label& l,
              std::optional<std::variant<std::tuple<float, float>, std::tuple<uint32_t, float, float>>> label_value)
          {
            if (l.costs.size() == 1 && l.costs[0].probability == -1.f)
            {
              throw std::invalid_argument("Shared examples cannot have action, cost, or probability.");
            }
            l.costs.clear();
            if (label_value.has_value())
            {
              std::visit(
                  overloaded{
                      [&](std::tuple<float, float> value)
                      {
                        auto [cost, probability] = value;
                        l.costs.emplace_back(VW::cb_class(cost, 0, probability));
                      },
                      [&](std::tuple<uint32_t, float, float> value)
                      {
                        auto [action, cost, probability] = value;
                        l.costs.emplace_back(VW::cb_class(cost, action, probability));
                      },
                  },
                  *label_value);
            }
          },
          R"docstring(
    The label for the example. The format of the label is (action, cost, probability). If the action is not specified, it will be set to 0.
)docstring")
      .def_readwrite("weight", &VW::cb_label::weight, R"docstring(
    The weight of the example.
)docstring");

  py::class_<VW::cs_label>(m, "CSLabel")
      .def(py::init(
               [](std::optional<std::vector<std::tuple<float, float>>> costs, bool is_shared)
               {
                 auto label = std::make_unique<VW::cs_label>();
                 if (is_shared)
                 {
                   if (costs.has_value())
                   {
                     throw std::invalid_argument("Shared examples cannot have action, cost, or probability.");
                   }

                   label->costs.emplace_back(-FLT_MAX, 0, 0.f, 0.f);
                   return label;
                 }

                 if (costs.has_value())
                 {
                   for (auto& [class_index, cost] : *costs) { label->costs.emplace_back(cost, class_index, 0.f, 0.f); }
                 }

                 return label;
               }),
          py::kw_only(), py::arg("costs") = py::none(), py::arg("shared") = false, R"docstring(
    A label representing a cost sensitive classification problem.

    Args:
      costs (Optional[List[Tuple[int, float]]]): List of classes and costs. If there is no label, this should be None.
      shared (bool): Whether the example represents the shared context
)docstring")
      .def_property_readonly(
          "shared", [](VW::cs_label& l) -> bool { return is_shared(l); },
          R"docstring(
    Whether the example represents the shared context.
)docstring")
      .def_property(
          "costs",
          [](VW::cs_label& l) -> std::optional<std::vector<std::tuple<uint32_t, float>>>
          {
            if (is_shared(l)) { return std::nullopt; }

            std::vector<std::tuple<uint32_t, float>> costs;
            costs.reserve(l.costs.size());
            for (auto& cost : l.costs) { costs.emplace_back(cost.class_index, cost.x); }
            return costs;
          },
          [](VW::cs_label& l, const std::vector<std::tuple<uint32_t, float>>& label)
          {
            if (is_shared(l)) { throw std::invalid_argument("Shared examples cannot have costs."); }

            l.costs.clear();
            for (auto& [class_index, cost] : label) { l.costs.emplace_back(cost, class_index, 0.f, 0.f); }
          },
          R"docstring(
    The costs for the example. The format of the costs is (class_index, cost).
)docstring");

  py::class_<VW::example, std::shared_ptr<VW::example>>(m, "Example")
      .def(py::init(
          []()
          {
            // shared ptr which returns to the pool upon deletion
            return get_example_from_pool();
          }))
      .def("_is_newline", [](VW::example& ex) -> bool { return ex.is_newline; })
      .def("_get_label",
          [](VW::example& ex, VW::label_type_t label_type)
              -> std::variant<py_simple_label, VW::multiclass_label, VW::cb_label, VW::cs_label, std::monostate>
          {
            switch (label_type)
            {
              case VW::label_type_t::SIMPLE:
              {
                const auto& simple_red_features = ex.ex_reduction_features.get<VW::simple_label_reduction_features>();
                return py_simple_label{ex.l.simple.label, simple_red_features.weight, simple_red_features.initial};
              }
              case VW::label_type_t::MULTICLASS:
                return ex.l.multi;
              case VW::label_type_t::CB:
                return ex.l.cb;
              case VW::label_type_t::CS:
                return ex.l.cs;
              case VW::label_type_t::NOLABEL:
                return std::monostate();
              default:
                throw std::runtime_error("Unsupported label type");
            }
          })
      .def("_set_label",
          [](VW::example& ex,
              const std::variant<py_simple_label*, VW::multiclass_label*, VW::cb_label*, VW::cs_label*, std::monostate>&
                  label) -> void
          {
            std::visit(
                overloaded{
                    // Corresponds to nolabel
                    [](std::monostate) {},
                    [&](py_simple_label* simple_label)
                    {
                      ex.l.simple.label = simple_label->label;
                      ex.ex_reduction_features.get<VW::simple_label_reduction_features>().weight = simple_label->weight;
                      ex.ex_reduction_features.get<VW::simple_label_reduction_features>().initial =
                          simple_label->initial;
                    },
                    [&](VW::multiclass_label* multiclass_label) { ex.l.multi = *multiclass_label; },
                    [&](VW::cb_label* cb_label) { ex.l.cb = *cb_label; },
                    [&](VW::cs_label* cs_label) { ex.l.cs = *cs_label; },
                },
                label);
          })
      .def("_get_tag", [](VW::example& ex) -> std::string { return std::string(ex.tag.data(), ex.tag.size()); });

  py::class_<workspace_with_logger_contexts>(m, "Workspace")
      .def(py::init(
               [](const std::vector<std::string>& args, const std::optional<py::bytes>& bytes,
                   bool record_feature_names, bool record_metrics)
               {
                 auto opts = std::make_unique<VW::config::options_cli>(args);
                 if (record_metrics)
                 {
                   // VW enables metrics by passing a file to write the metrics to.
                   opts->insert("extra_metrics", "THIS_FILE_SHOULD_NOT_EXIST1");
                 }

                 if (record_feature_names)
                 {
                   // We need to ensure hash_inv is enabled during initialize for things to work correctly.
                   // The only way to do that is via an option. We're using this one
                   opts->insert("dump_json_weights_experimental", "THIS_FILE_SHOULD_NOT_EXIST2");
                   opts->insert("dump_json_weights_include_feature_names_experimental", "");
                 }

                 std::unique_ptr<VW::io::reader> model_reader = nullptr;
                 std::string_view bytes_view;
                 if (bytes.has_value())
                 {
                   bytes_view = *bytes;
                   model_reader = VW::io::create_buffer_view(bytes_view.data(), bytes_view.size());
                 }

                 auto wrapped_object = std::make_unique<workspace_with_logger_contexts>();
                 wrapped_object->logger_context_ptr = std::make_unique<logger_context>();
                 py::object get_logger = py::module::import("logging").attr("getLogger");
                 wrapped_object->logger_context_ptr->driver_logger = get_logger("vowpal_wabbit_next.driver");
                 wrapped_object->logger_context_ptr->log_logger = get_logger("vowpal_wabbit_next.log");
                 auto logger = VW::io::create_custom_sink_logger(wrapped_object->logger_context_ptr.get(), log_log);
                 wrapped_object->workspace_ptr =
                     std::shared_ptr<VW::workspace>(VW::initialize_experimental(std::move(opts),
                         std::move(model_reader), driver_log, wrapped_object->logger_context_ptr.get(), &logger));
                 // This should cause parsing failures to be thrown instead of just logged.
                 wrapped_object->workspace_ptr->example_parser->strict_parse = true;

                 // Check for unsupported features.
                 // The main reason for this is we want to remove the concept of "setup_example" in the python bindings
                 // This is achieved by performing necessary steps in the learn/predict call and undoing them on the way
                 // out.
                 if (wrapped_object->workspace_ptr->example_parser->sort_features)
                 {
                   THROW("The command line option 'sort_features' is not supported in py-vowpal-wabbit-next.");
                 }

                 if (wrapped_object->workspace_ptr->ignore_some)
                 {
                   THROW("The command line option 'ignore' is not supported in py-vowpal-wabbit-next.");
                 }

                 if (wrapped_object->workspace_ptr->skip_gram_transformer != nullptr)
                 {
                   THROW("The command line option 'ngram' is not supported in py-vowpal-wabbit-next.");
                 }

                 if (!wrapped_object->workspace_ptr->limit_strings.empty())
                 {
                   THROW("The command line option 'feature_limit' is not supported in py-vowpal-wabbit-next.");
                 }

                 return wrapped_object;
               }),
          py::arg("args"), py::kw_only(), py::arg("model_data") = std::nullopt, py::arg("record_feature_names") = false,
          py::arg("record_metrics") = false)
      .def(
          "learn_one",
          [](workspace_with_logger_contexts& workspace, VW::example& example) -> void
          {
            py_setup_example(*workspace.workspace_ptr, example);
            auto on_exit = VW::scope_exit([&]() { py_unsetup_example(*workspace.workspace_ptr, example); });

            // Learner is used directly as VW makes decisions about training and
            // learn returns prediction in the workspace API and ends up calling
            // potentially the wrong thing.
            auto* learner = VW::LEARNER::require_singleline(workspace.workspace_ptr->l.get());
            assert(learner != nullptr);
            learner->learn(example);

            // TODO - when updating VW submodule if learn calls update stats then remove this to avoid a double call.
            update_stats_recursive(*workspace.workspace_ptr, *learner, example);
          },
          py::arg("examples"), py::kw_only())
      .def(
          "learn_multi_ex_one",
          [](workspace_with_logger_contexts& workspace, std::vector<VW::example*>& example) -> void
          {
            assert(!example.empty());
            py_setup_example(*workspace.workspace_ptr, example);
            auto on_exit = VW::scope_exit([&]() { py_unsetup_example(*workspace.workspace_ptr, example); });

            // Learner is used directly as VW makes decisions about training and
            // learn returns prediction in the workspace API and ends up calling
            // potentially the wrong thing.
            auto* learner = VW::LEARNER::require_multiline(workspace.workspace_ptr->l.get());
            assert(learner != nullptr);
            learner->learn(example);

            // TODO - when updating VW submodule if learn calls update stats then remove this to avoid a double call.
            update_stats_recursive(*workspace.workspace_ptr, *learner, example);
          },
          py::arg("examples"), py::kw_only())
      .def(
          "predict_one",
          [](workspace_with_logger_contexts& workspace, VW::example& example) -> prediction_t
          {
            py_setup_example(*workspace.workspace_ptr, example);
            auto on_exit = VW::scope_exit([&]() { py_unsetup_example(*workspace.workspace_ptr, example); });
            // We must save and restore test_only because the library sets this values and does not undo it.
            bool test_only = example.test_only;

            // Learner is used directly as VW makes decisions about training and
            // learn returns prediction in the workspace API and ends up calling
            // potentially the wrong thing.
            auto* learner = VW::LEARNER::require_singleline(workspace.workspace_ptr->l.get());
            learner->predict(example);

            // TODO - when updating VW submodule if learn calls update stats then remove this to avoid a double call.
            update_stats_recursive(*workspace.workspace_ptr, *learner, example);
            example.test_only = test_only;
            return to_prediction(example.pred, workspace.workspace_ptr->l->get_output_prediction_type());
          },
          py::arg("examples"), py::kw_only())
      .def(
          "predict_multi_ex_one",
          [](workspace_with_logger_contexts& workspace, std::vector<VW::example*>& example) -> prediction_t
          {
            assert(!example.empty());
            py_setup_example(*workspace.workspace_ptr, example);
            auto on_exit = VW::scope_exit([&]() { py_unsetup_example(*workspace.workspace_ptr, example); });
            // We must save and restore test_only because the library sets this values and does not undo it.
            std::vector<bool> test_onlys;
            test_onlys.reserve(example.size());
            for (auto ex : example) { test_onlys.push_back(ex->test_only); }

            // Learner is used directly as VW makes decisions about training and
            // learn returns prediction in the workspace API and ends up calling
            // potentially the wrong thing.
            auto* learner = VW::LEARNER::require_multiline(workspace.workspace_ptr->l.get());
            learner->predict(example);

            // TODO - when updating VW submodule if learn calls update stats then remove this to avoid a double call.
            update_stats_recursive(*workspace.workspace_ptr, *learner, example);
            for (size_t i = 0; i < example.size(); i++) { example[i]->test_only = test_onlys[i]; }
            return to_prediction(example[0]->pred, workspace.workspace_ptr->l->get_output_prediction_type());
          },
          py::arg("examples"), py::kw_only())
      .def(
          "predict_then_learn_one",
          [](workspace_with_logger_contexts& workspace, VW::example& example) -> prediction_t
          {
            py_setup_example(*workspace.workspace_ptr, example);
            auto on_exit = VW::scope_exit([&]() { py_unsetup_example(*workspace.workspace_ptr, example); });

            auto* learner = VW::LEARNER::require_singleline(workspace.workspace_ptr->l.get());
            if (workspace.workspace_ptr->l->learn_returns_prediction)
            {
              // Learner is used directly as VW makes decisions about training and
              // learn returns prediction in the workspace API and ends up calling
              // potentially the wrong thing.
              learner->learn(example);
            }
            else
            {
              // Learner is used directly as VW makes decisions about training and
              // learn returns prediction in the workspace API and ends up calling
              // potentially the wrong thing.
              // We must save and restore test_only because the library sets this values and does not undo it.
              bool test_only = example.test_only;
              learner->predict(example);
              example.test_only = test_only;

              learner->learn(example);
            }

            // TODO - when updating VW submodule if learn calls update stats then remove this to avoid a double call.
            update_stats_recursive(*workspace.workspace_ptr, *learner, example);
            return to_prediction(example.pred, workspace.workspace_ptr->l->get_output_prediction_type());
          },
          py::arg("examples"), py::kw_only())
      .def(
          "predict_then_learn_multi_ex_one",
          [](workspace_with_logger_contexts& workspace, std::vector<VW::example*>& example) -> prediction_t
          {
            py_setup_example(*workspace.workspace_ptr, example);
            auto on_exit = VW::scope_exit([&]() { py_unsetup_example(*workspace.workspace_ptr, example); });
            auto* learner = VW::LEARNER::require_multiline(workspace.workspace_ptr->l.get());
            if (workspace.workspace_ptr->l->learn_returns_prediction)
            {
              // Learner is used directly as VW makes decisions about training and
              // learn returns prediction in the workspace API and ends up calling
              // potentially the wrong thing.
              learner->learn(example);
            }
            else
            {
              // Learner is used directly as VW makes decisions about training and
              // learn returns prediction in the workspace API and ends up calling
              // potentially the wrong thing.
              // We must save and restore test_only because the library sets this values and does not undo it.
              std::vector<bool> test_onlys;
              test_onlys.reserve(example.size());
              for (auto ex : example) { test_onlys.push_back(ex->test_only); }
              learner->predict(example);
              for (size_t i = 0; i < example.size(); i++) { example[i]->test_only = test_onlys[i]; }

              learner->learn(example);
            }

            // TODO - when updating VW submodule if learn calls update stats then remove this to avoid a double call.
            update_stats_recursive(*workspace.workspace_ptr, *learner, example);
            return to_prediction(example[0]->pred, workspace.workspace_ptr->l->get_output_prediction_type());
          },
          py::arg("examples"), py::kw_only())
      .def("get_is_multiline",
          [](const workspace_with_logger_contexts& workspace) { return workspace.workspace_ptr->l->is_multiline(); })
      .def("get_metrics",
          [](const workspace_with_logger_contexts& workspace) -> py::dict
          {
            if (!workspace.workspace_ptr->global_metrics.are_metrics_enabled())
            {
              throw std::runtime_error(
                  "Metrics are not enabled. Pass records_metrics=True to Workspace constructor to enable.");
            }
            py::dict metrics;
            auto collected_metrics =
                workspace.workspace_ptr->global_metrics.collect_metrics(workspace.workspace_ptr->l.get());
            return convert_metrics_to_dict(collected_metrics);
          })
      .def("get_prediction_type",
          [](const workspace_with_logger_contexts& workspace)
          { return workspace.workspace_ptr->l->get_output_prediction_type(); })
      .def("get_label_type",
          [](const workspace_with_logger_contexts& workspace) -> VW::label_type_t
          { return workspace.workspace_ptr->l->get_input_label_type(); })
      .def("serialize",
          [](const workspace_with_logger_contexts& workspace) -> py::bytes
          {
            auto backing_vector = std::make_shared<std::vector<char>>();
            VW::io_buf io_writer;
            io_writer.add_file(VW::io::create_vector_writer(backing_vector));
            VW::save_predictor(*workspace.workspace_ptr, io_writer);
            io_writer.flush();
            return py::bytes(backing_vector->data(), backing_vector->size());  // Return the data without transcoding
          })
      .def(
          "get_index_for_scalar_feature",
          [](const workspace_with_logger_contexts& workspace, std::string_view feature_name,
              std::optional<std::string_view> feature_value, std::string_view namespace_name) -> uint64_t
          {
            auto& ws = *workspace.workspace_ptr;

            const auto ns_hash = ws.example_parser->hasher(namespace_name.data(), namespace_name.size(), ws.hash_seed);
            const auto feature_hash = ws.example_parser->hasher(feature_name.data(), feature_name.size(), ns_hash);
            uint32_t raw_index = 0;
            if (feature_value.has_value())
            {
              raw_index =
                  ws.example_parser->hasher(feature_value.value().data(), feature_value.value().size(), feature_hash);
            }
            else { raw_index = feature_hash; }

            // Apply parse mask.
            raw_index = raw_index & ws.parse_mask;

            // Now we need to handle if the multiplier were to cause truncation.
            const auto weight_mask = ws.weights.mask();
            const auto multiplier = static_cast<uint64_t>(ws.wpp) << static_cast<uint64_t>(ws.weights.stride_shift());

            // We essentially do what setup_example does by expanding the weight space then masking based on the weight
            // mask and then undo the multiplier.
            const auto final_index = ((static_cast<uint64_t>(raw_index) * multiplier) & weight_mask) / multiplier;
            return final_index;
          },
          py::arg("feature_name"), py::arg("feature_value") = std::nullopt, py::arg("namespace_name") = " ")
      .def("weights",
          [](const workspace_with_logger_contexts& workspace) -> std::unique_ptr<dense_weight_holder>
          {
            if (workspace.workspace_ptr->weights.sparse) { THROW("weights are sparse, cannot return dense weights"); }
            return std::make_unique<dense_weight_holder>(
                &workspace.workspace_ptr->weights.dense_weights, workspace.workspace_ptr->wpp, workspace.workspace_ptr);
          })
      .def(
          "json_weights",
          [](const workspace_with_logger_contexts& workspace, bool include_feature_names,
              bool include_online_state) -> std::string
          {
            // Invert hash is enabled with "--invert_hash"
            auto old_dump_json_weights_include_feature_names =
                workspace.workspace_ptr->dump_json_weights_include_feature_names;
            workspace.workspace_ptr->dump_json_weights_include_feature_names = include_feature_names;
            auto old_dump_json_weights_include_extra_online_state =
                workspace.workspace_ptr->dump_json_weights_include_extra_online_state;
            workspace.workspace_ptr->dump_json_weights_include_extra_online_state = include_online_state;
            auto on_exit = VW::scope_exit(
                [&]()
                {
                  workspace.workspace_ptr->dump_json_weights_include_feature_names =
                      old_dump_json_weights_include_feature_names;
                  workspace.workspace_ptr->dump_json_weights_include_extra_online_state =
                      old_dump_json_weights_include_extra_online_state;
                });
            return workspace.workspace_ptr->dump_weights_to_json_experimental();
          },
          py::kw_only(), py::arg("include_feature_names") = false, py::arg("include_online_state") = false)
      .def(
          "readable_model",
          [](const workspace_with_logger_contexts& workspace, bool include_feature_names) -> std::string
          {
            auto& all = *workspace.workspace_ptr;
            if (include_feature_names)
            {
              if (!all.hash_inv)
              {
                THROW(
                    "record_feature_names must be enabled (from Workspace constructor) to use "
                    "include_feature_names=True");
              }
            }

            auto print_invert_guard = VW::swap_guard(all.print_invert, include_feature_names);
            VW::io_buf buffer;
            auto vec_buffer = std::make_shared<std::vector<char>>();
            buffer.add_file(VW::io::create_vector_writer(vec_buffer));
            VW::details::dump_regressor(all, buffer, true);
            buffer.flush();
            return std::string(vec_buffer->data(), vec_buffer->size());
          },
          py::kw_only(), py::arg("include_feature_names") = false);

  m.def("_parse_line_text", &::parse_text_line, py::arg("workspace"), py::arg("line"));
  m.def("_parse_line_dsjson", &::parse_dsjson_line, py::arg("workspace"), py::arg("line"));
  m.def("_parse_line_json", &::parse_json_line, py::arg("workspace"), py::arg("line"));
  m.def("_write_cache_header", &::write_cache_header, py::arg("workspace"), py::arg("file"));
  m.def("_write_cache_example", &::write_cache_example, py::arg("workspace"), py::arg("example"), py::arg("file"));
  m.def("_run_cli_driver", &::run_cli_driver, py::arg("args"), py::kw_only(), py::arg("onethread") = false);

  py::class_<cache_reader>(m, "_CacheReader")
      .def(py::init([](workspace_with_logger_contexts& workspace, py::object file)
          { return std::make_unique<cache_reader>(workspace.workspace_ptr, file); }))
      .def("_get_next",
          [](cache_reader& reader) -> std::optional<std::shared_ptr<VW::example>>
          {
            auto next_example = reader.read_cache_example();
            if (next_example == nullptr) { return std::nullopt; }
            return next_example;
          });

  py::class_<VW::model_delta>(m, "ModelDelta")
      .def(py::init(
               [](const py::bytes& bytes)
               {
                 std::string_view bytes_view = bytes;
                 auto model_reader = VW::io::create_buffer_view(bytes_view.data(), bytes_view.size());
                 return VW::model_delta::deserialize(*model_reader);
                 ;
               }),
          py::arg("model_data"))
      .def("serialize",
          [](const VW::model_delta& delta) -> py::bytes
          {
            auto backing_vector = std::make_shared<std::vector<char>>();
            VW::io_buf io_writer;
            auto writer = VW::io::create_vector_writer(backing_vector);
            delta.serialize(*writer);
            return py::bytes(backing_vector->data(), backing_vector->size());  // Return the data without transcoding
          });

  m.def("_merge_deltas", &::merge_deltas, py::arg("deltas"));
  m.def("_calculate_delta", &::calculate_delta, py::arg("base_workspace"), py::arg("derived_workspace"));
  m.def("_apply_delta", &::apply_delta, py::arg("base_workspace"), py::arg("delta"));

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  m.attr("_vw_version") = VW::VERSION.to_string();
  m.attr("_vw_commit") = VW::GIT_COMMIT;
}

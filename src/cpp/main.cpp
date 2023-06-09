#include "debug_reduction.h"
#include "label.h"
#include "prediction.h"
#include "vw/common/text_utils.h"
#include "vw/config/options_cli.h"
#include "vw/core/array_parameters.h"
#include "vw/core/array_parameters_dense.h"
#include "vw/core/cache.h"
#include "vw/core/cb.h"
#include "vw/core/ccb_label.h"
#include "vw/core/ccb_reduction_features.h"
#include "vw/core/constant.h"
#include "vw/core/cost_sensitive.h"
#include "vw/core/debug_print.h"
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
#include "vw/core/reduction_stack.h"
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

#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>
#include <optional>
#include <variant>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
namespace
{

template <class... Ts>
struct overloaded : Ts...
{
  using Ts::operator()...;
};

template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

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
  bool debug;
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
    if (workspace.workspace_ptr->output_config.audit || workspace.workspace_ptr->output_config.hash_inv)
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

    if (workspace.workspace_ptr->output_config.audit || workspace.workspace_ptr->output_config.hash_inv)
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
  writer.write(reinterpret_cast<const char*>(&workspace.workspace_ptr->initial_weights_config.num_bits),
      sizeof(workspace.workspace_ptr->initial_weights_config.num_bits));
}

void write_cache_example(workspace_with_logger_contexts& workspace, VW::example& ex, py::object file)
{
  VW::parsers::cache::details::cache_temp_buffer temp_buffer;
  VW::io_buf output;
  output.add_file(VW::make_unique<python_writer>(file));
  VW::parsers::cache::write_example_to_cache(output, &ex,
      workspace.workspace_ptr->parser_runtime.example_parser->lbl_parser,
      workspace.workspace_ptr->runtime_state.parse_mask, temp_buffer);
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

std::shared_ptr<vwpy::debug_node> get_and_clear_debug_info(workspace_with_logger_contexts& workspace)
{
  assert(workspace.debug);
  auto debug_info = dynamic_cast<vwpy::debug_data_stash*>(workspace.workspace_ptr->parser_runtime.custom_parser.get());
  auto root = debug_info->shared_debug_state->root;
  debug_info->shared_debug_state->active = std::stack<std::shared_ptr<vwpy::debug_node>>{};
  debug_info->shared_debug_state->root = nullptr;
  return root;
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

  // If we hit this point, there was no update stats but other funcs were
  // defined so we should not forward. We log an error since this is probably an
  // issue.
  if (has_at_least_one_new_style_func)
  {
    workspace.logger.error(
        "No update_stats function was registered for a reduction but other finalization functions were. This is likely "
        "an issue with the reduction: '{}'. Please report this issue to the VW team.",
        learner.get_name());
    return;
  }

  // Recurse until we find a reduction with an update_stats function.
  auto* base = learner.get_base_learner();
  if (base != nullptr)
  {
    if (learner.is_multiline() != base->is_multiline())
    {
      THROW("Cannot forward update_stats call across multiline/singleline boundary.");
    }

    update_stats_recursive(workspace, *base, example);
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

  ex.weight = ws.parser_runtime.example_parser->lbl_parser.get_weight(ex.l, ex.ex_reduction_features);

  if (ws.feature_tweaks_config.add_constant)
  {
    // TODO make workspace a const arg here.
    VW::add_constant_feature(ws, &ex);
  }

  uint64_t multiplier = static_cast<uint64_t>(ws.reduction_state.total_feature_width) << ws.weights.stride_shift();

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
  ex.interactions = &ws.feature_tweaks_config.interactions;
  ex.extent_interactions = &ws.feature_tweaks_config.extent_interactions;
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

  if (ws.feature_tweaks_config.add_constant)
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

  uint32_t multiplier = ws.reduction_state.total_feature_width << ws.weights.stride_shift();
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

// Because of the GIL we can use globals here.
static bool SIGINT_CALLED = false;
static VW::workspace* CLI_DRIVER_WORKSPACE = nullptr;

// return type is an optional error information (nullopt if success), driver output, list of log messages
// stdin is not supported
std::tuple<std::optional<std::string>, std::string, std::vector<std::string>> run_cli_driver(
    const std::vector<std::string>& args, bool onethread)
{
  SIGINT_CALLED = false;
  CLI_DRIVER_WORKSPACE = nullptr;
  std::signal(SIGINT,
      [](int)
      {
        if (CLI_DRIVER_WORKSPACE != nullptr) { VW::details::set_done(*CLI_DRIVER_WORKSPACE); }
        SIGINT_CALLED = true;
      });

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
    auto all = VW::initialize_experimental(std::move(options), nullptr, driver_logger, &driver_log, &logger);
    all->runtime_config.vw_is_main = true;
    CLI_DRIVER_WORKSPACE = all.get();

    // If sigint was called before we got here, we should avoid running the driver.
    if (!SIGINT_CALLED)
    {
      if (onethread) { VW::LEARNER::generic_driver_onethread(*all); }
      else
      {
        VW::start_parser(*all);
        VW::LEARNER::generic_driver(*all);
        VW::end_parser(*all);
      }

      if (all->parser_runtime.example_parser->exc_ptr)
      {
        std::rethrow_exception(all->parser_runtime.example_parser->exc_ptr);
      }
      VW::sync_stats(*all);
      all->finish();
    }
  }
  catch (const std::exception& ex)
  {
    return std::make_tuple(ex.what(), driver_log.str(), log_log);
  }
  catch (...)
  {
    return std::make_tuple("Unknown exception occurred", driver_log.str(), log_log);
  }

  SIGINT_CALLED = false;
  CLI_DRIVER_WORKSPACE = nullptr;
  return std::make_tuple(std::nullopt, driver_log.str(), log_log);
}

struct dense_weight_holder
{
  dense_weight_holder(VW::dense_parameters* weights, size_t total_feature_width, std::shared_ptr<VW::workspace> ws)
      : weights(weights), total_feature_width(total_feature_width), ws(ws)
  {
  }

  VW::dense_parameters* weights;
  size_t total_feature_width;
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

bool is_shared(const VW::cs_label& label)
{
  const auto& costs = label.costs;
  if (costs.size() != 1) { return false; }
  if (costs[0].class_index != 0) { return false; }
  if (costs[0].x != -FLT_MAX) { return false; }
  return true;
}

// TODO: create a version of this that can be used in learn that doesn't involve
// copying the prediction and then not using the value.
std::variant<vwpy::prediction_t, std::tuple<vwpy::prediction_t, std::vector<std::shared_ptr<vwpy::debug_node>>>>
predict_then_learn(workspace_with_logger_contexts& workspace, VW::example& example)
{
  py_setup_example(*workspace.workspace_ptr, example);
  auto on_exit = VW::scope_exit([&]() { py_unsetup_example(*workspace.workspace_ptr, example); });

  auto* learner = VW::LEARNER::require_singleline(workspace.workspace_ptr->l.get());
  std::vector<std::shared_ptr<vwpy::debug_node>> debug_info;
  if (workspace.workspace_ptr->l->learn_returns_prediction)
  {
    // Learner is used directly as VW makes decisions about training and
    // learn returns prediction in the workspace API and ends up calling
    // potentially the wrong thing.
    learner->learn(example);
    if (workspace.debug) { debug_info.push_back(get_and_clear_debug_info(workspace)); }
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
    if (workspace.debug) { debug_info.push_back(get_and_clear_debug_info(workspace)); }

    learner->learn(example);
    if (workspace.debug) { debug_info.push_back(get_and_clear_debug_info(workspace)); }
  }

  // TODO - when updating VW submodule if learn calls update stats then remove this to avoid a double call.
  update_stats_recursive(*workspace.workspace_ptr, *learner, example);
  auto prediction = vwpy::to_prediction(example.pred, workspace.workspace_ptr->l->get_output_prediction_type());
  if (workspace.debug) { return std::make_tuple(prediction, debug_info); }
  return prediction;
}

std::variant<vwpy::prediction_t, std::tuple<vwpy::prediction_t, std::vector<std::shared_ptr<vwpy::debug_node>>>>
predict_then_learn(workspace_with_logger_contexts& workspace, std::vector<VW::example*>& example)
{
  py_setup_example(*workspace.workspace_ptr, example);
  auto on_exit = VW::scope_exit([&]() { py_unsetup_example(*workspace.workspace_ptr, example); });
  auto* learner = VW::LEARNER::require_multiline(workspace.workspace_ptr->l.get());
  std::vector<std::shared_ptr<vwpy::debug_node>> debug_info;
  if (workspace.workspace_ptr->l->learn_returns_prediction)
  {
    // Learner is used directly as VW makes decisions about training and
    // learn returns prediction in the workspace API and ends up calling
    // potentially the wrong thing.
    learner->learn(example);
    if (workspace.debug) { debug_info.push_back(get_and_clear_debug_info(workspace)); }
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
    if (workspace.debug) { debug_info.push_back(get_and_clear_debug_info(workspace)); }
    learner->learn(example);
    if (workspace.debug) { debug_info.push_back(get_and_clear_debug_info(workspace)); }
  }

  // TODO - when updating VW submodule if learn calls update stats then remove this to avoid a double call.
  update_stats_recursive(*workspace.workspace_ptr, *learner, example);
  auto prediction = vwpy::to_prediction(example[0]->pred, workspace.workspace_ptr->l->get_output_prediction_type());
  if (workspace.debug) { return std::make_tuple(prediction, debug_info); }
  return prediction;
}

size_t count_non_zero_weights(const VW::parameters& weights)
{
  if (weights.sparse)
  {
    return std::count_if(weights.sparse_weights.cbegin(), weights.sparse_weights.cend(), [](const float& w) { return w != 0.f; });
  }
  else
  {
    return std::count_if(weights.dense_weights.cbegin(), weights.dense_weights.cend(), [](const float& w) { return w != 0.f; });
  }
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
                {static_cast<ssize_t>(length), static_cast<ssize_t>(m.total_feature_width),
                    static_cast<ssize_t>(m.weights->stride())}, /* Buffer dimensions */
                {sizeof(float) * static_cast<ssize_t>(m.total_feature_width) *
                        static_cast<ssize_t>(m.weights->stride()),
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

  py::enum_<VW::ccb_example_type>(m, "CCBExampleType")
      .value("Unset", VW::ccb_example_type::UNSET)
      .value("Shared", VW::ccb_example_type::SHARED)
      .value("Action", VW::ccb_example_type::ACTION)
      .value("Slot", VW::ccb_example_type::SLOT);

  py::class_<vwpy::py_simple_label>(m, "SimpleLabel")
      .def(py::init(
               [](float label, float weight, float initial)
               {
                 vwpy::py_simple_label l;
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
      .def_readwrite("label", &vwpy::py_simple_label::label, R"docstring(
    The label.
)docstring")
      .def_readwrite("weight", &vwpy::py_simple_label::weight, R"docstring(
    The weight of this label.
)docstring")
      .def_readwrite("initial", &vwpy::py_simple_label::initial, R"docstring(
    The initial value of the prediction.
)docstring")
      .def("__repr__",
          [](const vwpy::py_simple_label& l)
          {
            std::stringstream ss;
            ss << "SimpleLabel(label=" << l.label << ", weight=" << l.weight << ", initial=" << l.initial << ")";
            return ss.str();
          });

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
)docstring")
      .def("__repr__",
          [](VW::cb_label& l)
          {
            std::stringstream ss;
            ss << "CBLabel(";
            if (l.costs.size() == 1 && l.costs[0].probability == -1.f) { ss << "shared=True"; }
            else
            {
              if (l.costs.size() == 0) { ss << "label=None"; }
              else
              {
                ss << "label=(" << l.costs[0].action << ", " << l.costs[0].cost << ", " << l.costs[0].probability
                   << ")";
              }
            }
            ss << ", weight=" << l.weight << ")";
            return ss.str();
          });

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
)docstring")
      .def("__repr__",
          [](VW::cs_label& l)
          {
            std::stringstream ss;
            ss << "CSLabel(";
            if (is_shared(l)) { ss << "shared=True"; }
            else
            {
              ss << "costs=[";
              for (size_t i = 0; i < l.costs.size(); i++)
              {
                if (i != 0) { ss << ", "; }
                ss << "(" << l.costs[i].class_index << ", " << l.costs[i].x << ")";
              }
              ss << "]";
            }
            ss << ")";
            return ss.str();
          });

  py::class_<VW::ccb_label>(m, "CCBLabel")
      .def(
          py::init(
              [](VW::ccb_example_type type,
                  const std::optional<std::tuple<float, std::vector<std::tuple<uint32_t, float>>>>& outcome,
                  const std::optional<std::vector<uint32_t>>& explicit_included_actions)
              {
                auto label = std::make_unique<VW::ccb_label>();
                label->type = type;
                if (type == VW::ccb_example_type::UNSET) { throw std::invalid_argument("CCBLabel must have a type."); }

                if (type == VW::ccb_example_type::SHARED)
                {
                  if (outcome.has_value() || explicit_included_actions.has_value())
                  {
                    throw std::invalid_argument("Shared examples cannot have an outcome or explicit_included_actions.");
                  }
                  return label;
                }

                if (type == VW::ccb_example_type::ACTION)
                {
                  if (outcome.has_value() || explicit_included_actions.has_value())
                  {
                    throw std::invalid_argument("Action examples cannot have an outcome or explicit_included_actions.");
                  }
                  return label;
                }

                assert(type == VW::ccb_example_type::SLOT);

                if (explicit_included_actions.has_value())
                {
                  const auto& actions = *explicit_included_actions;
                  label->explicit_included_actions.reserve(actions.size());
                  for (auto action : actions) { label->explicit_included_actions.push_back(action); }
                }

                if (outcome.has_value())
                {
                  const auto& [cost, action_probs] = *outcome;
                  label->outcome = new VW::ccb_outcome();
                  label->outcome->cost = cost;
                  label->outcome->probabilities.reserve(action_probs.size());
                  for (auto& [action, probability] : action_probs)
                  {
                    label->outcome->probabilities.push_back({action, probability});
                  }
                }

                return label;
              }),
          py::arg("type"), py::kw_only(), py::arg("outcome") = py::none(),
          py::arg("explicit_included_actions") = py::none(), R"docstring(
    A label representing a conditional contextual bandit problem.

    Args:
      type (CCBExampleType): The type of the example. Unset is invalid.
      outcome (Optional[Tuple[float, List[Tuple[int, float]]]]): The outcome of the example. The format of the outcome is (cost, [(action, probability)]).
      explicit_included_actions (Optional[List[int]]): The list of actions explicitly included from the slot.
)docstring")
      .def_property_readonly(
          "example_type", [](VW::ccb_label& l) -> VW::ccb_example_type { return l.type; },
          R"docstring(
    Whether the example represents the shared context.
)docstring")
      .def_property_readonly(
          "outcome",
          [](VW::ccb_label& l) -> std::optional<std::tuple<float, std::vector<std::tuple<uint32_t, float>>>>
          {
            if (l.type != VW::ccb_example_type::SLOT) { return std::nullopt; }

            if (l.outcome == nullptr) { return std::nullopt; }

            std::vector<std::tuple<uint32_t, float>> action_probs;
            action_probs.reserve(l.outcome->probabilities.size());
            for (auto& action_prob : l.outcome->probabilities)
            {
              action_probs.emplace_back(action_prob.action, action_prob.score);
            }
            return std::make_tuple(l.outcome->cost, action_probs);
          },
          R"docstring(
    The outcome of the example. The format of the outcome is (cost, [(action, probability)]).
)docstring")
      .def_property_readonly(
          "explicit_included_actions",
          [](VW::ccb_label& l) -> std::optional<std::vector<uint32_t>>
          {
            if (l.type != VW::ccb_example_type::SLOT) { return std::nullopt; }

            if (l.explicit_included_actions.empty()) { return std::nullopt; }

            std::vector<uint32_t> actions;
            actions.reserve(l.explicit_included_actions.size());
            for (auto action : l.explicit_included_actions) { actions.push_back(action); }
            return actions;
          },
          R"docstring(
    The list of actions explicitly excluded from the slot.
)docstring")
      .def("__repr__",
          [](VW::ccb_label& l)
          {
            std::stringstream ss;
            ss << "CCBLabel(";
            if (l.type == VW::ccb_example_type::SHARED) { ss << "type=CCBExampleType.SHARED"; }
            else if (l.type == VW::ccb_example_type::ACTION) { ss << "type=CCBExampleType.ACTION"; }
            else
            {
              ss << "type=CCBExampleType.SLOT";
              if (l.outcome != nullptr)
              {
                ss << ", outcome=(" << l.outcome->cost << ", [";
                for (size_t i = 0; i < l.outcome->probabilities.size(); i++)
                {
                  if (i != 0) { ss << ", "; }
                  ss << "(" << l.outcome->probabilities[i].action << ", " << l.outcome->probabilities[i].score << ")";
                }
                ss << "])";
              }
              if (!l.explicit_included_actions.empty())
              {
                ss << ", explicit_included_actions=[";
                for (size_t i = 0; i < l.explicit_included_actions.size(); i++)
                {
                  if (i != 0) { ss << ", "; }
                  ss << l.explicit_included_actions[i];
                }
                ss << "]";
              }
            }
            ss << ")";
            return ss.str();
          });

  py::class_<vwpy::debug_node, std::shared_ptr<vwpy::debug_node>>(m, "DebugNode", R"docstring(
    A node in the computation tree of a single learn/predict call. This represents the state of the example as it is entering a given reduction.

    .. warning::
      This is a highly experimental feature.

)docstring")
      .def_property_readonly(
          "children",
          [](const vwpy::debug_node& d) -> std::vector<std::shared_ptr<vwpy::debug_node>> { return d.children; },
          "The child computations that this node processed. This represents traversal of the stack.")
      .def_property_readonly(
          "name", [](const vwpy::debug_node& d) -> std::string { return d.name; },
          "Name of the reduction being called.")
      .def_property_readonly(
          "output_prediction", [](const vwpy::debug_node& d) -> vwpy::prediction_t { return d.output_prediction; },
          "The prediction that this reduction produced.")
      .def_property_readonly(
          "input_labels",
          [](const vwpy::debug_node& d) -> std::variant<vwpy::label_variant_t, std::vector<vwpy::label_variant_t>>
          { return d.input_labels; },
          "The label that was passed into this reduction. Or, list of labels if this reduction is a multi-example "
          "reduction.")
      .def_property_readonly(
          "interactions",
          [](const vwpy::debug_node& d) -> std::variant<std::vector<std::string>, std::vector<std::vector<std::string>>>
          { return d.interactions; },
          "The interactions that were used to generate the features for this reduction. Or, list of interactions if "
          "this reduction is a multi-example reduction.")
      .def_property_readonly(
          "function", [](const vwpy::debug_node& d) -> std::string { return d.function; },
          "The function that was called on this reduction. Either 'learn' or 'predict'.")
      .def_property_readonly(
          "is_multiline", [](const vwpy::debug_node& d) -> bool { return d.is_multiline; },
          "Whether this reduction is a multi-example reduction.")
      .def_property_readonly(
          "num_examples", [](const vwpy::debug_node& d) -> size_t { return d.num_examples; },
          "The number of examples that were processed by this reduction. This is always 1 for single example "
          "reductions.")
      .def_property_readonly(
          "weight", [](const vwpy::debug_node& d) -> std::variant<float, std::vector<float>> { return d.weight; },
          "The weight of the example. Or, list of weights if this reduction is a multi-example reduction.")
      .def_property_readonly(
          "updated_prediction",
          [](const vwpy::debug_node& d) -> std::variant<float, std::vector<float>> { return d.updated_prediction; },
          "The partial prediction on the example after this reduction ran. Or, list of partial predictions if this "
          "reduction is a multi-example reduction. This is generally only set by the bottom of the stack.")
      .def_property_readonly(
          "partial_prediction",
          [](const vwpy::debug_node& d) -> std::variant<float, std::vector<float>> { return d.partial_prediction; },
          "The partial prediction on the example after this reduction ran. Or, list of partial predictions if this "
          "reduction is a multi-example reduction. This is generally only set by the bottom of the stack.")
      .def_property_readonly(
          "offset", [](const vwpy::debug_node& d) -> std::variant<uint64_t, std::vector<uint64_t>> { return d.offset; },
          "The offset of the example. Or, list of offsets if this reduction is a multi-example reduction. This also "
          "includes the stride of the bottom learner.")
      .def_property_readonly(
          "self_duration_ns",
          [](const vwpy::debug_node& d) -> size_t
          { return std::chrono::duration_cast<std::chrono::nanoseconds>(d.calc_self_time()).count(); },
          "The duration of this reduction in nanoseconds. It does not include time it takes to call children.")
      .def_property_readonly(
          "total_duration_ns",
          [](const vwpy::debug_node& d) -> size_t
          { return std::chrono::duration_cast<std::chrono::nanoseconds>(d.calc_overall_time()).count(); },
          "The duration of this reduction and all children in nanoseconds.");

  py::class_<VW::example, std::shared_ptr<VW::example>>(m, "Example")
      .def(py::init(
          []()
          {
            // shared ptr which returns to the pool upon deletion
            return get_example_from_pool();
          }))
      .def("_is_newline", [](VW::example& ex) -> bool { return ex.is_newline; })
      .def("_get_label",
          [](VW::example& ex, VW::label_type_t label_type) -> vwpy::label_variant_t
          { return vwpy::to_label_variant(ex, label_type); })
      .def("_set_label",
          [](VW::example& ex, const vwpy::label_variant_ptrs_t& label) -> void { vwpy::from_label_variant(ex, label); })
      .def("_get_tag", [](VW::example& ex) -> std::string { return std::string(ex.tag.data(), ex.tag.size()); });

  py::class_<workspace_with_logger_contexts>(m, "Workspace")
      .def(py::init(
               [](const std::vector<std::string>& args, const std::optional<py::bytes>& bytes,
                   bool record_feature_names, bool record_metrics, bool debug)
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

                 std::unique_ptr<vwpy::debug_stack_builder> stack = nullptr;
                 if (debug)
                 {
                   wrapped_object->debug = true;
                   stack = std::make_unique<vwpy::debug_stack_builder>();
                 }
                 wrapped_object->workspace_ptr = std::shared_ptr<VW::workspace>(
                     VW::initialize_experimental(std::move(opts), std::move(model_reader), driver_log,
                         wrapped_object->logger_context_ptr.get(), &logger, std::move(stack)));
                 // This should cause parsing failures to be thrown instead of just logged.
                 wrapped_object->workspace_ptr->parser_runtime.example_parser->strict_parse = true;

                 // Check for unsupported features.
                 // The main reason for this is we want to remove the concept of "setup_example" in the python bindings
                 // This is achieved by performing necessary steps in the learn/predict call and undoing them on the way
                 // out.
                 if (wrapped_object->workspace_ptr->parser_runtime.example_parser->sort_features)
                 {
                   THROW("The command line option 'sort_features' is not supported in py-vowpal-wabbit-next.");
                 }

                 if (wrapped_object->workspace_ptr->feature_tweaks_config.ignore_some)
                 {
                   THROW("The command line option 'ignore' is not supported in py-vowpal-wabbit-next.");
                 }

                 if (wrapped_object->workspace_ptr->feature_tweaks_config.skip_gram_transformer != nullptr)
                 {
                   THROW("The command line option 'ngram' is not supported in py-vowpal-wabbit-next.");
                 }

                 if (!wrapped_object->workspace_ptr->feature_tweaks_config.limit_strings.empty())
                 {
                   THROW("The command line option 'feature_limit' is not supported in py-vowpal-wabbit-next.");
                 }

                 return wrapped_object;
               }),
          py::arg("args"), py::kw_only(), py::arg("model_data") = std::nullopt, py::arg("record_feature_names") = false,
          py::arg("record_metrics") = false, py::arg("debug") = false)
      .def(
          "learn_one",
          [](workspace_with_logger_contexts& workspace,
              VW::example& example) -> std::variant<std::monostate, std::vector<std::shared_ptr<vwpy::debug_node>>>
          {
            // If debug then we need to get out the debug info otherwise we can ignore the result.
            if (workspace.debug) { return std::get<1>(std::get<1>(predict_then_learn(workspace, example))); }
            else
            {
              predict_then_learn(workspace, example);
              return std::monostate{};
            }
          },
          py::arg("examples"), py::kw_only())
      .def(
          "learn_multi_ex_one",
          [](workspace_with_logger_contexts& workspace, std::vector<VW::example*>& example)
              -> std::variant<std::monostate, std::vector<std::shared_ptr<vwpy::debug_node>>>
          {
            assert(!example.empty());
            // If debug then we need to get out the debug info otherwise we can ignore the result.
            if (workspace.debug) { return std::get<1>(std::get<1>(predict_then_learn(workspace, example))); }
            else
            {
              predict_then_learn(workspace, example);
              return std::monostate{};
            }
          },
          py::arg("examples"), py::kw_only())
      .def(
          "predict_one",
          [](workspace_with_logger_contexts& workspace, VW::example& example)
              -> std::variant<vwpy::prediction_t, std::tuple<vwpy::prediction_t, std::shared_ptr<vwpy::debug_node>>>
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
            auto prediction =
                vwpy::to_prediction(example.pred, workspace.workspace_ptr->l->get_output_prediction_type());
            if (workspace.debug)
            {
              auto debug_info = get_and_clear_debug_info(workspace);
              return std::make_tuple(prediction, debug_info);
            }
            return prediction;
          },
          py::arg("examples"), py::kw_only())
      .def(
          "predict_multi_ex_one",
          [](workspace_with_logger_contexts& workspace, std::vector<VW::example*>& example)
              -> std::variant<vwpy::prediction_t, std::tuple<vwpy::prediction_t, std::shared_ptr<vwpy::debug_node>>>
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

            auto prediction =
                vwpy::to_prediction(example[0]->pred, workspace.workspace_ptr->l->get_output_prediction_type());
            if (workspace.debug)
            {
              auto debug_info = get_and_clear_debug_info(workspace);
              return std::make_tuple(prediction, debug_info);
            }
            return prediction;
          },
          py::arg("examples"), py::kw_only())
      .def(
          "predict_then_learn_one",
          [](workspace_with_logger_contexts& workspace,
              VW::example& example) -> std::variant<vwpy::prediction_t,
                                        std::tuple<vwpy::prediction_t, std::vector<std::shared_ptr<vwpy::debug_node>>>>
          { return predict_then_learn(workspace, example); },
          py::arg("examples"), py::kw_only())
      .def(
          "predict_then_learn_multi_ex_one",
          [](workspace_with_logger_contexts& workspace, std::vector<VW::example*>& example)
              -> std::variant<vwpy::prediction_t,
                  std::tuple<vwpy::prediction_t, std::vector<std::shared_ptr<vwpy::debug_node>>>>
          { return predict_then_learn(workspace, example); },
          py::arg("examples"), py::kw_only())
      .def("end_pass",
          [](workspace_with_logger_contexts& workspace)
          {
            workspace.workspace_ptr->passes_config.current_pass++;
            workspace.workspace_ptr->l->end_pass();
          })
      .def("get_is_multiline",
          [](const workspace_with_logger_contexts& workspace) { return workspace.workspace_ptr->l->is_multiline(); })
      .def("get_metrics",
          [](const workspace_with_logger_contexts& workspace) -> py::dict
          {
            if (!workspace.workspace_ptr->output_runtime.global_metrics.are_metrics_enabled())
            {
              throw std::runtime_error(
                  "Metrics are not enabled. Pass records_metrics=True to Workspace constructor to enable.");
            }
            py::dict metrics;
            auto collected_metrics = workspace.workspace_ptr->output_runtime.global_metrics.collect_metrics(
                workspace.workspace_ptr->l.get());
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
            // Determine size estimate by counting non-zero weights.
            const auto non_zero_weights = count_non_zero_weights(workspace.workspace_ptr->weights);
            const auto size_estimate_for_weights = non_zero_weights * sizeof(float) * workspace.workspace_ptr->weights.stride();
            const auto size_estimate_overall = size_estimate_for_weights + 1024; // Add 1KB for other info
            // Best effort reserve of likely final size to avoid reallocations.
            backing_vector->reserve(size_estimate_overall);

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

            const auto ns_hash = ws.parser_runtime.example_parser->hasher(
                namespace_name.data(), namespace_name.size(), ws.runtime_config.hash_seed);
            const auto feature_hash =
                ws.parser_runtime.example_parser->hasher(feature_name.data(), feature_name.size(), ns_hash);
            uint32_t raw_index = 0;
            if (feature_value.has_value())
            {
              raw_index = ws.parser_runtime.example_parser->hasher(
                  feature_value.value().data(), feature_value.value().size(), feature_hash);
            }
            else { raw_index = feature_hash; }

            // Apply parse mask.
            raw_index = raw_index & ws.runtime_state.parse_mask;

            // Now we need to handle if the multiplier were to cause truncation.
            const auto weight_mask = ws.weights.mask();
            const auto multiplier = static_cast<uint64_t>(ws.reduction_state.total_feature_width)
                << static_cast<uint64_t>(ws.weights.stride_shift());

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
            return std::make_unique<dense_weight_holder>(&workspace.workspace_ptr->weights.dense_weights,
                workspace.workspace_ptr->reduction_state.total_feature_width, workspace.workspace_ptr);
          })
      .def(
          "json_weights",
          [](const workspace_with_logger_contexts& workspace, bool include_feature_names,
              bool include_online_state) -> std::string
          {
            // Invert hash is enabled with "--invert_hash"
            auto old_dump_json_weights_include_feature_names =
                workspace.workspace_ptr->output_model_config.dump_json_weights_include_feature_names;
            workspace.workspace_ptr->output_model_config.dump_json_weights_include_feature_names =
                include_feature_names;
            auto old_dump_json_weights_include_extra_online_state =
                workspace.workspace_ptr->output_model_config.dump_json_weights_include_extra_online_state;
            workspace.workspace_ptr->output_model_config.dump_json_weights_include_extra_online_state =
                include_online_state;
            auto on_exit = VW::scope_exit(
                [&]()
                {
                  workspace.workspace_ptr->output_model_config.dump_json_weights_include_feature_names =
                      old_dump_json_weights_include_feature_names;
                  workspace.workspace_ptr->output_model_config.dump_json_weights_include_extra_online_state =
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
              if (!all.output_config.hash_inv)
              {
                THROW(
                    "record_feature_names must be enabled (from Workspace constructor) to use "
                    "include_feature_names=True");
              }
            }

            auto print_invert_guard = VW::swap_guard(all.output_config.print_invert, include_feature_names);
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

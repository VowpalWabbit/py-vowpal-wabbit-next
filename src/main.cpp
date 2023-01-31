#include "vw/common/text_utils.h"
#include "vw/config/options_cli.h"
#include "vw/core/cache.h"
#include "vw/core/constant.h"
#include "vw/core/example.h"
#include "vw/core/global_data.h"
#include "vw/core/label_type.h"
#include "vw/core/learner.h"
#include "vw/core/loss_functions.h"
#include "vw/core/memory.h"
#include "vw/core/object_pool.h"
#include "vw/core/parse_example.h"
#include "vw/core/prediction_type.h"
#include "vw/core/simple_label.h"
#include "vw/core/vw.h"
#include "vw/io/io_adapter.h"
#include "vw/io/logger.h"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <sys/types.h>

#include <iostream>
#include <memory>
#include <optional>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

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
  return std::shared_ptr<VW::example>(SHARED_EXAMPLE_POOL.get_object(),
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
using active_multiclass_pred_t = std::tuple<float, std::vector<uint32_t>>;

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
}  // namespace

PYBIND11_MODULE(_core, m)
{
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

  py::class_<VW::example, std::shared_ptr<VW::example>>(m, "Example")
      .def(py::init(
          []()
          {
            // shared ptr which returns to the pool upon deletion
            return get_example_from_pool();
          }))
      .def("_is_newline", [](VW::example& ex) -> bool { return ex.is_newline; });

  py::class_<workspace_with_logger_contexts>(m, "Workspace")
      .def(py::init(
               [](const std::vector<std::string>& args, const std::optional<py::bytes>& bytes)
               {
                 auto opts = std::make_unique<VW::config::options_cli>(args);
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
                 return wrapped_object;
               }),
          py::arg("args"), py::kw_only(), py::arg("model_data") = std::nullopt)
      .def(
          "learn_one",
          [](workspace_with_logger_contexts& workspace, VW::example& example) -> void
          {
            // TODO check if setup.
            workspace.workspace_ptr->learn(example);

            // TODO - when updating VW submodule if learn calls update stats then remove this to avoid a double call.
            VW::LEARNER::as_singleline(workspace.workspace_ptr->l)->update_stats(*workspace.workspace_ptr, example);
          },
          py::arg("examples"), py::kw_only())
      .def(
          "learn_multi_ex_one",
          [](workspace_with_logger_contexts& workspace, std::vector<VW::example*>& example) -> void
          {
            assert(!example.empty());
            workspace.workspace_ptr->learn(example);

            // TODO - when updating VW submodule if learn calls update stats then remove this to avoid a double call.
            VW::LEARNER::as_multiline(workspace.workspace_ptr->l)->update_stats(*workspace.workspace_ptr, example);
          },
          py::arg("examples"), py::kw_only())
      .def(
          "predict_one",
          [](workspace_with_logger_contexts& workspace, VW::example& example) -> prediction_t
          {
            // TODO check if setup.
            // We must save and restore test_only because the library sets this values and does not undo it.
            bool test_only = example.test_only;
            workspace.workspace_ptr->predict(example);

            // TODO - when updating VW submodule if learn calls update stats then remove this to avoid a double call.
            VW::LEARNER::as_singleline(workspace.workspace_ptr->l)->update_stats(*workspace.workspace_ptr, example);
            example.test_only = test_only;
            return to_prediction(example.pred, workspace.workspace_ptr->l->get_output_prediction_type());
          },
          py::arg("examples"), py::kw_only())
      .def(
          "predict_multi_ex_one",
          [](workspace_with_logger_contexts& workspace, std::vector<VW::example*>& example) -> prediction_t
          {
            assert(!example.empty());
            // We must save and restore test_only because the library sets this values and does not undo it.
            std::vector<bool> test_onlys;
            test_onlys.reserve(example.size());
            for (auto ex : example) { test_onlys.push_back(ex->test_only); }
            workspace.workspace_ptr->predict(example);

            // TODO - when updating VW submodule if learn calls update stats then remove this to avoid a double call.
            VW::LEARNER::as_multiline(workspace.workspace_ptr->l)->update_stats(*workspace.workspace_ptr, example);
            for (size_t i = 0; i < example.size(); i++) { example[i]->test_only = test_onlys[i]; }
            return to_prediction(example[0]->pred, workspace.workspace_ptr->l->get_output_prediction_type());
          },
          py::arg("examples"), py::kw_only())
      .def("get_is_multiline",
          [](const workspace_with_logger_contexts& workspace) { return workspace.workspace_ptr->l->is_multiline(); })
      .def("get_prediction_type",
          [](const workspace_with_logger_contexts& workspace)
          { return workspace.workspace_ptr->l->get_output_prediction_type(); })
      .def("get_label_type",
          [](const workspace_with_logger_contexts& workspace) -> VW::label_type_t
          { return workspace.workspace_ptr->l->get_input_label_type(); })
      .def("setup_example",
          [](const workspace_with_logger_contexts& workspace, VW::example& ex)
          { VW::setup_example(*workspace.workspace_ptr, &ex); });

  m.def("_parse_line_text", &::parse_text_line, py::arg("workspace"), py::arg("line"));
  m.def("_write_cache_header", &::write_cache_header, py::arg("workspace"), py::arg("file"));
  m.def("_write_cache_example", &::write_cache_example, py::arg("workspace"), py::arg("example"), py::arg("file"));

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

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}

#include "vw/config/options_cli.h"
#include "vw/core/constant.h"
#include "vw/core/example.h"
#include "vw/core/global_data.h"
#include "vw/core/loss_functions.h"
#include "vw/core/simple_label.h"
#include "vw/core/vw.h"
#include "vw/io/io_adapter.h"
#include "vw/io/logger.h"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <iostream>
#include <memory>
#include <optional>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

struct logger_context
{
  py::object driver_logger;
  py::object log_logger;
};

struct workspace_with_logger_contexts
{
  std::unique_ptr<logger_context> logger_context_ptr;
  std::unique_ptr<VW::workspace> workspace_ptr;
};

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

PYBIND11_MODULE(_core, m)
{
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
                 wrapped_object->workspace_ptr = VW::initialize_experimental(std::move(opts), std::move(model_reader),
                     driver_log, wrapped_object->logger_context_ptr.get(), &logger);
                 return wrapped_object;
               }),
          py::arg("args"), py::kw_only(), py::arg("model_data") = std::nullopt);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}

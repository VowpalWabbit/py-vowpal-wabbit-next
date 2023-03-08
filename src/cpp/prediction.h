#pragma once

#include "vw/core/example.h"
#include "vw/core/prediction_type.h"

#include <pybind11/pytypes.h>

#include <vector>
#include <variant>

namespace py = pybind11;

namespace vwpy
{

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

prediction_t to_prediction(const VW::polyprediction& polypred, VW::prediction_type_t type);
}  // namespace vwpy

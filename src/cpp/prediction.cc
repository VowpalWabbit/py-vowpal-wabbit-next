#include "prediction.h"

#include <vector>

// TODO: consider a more efficient way of exposing these values
vwpy::prediction_t vwpy::to_prediction(const VW::polyprediction& polypred, VW::prediction_type_t type)
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

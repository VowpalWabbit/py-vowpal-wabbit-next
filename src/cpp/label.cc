#include "label.h"

vwpy::label_variant_t vwpy::to_label_variant(const VW::example& ex, VW::label_type_t type)
{
  switch (type)
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
      return std::monostate{};
    default:
      throw std::runtime_error("Unsupported label type");
  }
}
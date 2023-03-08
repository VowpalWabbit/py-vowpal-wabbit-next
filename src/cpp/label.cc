#include "label.h"

namespace
{

// helper type for the visitor #4
template <class... Ts>
struct overloaded : Ts...
{
  using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;
}  // namespace

void vwpy::from_label_variant(VW::example& ex, const vwpy::label_variant_ptrs_t& label)
{
  std::visit(
      overloaded{
          // Corresponds to nolabel
          [](std::monostate) {},
          [&](vwpy::py_simple_label* simple_label)
          {
            ex.l.simple.label = simple_label->label;
            ex.ex_reduction_features.get<VW::simple_label_reduction_features>().weight = simple_label->weight;
            ex.ex_reduction_features.get<VW::simple_label_reduction_features>().initial = simple_label->initial;
          },
          [&](VW::multiclass_label* multiclass_label) { ex.l.multi = *multiclass_label; },
          [&](VW::cb_label* cb_label) { ex.l.cb = *cb_label; },
          [&](VW::cs_label* cs_label) { ex.l.cs = *cs_label; },
          [&](VW::ccb_label* ccb_label) { ex.l.conditional_contextual_bandit = *ccb_label; },
      },
      label);
}

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
    case VW::label_type_t::CCB:
      return ex.l.conditional_contextual_bandit;
    case VW::label_type_t::NOLABEL:
      return std::monostate{};
    default:
      throw std::runtime_error("Unsupported label type");
  }
}
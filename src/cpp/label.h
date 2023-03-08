#pragma once

#include "vw/core/cb.h"
#include "vw/core/ccb_label.h"
#include "vw/core/cost_sensitive.h"
#include "vw/core/example.h"
#include "vw/core/multiclass.h"
#include "vw/core/simple_label.h"

#include <variant>

namespace vwpy
{

struct py_simple_label
{
  float label;
  float weight;
  float initial;
};

using label_variant_t =
    std::variant<py_simple_label, VW::multiclass_label, VW::cb_label, VW::cs_label, VW::ccb_label, std::monostate>;
using label_variant_ptrs_t =
    std::variant<py_simple_label*, VW::multiclass_label*, VW::cb_label*, VW::cs_label*, VW::ccb_label*, std::monostate>;

label_variant_t to_label_variant(const VW::example& ex, VW::label_type_t type);
void from_label_variant(VW::example& ex, const label_variant_ptrs_t& label);

}  // namespace vwpy
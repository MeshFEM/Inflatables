#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
////////////////////////////////////////////////////////////////////////////////
// For returning extended precision numbers (long doubles)...
// From https://github.com/pybind/pybind11/issues/1785
// https://github.com/eacousineau/repro/blob/5e6acf6/python/pybind11/custom_tests/test_numpy_issue1785.cc#L103-L105
////////////////////////////////////////////////////////////////////////////////
namespace py = pybind11;

using float128 = long double;
static_assert(sizeof(float128) == 16, "Bad size");

template <typename T, int Dim>
using Vector = Eigen::Matrix<T, Dim, 1>;

namespace pybind11 { namespace detail {

template <typename T>
struct npy_scalar_caster {
  PYBIND11_TYPE_CASTER(T, _("PleaseOverride"));
  using Array = array_t<T>;

  bool load(handle src, bool convert) {
    // Taken from Eigen casters. Permits either scalar dtype or scalar array.
    handle type = dtype::of<T>().attr("type");  // Could make more efficient.
    if (!convert && !isinstance<Array>(src) && !isinstance(src, type))
      return false;
    Array tmp = Array::ensure(src);
    if (tmp && tmp.size() == 1 && tmp.ndim() == 0) {
      this->value = *tmp.data();
      return true;
    }
    return false;
  }

  static handle cast(T src, return_value_policy, handle) {
    Array tmp({1});
    tmp.mutable_at(0) = src;
    tmp.resize({});
    // You could also just return the array if you want a scalar array.
    object scalar = tmp[py::tuple()];
    return scalar.release();
  }
};

template <>
struct type_caster<float128> : npy_scalar_caster<float128> {
  static constexpr auto name = _("float128");
};

}}  // namespace pybind11::detail


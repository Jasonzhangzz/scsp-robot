#!/usr/bin/env bash
# Compile dairlib SamplingC3Controller and LCS/C3 into the acc environment.
set -euo pipefail

ROOT="/home/lab423/scsp/thirdparty/dairlib"
DRAKE="/home/lab423/drake"
EIGEN="/tmp/eigen-src/eigen-3.4.0"
OSQP_INC="/tmp/osqp-0.6.3/include"
OSQP_LIB="/home/lab423/scsp/thirdparty/osqp-0.6.3/lib"
GUROBI_INC="/home/lab423/scsp/thirdparty/gurobi/gurobi1103/linux64/include"
GUROBI_LIB="/home/lab423/scsp/thirdparty/gurobi/gurobi1103/linux64/lib"
GUROBI_CPP="/home/lab423/scsp/thirdparty/gurobi/gurobi1103/linux64/src/build"
LCM="/tmp/dairlib-lcm"
PYBIND="/home/lab423/miniconda3/envs/acc/lib/python3.8/site-packages/pybind11/include"
PYTHON="/home/lab423/miniconda3/envs/acc/include/python3.8"
ACC_SITE="/home/lab423/miniconda3/envs/acc/lib/python3.8/site-packages"
OBJ="/tmp/dairlib-sampling-c3-objs"
BIND="/home/lab423/scsp/scsp-robot/planning/csrc/dairlib_sampling_c3_bind.cc"

mkdir -p "$OBJ"
CXXFLAGS=(
  -std=c++20 -fPIC -O2 -fopenmp
  -ffunction-sections -fdata-sections
  -Wno-deprecated-enum-enum-conversion -Wno-deprecated-declarations
  -I"$ROOT" -I"$DRAKE/include" -I"$DRAKE/include/lcm"
  -I"$EIGEN" -I"$OSQP_INC" -I"$GUROBI_INC" -I"$LCM"
)

SOURCES=(
  solvers/c3.cc
  solvers/c3_qp.cc
  solvers/c3_miqp.cc
  solvers/lcs.cc
  solvers/lcs_factory.cc
  solvers/fast_osqp_solver.cc
  solvers/fast_osqp_solver_common.cc
  solvers/c3_output.cc
  multibody/geom_geom_collider.cc
  multibody/multibody_utils.cc
  examples/sampling_c3/generate_samples.cc
  examples/sampling_c3/reposition.cc
  examples/sampling_c3/sampling_c3_utils.cc
  systems/controllers/sampling_based_c3_controller.cc
  systems/framework/timestamped_vector.cc
  common/update_context.cc
  common/quaternion_error_hessian.cc
  common/find_resource.cc
  third_party/spruce/spruce.cc
  lcm/lcm_trajectory.cc
)

OBJECTS=()
for src in "${SOURCES[@]}"; do
  obj="$OBJ/$(basename "$src").o"
  echo "CC $src"
  g++ "${CXXFLAGS[@]}" -c "$ROOT/$src" -o "$obj"
  OBJECTS+=("$obj")
done

echo "CC dairlib_sampling_c3_bind.cc"
g++ "${CXXFLAGS[@]}" -I"$PYBIND" -I"$PYTHON" -c "$BIND" -o "$OBJ/bind.o"
OBJECTS+=("$OBJ/bind.o")

OUT="$ACC_SITE/dairlib_sampling_c3.cpython-38-x86_64-linux-gnu.so"
echo "LINK $OUT"
g++ -shared -fopenmp -Wl,--gc-sections \
  -o "$OUT" "${OBJECTS[@]}" \
  -L"$OSQP_LIB" -L"$GUROBI_LIB" -L"$DRAKE/lib" \
  "$GUROBI_CPP/libgurobi_c++.a" \
  /usr/lib/x86_64-linux-gnu/libfmt.a \
  -Wl,--no-as-needed -lgurobi110 -losqp -ldrake \
  -Wl,-rpath,"$DRAKE/lib" \
  -Wl,-rpath,"$OSQP_LIB" \
  -Wl,-rpath,"$GUROBI_LIB" \
  -lpthread -ldl
echo "INSTALLED $OUT"

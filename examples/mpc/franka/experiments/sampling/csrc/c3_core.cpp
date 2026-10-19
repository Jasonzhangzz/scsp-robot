// Consensus Complementarity Control (C3) for a linear complementarity system.
//
// The ADMM loop follows Aydinoglu, Fazeli, Posa, "Contact-Implicit Trajectory
// Optimization Using Orthogonal Collocation" / the C3 MPC used by dairlib's
// sampling_based_c3 controller: a dynamics QP, then a convex QP projection
// onto lambda >= 0 and Ex + F lambda + Hu + c >= 0. This file does not link
// Drake, Gurobi, OSQP, or the dairlib simulator.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

namespace py = pybind11;

namespace {

constexpr double kTol = 1e-9;
constexpr int kMaxQpIter = 80;

Eigen::VectorXd SolveKkt(const Eigen::MatrixXd& hessian,
                         const Eigen::VectorXd& gradient,
                         const Eigen::MatrixXd& constraints,
                         const Eigen::VectorXd& bound) {
  const int n = static_cast<int>(hessian.rows());
  const int m = static_cast<int>(constraints.rows());
  if (m == 0) {
    return hessian.ldlt().solve(-gradient);
  }
  Eigen::MatrixXd kkt(n + m, n + m);
  kkt.setZero();
  kkt.topLeftCorner(n, n) = hessian;
  kkt.topRightCorner(n, m) = constraints.transpose();
  kkt.bottomLeftCorner(m, n) = constraints;
  Eigen::VectorXd rhs(n + m);
  rhs.head(n) = -gradient;
  rhs.tail(m) = bound;
  const Eigen::VectorXd solution =
      kkt.colPivHouseholderQr().solve(rhs);
  return solution.head(n);
}

Eigen::VectorXd Multipliers(const Eigen::MatrixXd& hessian,
                            const Eigen::VectorXd& gradient,
                            const Eigen::MatrixXd& constraints,
                            const Eigen::VectorXd& bound) {
  const int n = static_cast<int>(hessian.rows());
  const int m = static_cast<int>(constraints.rows());
  if (m == 0) return Eigen::VectorXd::Zero(0);
  Eigen::MatrixXd kkt(n + m, n + m);
  kkt.setZero();
  kkt.topLeftCorner(n, n) = hessian;
  kkt.topRightCorner(n, m) = constraints.transpose();
  kkt.bottomLeftCorner(m, n) = constraints;
  Eigen::VectorXd rhs(n + m);
  rhs.head(n) = -gradient;
  rhs.tail(m) = bound;
  const Eigen::VectorXd solution =
      kkt.colPivHouseholderQr().solve(rhs);
  return -solution.tail(m);
}

Eigen::VectorXd FeasibleActiveSetQp(const Eigen::MatrixXd& hessian,
                            const Eigen::VectorXd& gradient,
                            const Eigen::MatrixXd& constraints,
                            const Eigen::VectorXd& bound,
                            Eigen::VectorXd z) {
  const int n = static_cast<int>(hessian.rows());
  const int m = static_cast<int>(constraints.rows());
  std::vector<int> active;
  for (int iteration = 0; iteration < kMaxQpIter; ++iteration) {
    Eigen::MatrixXd working(static_cast<int>(active.size()), n);
    Eigen::VectorXd working_bound(static_cast<int>(active.size()));
    for (int row = 0; row < static_cast<int>(active.size()); ++row) {
      working.row(row) = constraints.row(active[row]);
      working_bound(row) = bound(active[row]);
    }
    const Eigen::VectorXd z_eq =
        SolveKkt(hessian, gradient, working, working_bound);
    const Eigen::VectorXd step = z_eq - z;
    if (step.norm() < kTol) {
      const Eigen::VectorXd multipliers =
          Multipliers(hessian, gradient, working, working_bound);
      if (multipliers.size() == 0 || multipliers.minCoeff() >= -1e-8) {
        return z_eq;
      }
      int drop_slot = 0;
      multipliers.minCoeff(&drop_slot);
      active.erase(active.begin() + drop_slot);
      continue;
    }
    double alpha = 1.0;
    int blocker = -1;
    const Eigen::VectorXd along = constraints * step;
    const Eigen::VectorXd at = constraints * z;
    for (int row = 0; row < m; ++row) {
      if (std::find(active.begin(), active.end(), row) != active.end()) continue;
      if (along(row) < -1e-12) {
        const double candidate = (bound(row) - at(row)) / along(row);
        if (candidate < alpha) {
          alpha = std::max(0.0, candidate);
          blocker = row;
        }
      }
    }
    z += alpha * step;
    if (blocker >= 0 && alpha < 1.0 - 1e-12) active.push_back(blocker);
  }
  throw std::runtime_error("active-set QP did not converge");
}

Eigen::VectorXd ActiveSetQp(const Eigen::MatrixXd& hessian_in,
                            const Eigen::VectorXd& gradient,
                            const Eigen::MatrixXd& constraints,
                            const Eigen::VectorXd& bound) {
  const int n = static_cast<int>(hessian_in.rows());
  Eigen::MatrixXd hessian = 0.5 * (hessian_in + hessian_in.transpose());
  hessian.diagonal().array() += 1e-10;
  const Eigen::VectorXd unconstrained = hessian.ldlt().solve(-gradient);
  if (constraints.rows() == 0 ||
      (constraints * unconstrained - bound).minCoeff() >= -1e-9) {
    return unconstrained;
  }
  Eigen::VectorXd start = Eigen::VectorXd::Zero(n);
  if ((constraints * start - bound).minCoeff() < -1e-9) {
    // A primal active-set method must start feasible. In particular the
    // contact projection's zero vector has gap=-radius. Starting there can
    // hit the lambda>=0 blocker at step length zero and return all zeros,
    // pulling the next consensus QP toward the world origin.
    // Phase I adds a common slack: C*z+s>=b, s>=0, then minimizes s.
    const int m = static_cast<int>(constraints.rows());
    Eigen::MatrixXd phase_constraints = Eigen::MatrixXd::Zero(m + 1, n + 1);
    phase_constraints.topLeftCorner(m, n) = constraints;
    phase_constraints.block(0, n, m, 1).setOnes();
    phase_constraints(m, n) = 1.0;
    Eigen::VectorXd phase_bound = Eigen::VectorXd::Zero(m + 1);
    phase_bound.head(m) = bound;
    Eigen::MatrixXd phase_hessian = 1e-8 * Eigen::MatrixXd::Identity(n + 1, n + 1);
    Eigen::VectorXd phase_gradient = Eigen::VectorXd::Zero(n + 1);
    phase_gradient(n) = 1.0;
    Eigen::VectorXd phase_start = Eigen::VectorXd::Zero(n + 1);
    phase_start(n) = std::max(0.0, bound.maxCoeff()) + 1.0;
    const Eigen::VectorXd phase = FeasibleActiveSetQp(
        phase_hessian, phase_gradient, phase_constraints, phase_bound, phase_start);
    start = phase.head(n);
    if ((constraints * start - bound).minCoeff() < -1e-7) {
      throw std::runtime_error("QP constraints are infeasible");
    }
  }
  const Eigen::VectorXd solution = FeasibleActiveSetQp(
      hessian, gradient, constraints, bound, start);
  if (!solution.allFinite() ||
      (constraints * solution - bound).minCoeff() < -1e-7) {
    throw std::runtime_error("QP returned an infeasible solution");
  }
  return solution;
}

void AddQuadratic(Eigen::MatrixXd* hessian, Eigen::VectorXd* gradient,
                  const Eigen::MatrixXd& map, const Eigen::VectorXd& offset,
                  const Eigen::MatrixXd& weight) {
  if (map.size() == 0 || weight.size() == 0) return;
  const Eigen::MatrixXd weighted = weight * map;
  *hessian += 2.0 * map.transpose() * weighted;
  *gradient += 2.0 * map.transpose() * weight * offset;
}

struct Model {
  std::vector<Eigen::MatrixXd> A, B, D, E, F, H;
  std::vector<Eigen::VectorXd> d, c;
  int horizon = 0;
  int n = 0;
  int m = 0;
  int k = 0;
  double scale = 1.0;
};

Model ScaleModel(const std::vector<Eigen::MatrixXd>& A,
                 const std::vector<Eigen::MatrixXd>& B,
                 const std::vector<Eigen::MatrixXd>& D,
                 const std::vector<Eigen::VectorXd>& d,
                 const std::vector<Eigen::MatrixXd>& E,
                 const std::vector<Eigen::MatrixXd>& F,
                 const std::vector<Eigen::MatrixXd>& H,
                 const std::vector<Eigen::VectorXd>& cvec) {
  Model model;
  model.horizon = static_cast<int>(A.size());
  model.n = static_cast<int>(A[0].rows());
  model.k = static_cast<int>(B[0].cols());
  model.m = static_cast<int>(D[0].cols());
  const double dynamics_norm = A[0].norm();
  const double contact_norm = D[0].norm();
  model.scale = contact_norm > 1e-12 ? dynamics_norm / contact_norm : 1.0;
  model.A = A;
  model.B = B;
  model.d = d;
  model.F = F;
  model.D.resize(model.horizon);
  model.E.resize(model.horizon);
  model.H.resize(model.horizon);
  model.c.resize(model.horizon);
  for (int t = 0; t < model.horizon; ++t) {
    model.D[t] = D[t] * model.scale;
    model.E[t] = E[t] / model.scale;
    model.H[t] = H[t] / model.scale;
    model.c[t] = cvec[t] / model.scale;
  }
  return model;
}

struct ReducedQp {
  Eigen::MatrixXd hessian;
  Eigen::VectorXd gradient;
  Eigen::MatrixXd constraints;
  Eigen::VectorXd bound;
  std::vector<Eigen::MatrixXd> state_map;
  std::vector<Eigen::VectorXd> state_offset;
  std::vector<Eigen::MatrixXd> force_map;
  std::vector<Eigen::MatrixXd> input_map;
};

ReducedQp AssembleQp(const Model& model, const Eigen::VectorXd& x0,
                     const std::vector<Eigen::MatrixXd>& Q,
                     const std::vector<Eigen::MatrixXd>& R,
                     const std::vector<Eigen::MatrixXd>& G,
                     const std::vector<Eigen::VectorXd>& desired,
                     const std::vector<Eigen::VectorXd>& consensus,
                     const Eigen::VectorXd& u_lower,
                     const Eigen::VectorXd& u_upper) {
  const int N = model.horizon;
  const int n = model.n;
  const int m = model.m;
  const int k = model.k;
  const int nv = N * (m + k);
  ReducedQp qp;
  qp.hessian = Eigen::MatrixXd::Zero(nv, nv);
  qp.gradient = Eigen::VectorXd::Zero(nv);
  qp.state_map.assign(N + 1, Eigen::MatrixXd::Zero(n, nv));
  qp.state_offset.assign(N + 1, Eigen::VectorXd::Zero(n));
  qp.state_offset[0] = x0;
  qp.force_map.assign(N, Eigen::MatrixXd::Zero(m, nv));
  qp.input_map.assign(N, Eigen::MatrixXd::Zero(k, nv));

  for (int t = 0; t < N; ++t) {
    const int base = t * (m + k);
    qp.force_map[t].block(0, base, m, m).setIdentity();
    qp.input_map[t].block(0, base + m, k, k).setIdentity();
    qp.state_map[t + 1] = model.A[t] * qp.state_map[t] +
                          model.B[t] * qp.input_map[t] +
                          model.D[t] * qp.force_map[t];
    qp.state_offset[t + 1] = model.A[t] * qp.state_offset[t] + model.d[t];

    Eigen::MatrixXd map(n + m + k, nv);
    map << qp.state_map[t], qp.force_map[t], qp.input_map[t];
    Eigen::VectorXd offset(n + m + k);
    offset << qp.state_offset[t] - consensus[t].head(n),
        -consensus[t].segment(n, m), -consensus[t].tail(k);
    AddQuadratic(&qp.hessian, &qp.gradient, map, offset, G[t]);
    AddQuadratic(&qp.hessian, &qp.gradient, qp.state_map[t],
                 qp.state_offset[t] - desired[t], Q[t]);
    AddQuadratic(&qp.hessian, &qp.gradient, qp.input_map[t],
                 Eigen::VectorXd::Zero(k), R[t]);
  }
  AddQuadratic(&qp.hessian, &qp.gradient, qp.state_map[N],
               qp.state_offset[N] - desired[N], Q[N]);

  if (u_lower.size() == k && u_upper.size() == k) {
    qp.constraints = Eigen::MatrixXd::Zero(2 * N * k, nv);
    qp.bound = Eigen::VectorXd::Zero(2 * N * k);
    for (int t = 0; t < N; ++t) {
      qp.constraints.block(t * k, 0, k, nv) = qp.input_map[t];
      qp.bound.segment(t * k, k) = u_lower;
      qp.constraints.block((N + t) * k, 0, k, nv) = -qp.input_map[t];
      qp.bound.segment((N + t) * k, k) = -u_upper;
    }
  } else {
    qp.constraints = Eigen::MatrixXd::Zero(0, nv);
    qp.bound = Eigen::VectorXd::Zero(0);
  }
  return qp;
}

std::vector<Eigen::VectorXd> Unpack(const Model& model, const ReducedQp& qp,
                                    const Eigen::VectorXd& decision) {
  std::vector<Eigen::VectorXd> z(model.horizon);
  for (int t = 0; t < model.horizon; ++t) {
    z[t] = Eigen::VectorXd::Zero(model.n + model.m + model.k);
    z[t].head(model.n) =
        qp.state_map[t] * decision + qp.state_offset[t];
    z[t].segment(model.n, model.m) = qp.force_map[t] * decision;
    z[t].tail(model.k) = qp.input_map[t] * decision;
  }
  return z;
}

Eigen::VectorXd Project(const Eigen::MatrixXd& weight,
                        const Eigen::VectorXd& target,
                        const Eigen::MatrixXd& E, const Eigen::MatrixXd& F,
                        const Eigen::MatrixXd& H, const Eigen::VectorXd& cvec,
                        double alpha) {
  const int n = static_cast<int>(E.cols());
  const int m = static_cast<int>(E.rows());
  const int k = static_cast<int>(H.cols());
  Eigen::MatrixXd shaped = weight;
  shaped.block(n, n, m, m) = alpha * F;
  Eigen::MatrixXd hessian = 0.5 * (shaped + shaped.transpose());
  hessian.block(n, n, m, m) +=
      (1.0 - alpha) * 0.5 * (F + F.transpose());
  const Eigen::VectorXd gradient = -shaped.transpose() * target;
  Eigen::MatrixXd constraints(2 * m, n + m + k);
  constraints.setZero();
  constraints.block(0, n, m, m).setIdentity();
  constraints.block(m, 0, m, n) = E;
  constraints.block(m, n, m, m) = F;
  constraints.block(m, n + m, m, k) = H;
  Eigen::VectorXd bound(2 * m);
  bound.head(m).setZero();
  bound.tail(m) = -cvec;
  return ActiveSetQp(hessian, gradient, constraints, bound);
}

}  // namespace

py::dict SolveC3Impl(const std::vector<Eigen::MatrixXd>& A,
                     const std::vector<Eigen::MatrixXd>& B,
                     const std::vector<Eigen::MatrixXd>& D,
                     const std::vector<Eigen::VectorXd>& d,
                     const std::vector<Eigen::MatrixXd>& E,
                     const std::vector<Eigen::MatrixXd>& F,
                     const std::vector<Eigen::MatrixXd>& H,
                     const std::vector<Eigen::VectorXd>& cvec,
                     const std::vector<Eigen::MatrixXd>& Q,
                     const std::vector<Eigen::MatrixXd>& R,
                     const std::vector<Eigen::MatrixXd>& G,
                     const std::vector<Eigen::MatrixXd>& U,
                     const Eigen::VectorXd& x0,
                     const std::vector<Eigen::VectorXd>& desired,
                     int admm_iterations, double rho_scale,
                     double projection_alpha, const Eigen::VectorXd& u_lower,
                     const Eigen::VectorXd& u_upper);

Eigen::MatrixXd MatrixFrom(const py::buffer& buffer) {
  const py::buffer_info info = buffer.request();
  if (info.size == 0) {
    if (info.ndim == 1) return Eigen::MatrixXd::Zero(info.shape[0], 1);
    return Eigen::MatrixXd::Zero(info.shape[0], info.shape[1]);
  }
  if (info.ndim == 1) {
    Eigen::MatrixXd out(info.shape[0], 1);
    const double* data = static_cast<const double*>(info.ptr);
    for (py::ssize_t i = 0; i < info.shape[0]; ++i) {
      out(i, 0) = data[i * info.strides[0] / static_cast<py::ssize_t>(sizeof(double))];
    }
    return out;
  }
  if (info.ndim != 2) throw std::invalid_argument("expected a one- or two-dimensional array");
  Eigen::MatrixXd out(info.shape[0], info.shape[1]);
  const double* data = static_cast<const double*>(info.ptr);
  const py::ssize_t row_stride = info.strides[0] / static_cast<py::ssize_t>(sizeof(double));
  const py::ssize_t col_stride = info.strides[1] / static_cast<py::ssize_t>(sizeof(double));
  for (py::ssize_t row = 0; row < info.shape[0]; ++row) {
    for (py::ssize_t col = 0; col < info.shape[1]; ++col) {
      out(row, col) = data[row * row_stride + col * col_stride];
    }
  }
  return out;
}

Eigen::VectorXd VectorFrom(const py::buffer& buffer) {
  const Eigen::MatrixXd matrix = MatrixFrom(buffer);
  if (matrix.cols() == 1) return matrix.col(0);
  if (matrix.rows() == 1) return matrix.row(0).transpose();
  throw std::invalid_argument("expected a vector");
}

py::array_t<double> MatrixToArray(const Eigen::MatrixXd& matrix) {
  py::array_t<double> array({matrix.rows(), matrix.cols()});
  const py::buffer_info info = array.request();
  double* data = static_cast<double*>(info.ptr);
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
      data[row * matrix.cols() + col] = matrix(row, col);
    }
  }
  return array;
}

std::vector<Eigen::MatrixXd> MatrixList(const py::list& items) {
  std::vector<Eigen::MatrixXd> out;
  out.reserve(items.size());
  for (const py::handle& item : items) out.push_back(MatrixFrom(item.cast<py::buffer>()));
  return out;
}

std::vector<Eigen::VectorXd> VectorList(const py::list& items) {
  std::vector<Eigen::VectorXd> out;
  out.reserve(items.size());
  for (const py::handle& item : items) out.push_back(VectorFrom(item.cast<py::buffer>()));
  return out;
}

py::dict SolveC3(const py::list& A_list, const py::list& B_list,
                 const py::list& D_list, const py::list& d_list,
                 const py::list& E_list, const py::list& F_list,
                 const py::list& H_list, const py::list& c_list,
                 const py::list& Q_list, const py::list& R_list,
                 const py::list& G_list, const py::list& U_list,
                 const py::buffer& x0_buffer, const py::list& desired_list,
                 int admm_iterations, double rho_scale, double projection_alpha,
                 const py::buffer& u_lower_buffer, const py::buffer& u_upper_buffer) {
  const std::vector<Eigen::MatrixXd> A = MatrixList(A_list);
  const std::vector<Eigen::MatrixXd> B = MatrixList(B_list);
  const std::vector<Eigen::MatrixXd> D = MatrixList(D_list);
  const std::vector<Eigen::VectorXd> d = VectorList(d_list);
  const std::vector<Eigen::MatrixXd> E = MatrixList(E_list);
  const std::vector<Eigen::MatrixXd> F = MatrixList(F_list);
  const std::vector<Eigen::MatrixXd> H = MatrixList(H_list);
  const std::vector<Eigen::VectorXd> cvec = VectorList(c_list);
  const std::vector<Eigen::MatrixXd> Q = MatrixList(Q_list);
  const std::vector<Eigen::MatrixXd> R = MatrixList(R_list);
  const std::vector<Eigen::MatrixXd> G = MatrixList(G_list);
  const std::vector<Eigen::MatrixXd> U = MatrixList(U_list);
  const Eigen::VectorXd x0 = VectorFrom(x0_buffer);
  const std::vector<Eigen::VectorXd> desired = VectorList(desired_list);
  Eigen::VectorXd u_lower = VectorFrom(u_lower_buffer);
  Eigen::VectorXd u_upper = VectorFrom(u_upper_buffer);
  if (u_lower.size() == 0 || u_upper.size() == 0) {
    u_lower.resize(0);
    u_upper.resize(0);
  }
  py::dict out = SolveC3Impl(
      A, B, D, d, E, F, H, cvec, Q, R, G, U, x0, desired, admm_iterations,
      rho_scale, projection_alpha, u_lower, u_upper);
  return out;
}

py::dict SolveC3Impl(const std::vector<Eigen::MatrixXd>& A,
                 const std::vector<Eigen::MatrixXd>& B,
                 const std::vector<Eigen::MatrixXd>& D,
                 const std::vector<Eigen::VectorXd>& d,
                 const std::vector<Eigen::MatrixXd>& E,
                 const std::vector<Eigen::MatrixXd>& F,
                 const std::vector<Eigen::MatrixXd>& H,
                 const std::vector<Eigen::VectorXd>& cvec,
                 const std::vector<Eigen::MatrixXd>& Q,
                 const std::vector<Eigen::MatrixXd>& R,
                 const std::vector<Eigen::MatrixXd>& G,
                 const std::vector<Eigen::MatrixXd>& U,
                 const Eigen::VectorXd& x0,
                 const std::vector<Eigen::VectorXd>& desired,
                 int admm_iterations, double rho_scale,
                 double projection_alpha, const Eigen::VectorXd& u_lower,
                 const Eigen::VectorXd& u_upper) {
  if (A.empty() || A.size() != B.size() || Q.size() != A.size() + 1) {
    throw std::invalid_argument(
        "C3 expects N dynamics matrices and N+1 state costs");
  }
  const Model model = ScaleModel(A, B, D, d, E, F, H, cvec);
  const int width = model.n + model.m + model.k;
  std::vector<Eigen::VectorXd> delta(model.horizon, Eigen::VectorXd::Zero(width));
  std::vector<Eigen::VectorXd> dual(model.horizon, Eigen::VectorXd::Zero(width));
  for (int t = 0; t < model.horizon; ++t) delta[t].head(model.n) = x0;
  std::vector<Eigen::MatrixXd> consensus_weight = G;
  const bool use_consensus_metric = U.empty() || U[0].isZero(0);
  std::vector<Eigen::VectorXd> z(model.horizon, Eigen::VectorXd::Zero(width));

  auto horizon_qp = [&](const std::vector<Eigen::VectorXd>& target) {
    const ReducedQp qp =
        AssembleQp(model, x0, Q, R, consensus_weight, desired, target,
                   u_lower, u_upper);
    const Eigen::VectorXd decision =
        ActiveSetQp(qp.hessian, qp.gradient, qp.constraints, qp.bound);
    return Unpack(model, qp, decision);
  };

  for (int iteration = 0; iteration < admm_iterations; ++iteration) {
    std::vector<Eigen::VectorXd> target(model.horizon);
    for (int t = 0; t < model.horizon; ++t) target[t] = delta[t] - dual[t];
    z = horizon_qp(target);
    for (int t = 0; t < model.horizon; ++t) {
      const Eigen::MatrixXd& metric =
          use_consensus_metric ? consensus_weight[t] : U[t];
      delta[t] = Project(metric, dual[t] + z[t], model.E[t], model.F[t],
                         model.H[t], model.c[t], projection_alpha);
      dual[t] = (dual[t] + z[t] - delta[t]) / rho_scale;
      consensus_weight[t] *= rho_scale;
    }
  }
  std::vector<Eigen::VectorXd> target(model.horizon);
  for (int t = 0; t < model.horizon; ++t) target[t] = delta[t] - dual[t];
  z = horizon_qp(target);

  Eigen::MatrixXd states(model.n, model.horizon + 1);
  Eigen::MatrixXd inputs(model.k, model.horizon);
  Eigen::MatrixXd forces(model.m, model.horizon);
  states.col(0) = x0;
  for (int t = 0; t < model.horizon; ++t) {
    inputs.col(t) = z[t].tail(model.k);
    forces.col(t) = model.scale * z[t].segment(model.n, model.m);
    states.col(t + 1) = A[t] * states.col(t) + B[t] * inputs.col(t) +
                        D[t] * forces.col(t) + d[t];
  }
  const bool finite = states.allFinite() && inputs.allFinite() && forces.allFinite();
  py::dict out;
  out["x"] = MatrixToArray(states);
  out["u"] = MatrixToArray(inputs);
  out["lambda"] = MatrixToArray(forces);
  out["success"] = finite;
  out["force_scale"] = model.scale;
  return out;
}

py::array_t<double> SolveQp(const py::buffer& hessian, const py::buffer& gradient,
                            const py::buffer& constraints, const py::buffer& bound) {
  const Eigen::MatrixXd cost = MatrixFrom(hessian);
  const Eigen::VectorXd linear = VectorFrom(gradient);
  Eigen::MatrixXd limits = MatrixFrom(constraints);
  Eigen::VectorXd lower = VectorFrom(bound);
  if (limits.rows() == 0) lower.resize(0);
  const Eigen::VectorXd solution = ActiveSetQp(cost, linear, limits, lower);
  return MatrixToArray(solution);
}

PYBIND11_MODULE(c3_core, module) {
  module.doc() = "Standalone C3 ADMM for linear complementarity systems";
  module.def("solve_qp", &SolveQp, py::arg("hessian"), py::arg("gradient"),
             py::arg("constraints"), py::arg("bound"));
  module.def("solve_c3", &SolveC3, py::arg("A"), py::arg("B"), py::arg("D"),
             py::arg("d"), py::arg("E"), py::arg("F"), py::arg("H"),
             py::arg("c"), py::arg("Q"), py::arg("R"), py::arg("G"),
             py::arg("U"), py::arg("x0"), py::arg("x_desired"),
             py::arg("admm_iterations") = 3, py::arg("rho_scale") = 3.0,
             py::arg("projection_alpha") = 0.01, py::arg("u_lower"),
             py::arg("u_upper"));
}

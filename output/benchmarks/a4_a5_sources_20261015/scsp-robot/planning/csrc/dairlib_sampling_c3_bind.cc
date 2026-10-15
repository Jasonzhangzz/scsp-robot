// Thin driver around dairlib's SamplingC3Controller. The plan itself is
// SamplingC3Controller::ComputePlan, and the contact model is
// LCSFactory::LinearizePlantToLCS on the jacktoy plant those parameters expect.

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dairlib/lcmt_radio_out.hpp"
#include "dairlib/lcmt_timestamped_saved_traj.hpp"
#include "examples/sampling_c3/parameter_headers/sampling_c3_controller_params.h"
#include "examples/sampling_c3/sampling_c3_utils.h"
#include "lcm/lcm_trajectory.h"
#include "systems/controllers/sampling_based_c3_controller.h"
#include "systems/framework/timestamped_vector.h"

#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/diagram_builder.h"
#include "gurobi_c++.h"

namespace py = pybind11;

namespace {

constexpr const char* kDairlibRoot = "/home/lab423/scsp/thirdparty/dairlib";
constexpr const char* kGurobiLicense =
    "/home/lab423/miniconda3/envs/acc/lib/python3.8/site-packages/gurobipy/.libs/gurobi.lic";

class WorkingDirectory {
 public:
  explicit WorkingDirectory(const std::string& directory) {
    if (getcwd(previous_, sizeof(previous_)) == nullptr) {
      throw std::runtime_error("getcwd failed");
    }
    if (chdir(directory.c_str()) != 0) {
      throw std::runtime_error("cannot enter " + directory);
    }
  }
  ~WorkingDirectory() { chdir(previous_); }

 private:
  char previous_[4096];
};

class CompiledSamplingC3 {
 public:
  explicit CompiledSamplingC3(const std::string& dairlib_root)
      : dairlib_root_(dairlib_root) {
    if (getenv("GRB_LICENSE_FILE") == nullptr) {
      setenv("GRB_LICENSE_FILE", kGurobiLicense, 0);
    }
    WorkingDirectory cwd(dairlib_root_);

    const std::string params_file =
        "examples/sampling_c3/jacktoy/parameters/"
        "sampling_c3_controller_params.yaml";
    SamplingC3ControllerParams params =
        drake::yaml::LoadYamlFile<SamplingC3ControllerParams>(params_file);

    drake::systems::DiagramBuilder<double> builder;
    auto [plant, scene_graph] =
        drake::multibody::AddMultibodyPlantSceneGraph(&builder, 0.0);
    dairlib::AddLCSModelsToPlant(
        &plant, &scene_graph, params.object_model,
        params.include_end_effector_orientation);
    plant.Finalize();
    n_x_ = plant.num_positions() + plant.num_velocities();
    if (n_x_ != 19 || plant.num_actuators() != 3) {
      throw std::runtime_error(
          "jacktoy LCS plant is expected to have 19 states and 3 actuators");
    }
    plant_ad_ = drake::systems::System<double>::ToAutoDiffXd(plant);
    diagram_ = builder.Build();
    diagram_context_ = diagram_->CreateDefaultContext();
    plant_ = &plant;
    plant_context_ = &diagram_->GetMutableSubsystemContext(
        plant, diagram_context_.get());
    context_ad_ = plant_ad_->CreateDefaultContext();

    auto contact_pairs = JacktoyContactPairs(*plant_);
    try {
      controller_ = std::make_unique<dairlib::systems::SamplingC3Controller>(
          *plant_, plant_context_, *plant_ad_, context_ad_.get(), contact_pairs,
          params, false);
    } catch (const GRBException& exc) {
      throw std::runtime_error("Gurobi: " + exc.getMessage());
    }
    controller_context_ = controller_->CreateDefaultContext();

    dairlib::lcmt_radio_out radio{};
    for (int i = 0; i < 16; ++i) radio.channel[i] = 0.0;
    controller_->get_input_port_radio().FixValue(controller_context_.get(),
                                                  radio);
    discrete_state_ = controller_->AllocateDiscreteVariables();
  }

  int n_x() const { return n_x_; }

  py::dict Step(const Eigen::VectorXd& x, const Eigen::VectorXd& x_des,
                const Eigen::VectorXd& x_final, double time) {
    if (x.size() != n_x_ || x_des.size() != n_x_ || x_final.size() != n_x_) {
      throw std::runtime_error("LCS state and targets must have length n_x");
    }
    dairlib::systems::TimestampedVector<double> state(n_x_);
    state.SetDataVector(x);
    state.set_timestamp(time);
    controller_->get_input_port_lcs_state().FixValue(controller_context_.get(),
                                                      state);
    controller_->get_input_port_target().FixValue(controller_context_.get(),
                                                   x_des);
    controller_->get_input_port_final_target().FixValue(
        controller_context_.get(), x_final);
    controller_context_->SetTime(time);

    controller_->CalcForcedDiscreteVariableUpdate(*controller_context_,
                                                   discrete_state_.get());
    controller_context_->get_mutable_discrete_state().SetFrom(*discrete_state_);

    const auto& executed =
        controller_->get_output_port_traj_execute_actor()
            .Eval<dairlib::lcmt_timestamped_saved_traj>(*controller_context_);
    dairlib::LcmTrajectory trajectory(executed.saved_traj);
    const auto& ee =
        trajectory.GetTrajectory("end_effector_position_target");
    const auto& force =
        trajectory.GetTrajectory("end_effector_force_target");
    const auto& mode_msg =
        controller_->get_output_port_is_c3_mode()
            .Eval<dairlib::lcmt_timestamped_saved_traj>(*controller_context_);
    dairlib::LcmTrajectory mode_traj(mode_msg.saved_traj);
    const double is_c3 =
        mode_traj.GetTrajectory("is_c3_mode").datapoints(0, 0);

    py::dict out;
    out["is_c3"] = is_c3 > 0.5;
    out["ee"] = ee.datapoints;
    out["force"] = force.datapoints;
    out["time"] = ee.time_vector;
    return out;
  }

 private:
  static std::vector<std::vector<drake::SortedPair<drake::geometry::GeometryId>>>
  JacktoyContactPairs(const drake::multibody::MultibodyPlant<double>& plant) {
    using drake::SortedPair;
    using drake::geometry::GeometryId;
    std::vector<std::vector<SortedPair<GeometryId>>> contact_pairs;
    std::vector<SortedPair<GeometryId>> ee_contact_pairs;
    std::vector<SortedPair<GeometryId>> ground_object_contact_pairs;
    std::unordered_map<std::string, GeometryId> contact_geoms;

    contact_geoms["EE"] = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("end_effector_simple"))[0];
    contact_geoms["GROUND"] = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("ground"))[0];
    std::vector<SortedPair<GeometryId>> ee_ground_contact{
        SortedPair(contact_geoms["EE"], contact_geoms["GROUND"])};

    contact_geoms["CAPSULE_1"] = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("capsule_1"))[0];
    contact_geoms["CAPSULE_2"] = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("capsule_2"))[0];
    contact_geoms["CAPSULE_3"] = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("capsule_3"))[0];
    contact_geoms["CAPSULE_1_SPHERE_1"] = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("capsule_1"))[1];
    contact_geoms["CAPSULE_1_SPHERE_2"] = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("capsule_1"))[2];
    contact_geoms["CAPSULE_2_SPHERE_1"] = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("capsule_2"))[1];
    contact_geoms["CAPSULE_2_SPHERE_2"] = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("capsule_2"))[2];
    contact_geoms["CAPSULE_3_SPHERE_1"] = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("capsule_3"))[1];
    contact_geoms["CAPSULE_3_SPHERE_2"] = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("capsule_3"))[2];

    ee_contact_pairs.push_back(
        SortedPair(contact_geoms["EE"], contact_geoms["CAPSULE_1"]));
    ee_contact_pairs.push_back(
        SortedPair(contact_geoms["EE"], contact_geoms["CAPSULE_2"]));
    ee_contact_pairs.push_back(
        SortedPair(contact_geoms["EE"], contact_geoms["CAPSULE_3"]));
    ground_object_contact_pairs.push_back(SortedPair(
        contact_geoms["CAPSULE_1_SPHERE_1"], contact_geoms["GROUND"]));
    ground_object_contact_pairs.push_back(SortedPair(
        contact_geoms["CAPSULE_1_SPHERE_2"], contact_geoms["GROUND"]));
    ground_object_contact_pairs.push_back(SortedPair(
        contact_geoms["CAPSULE_2_SPHERE_1"], contact_geoms["GROUND"]));
    ground_object_contact_pairs.push_back(SortedPair(
        contact_geoms["CAPSULE_2_SPHERE_2"], contact_geoms["GROUND"]));
    ground_object_contact_pairs.push_back(SortedPair(
        contact_geoms["CAPSULE_3_SPHERE_1"], contact_geoms["GROUND"]));
    ground_object_contact_pairs.push_back(SortedPair(
        contact_geoms["CAPSULE_3_SPHERE_2"], contact_geoms["GROUND"]));

    contact_pairs.push_back(ee_ground_contact);
    contact_pairs.push_back(ee_contact_pairs);
    contact_pairs.push_back(ground_object_contact_pairs);
    return contact_pairs;
  }

  std::string dairlib_root_;
  int n_x_ = 0;
  drake::multibody::MultibodyPlant<double>* plant_ = nullptr;
  drake::systems::Context<double>* plant_context_ = nullptr;
  std::unique_ptr<drake::multibody::MultibodyPlant<drake::AutoDiffXd>> plant_ad_;
  std::unique_ptr<drake::systems::Context<drake::AutoDiffXd>> context_ad_;
  std::unique_ptr<drake::systems::Diagram<double>> diagram_;
  std::unique_ptr<drake::systems::Context<double>> diagram_context_;
  std::unique_ptr<dairlib::systems::SamplingC3Controller> controller_;
  std::unique_ptr<drake::systems::Context<double>> controller_context_;
  std::unique_ptr<drake::systems::DiscreteValues<double>> discrete_state_;
};

}  // namespace

PYBIND11_MODULE(dairlib_sampling_c3, m) {
  m.attr("planner") = "dairlib.systems.SamplingC3Controller";
  m.attr("contact_model") = "dairlib.solvers.LCSFactory::LinearizePlantToLCS";
  py::class_<CompiledSamplingC3>(m, "CompiledSamplingC3")
      .def(py::init<const std::string&>(), py::arg("dairlib_root") = kDairlibRoot)
      .def_property_readonly("n_x", &CompiledSamplingC3::n_x)
      .def("step", &CompiledSamplingC3::Step, py::arg("x"), py::arg("x_des"),
           py::arg("x_final"), py::arg("time"));
}

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

// Kinematics functions
Eigen::Vector3f config_to_translationMatrix(float alpha, float beta, float arcLength) {
    // alpha can't be exactly 0
    if(alpha == 0)
        alpha = 0.000001;

    float x_temp = arcLength/alpha * (1-cos(alpha)) * sin(beta);
    float y_temp = arcLength/alpha * (1-cos(alpha)) * cos(beta);
    float z_temp = arcLength/alpha * sin(alpha);
    return Eigen::Vector3f(x_temp, y_temp, z_temp);
}

Eigen::Matrix3f config_to_rotationMatrix(float alpha, float beta) {
    // alpha can't be exactly 0
    if(alpha == 0)
        alpha = 0.000001;

    Eigen::Matrix3f Rot;
    Rot << cos(beta)*cos(beta)*(1-cos(alpha)) + cos(alpha), -cos(beta)*sin(beta)*(1-cos(alpha)), sin(alpha)*sin(beta),
           -cos(beta)*sin(beta)*(1-cos(alpha)), sin(beta)*sin(beta)*(1-cos(alpha)) + cos(alpha), sin(alpha)*cos(beta),
           -sin(alpha)*sin(beta), -sin(alpha)*cos(beta), cos(alpha);
    return Rot;
}

// Robot arm configuration structures
struct Pcc_config {
    float alpha = 0;
    float beta = 0;
};

struct Arm_cons {
    float seg_n = 0;
    float seg_len = 0;
};

class PyRobotArm {
private:
    Arm_cons arm_cons;
    Pcc_config section_1;
    Pcc_config section_2;
    Pcc_config section_3;

public:
    PyRobotArm(int n_segments, float segment_length) {
        arm_cons.seg_n = n_segments;
        arm_cons.seg_len = segment_length;
    }
    
    std::vector<float> get_end_effector_position() {
        Eigen::Vector3f pos = config_to_translationMatrix(
            section_3.alpha, section_3.beta, arm_cons.seg_len);
        return {pos[0], pos[1], pos[2]};
    }
    
    std::vector<float> get_end_effector_direction() {
        Eigen::Matrix3f rot = config_to_rotationMatrix(
            section_3.alpha, section_3.beta);
        Eigen::Vector3f dir = rot.col(2);
        return {dir[0], dir[1], dir[2]};
    }
    
    void set_config(const std::vector<float>& config) {
        if (config.size() != 6) {
            throw std::runtime_error("Config must have 6 values (alpha1, beta1, alpha2, beta2, alpha3, beta3)");
        }
        
        section_1.alpha = config[0];
        section_1.beta = config[1];
        section_2.alpha = config[2];
        section_2.beta = config[3];
        section_3.alpha = config[4];
        section_3.beta = config[5];
    }
    
    std::vector<std::vector<float>> get_positions() {
        std::vector<std::vector<float>> positions;
        
        // Base position
        positions.push_back({0.0f, 0.0f, 0.0f});
        
        // Section 1
        Eigen::Vector3f pos1 = config_to_translationMatrix(
            section_1.alpha, section_1.beta, arm_cons.seg_len);
        positions.push_back({pos1[0], pos1[1], pos1[2]});
        
        // Section 2
        Eigen::Matrix3f rot1 = config_to_rotationMatrix(section_1.alpha, section_1.beta);
        Eigen::Vector3f pos2 = pos1 + rot1 * config_to_translationMatrix(
            section_2.alpha, section_2.beta, arm_cons.seg_len);
        positions.push_back({pos2[0], pos2[1], pos2[2]});
        
        // Section 3
        Eigen::Matrix3f rot2 = rot1 * config_to_rotationMatrix(section_2.alpha, section_2.beta);
        Eigen::Vector3f pos3 = pos2 + rot2 * config_to_translationMatrix(
            section_3.alpha, section_3.beta, arm_cons.seg_len);
        positions.push_back({pos3[0], pos3[1], pos3[2]});
        
        return positions;
    }
};

PYBIND11_MODULE(cpp_robot_arm, m) {
    py::class_<PyRobotArm>(m, "RobotArm")
        .def(py::init<int, float>())
        .def("get_end_effector_position", &PyRobotArm::get_end_effector_position)
        .def("get_end_effector_direction", &PyRobotArm::get_end_effector_direction)
        .def("set_config", &PyRobotArm::set_config)
        .def("get_positions", &PyRobotArm::get_positions);
}

// Copyright 2025 Xiaojian Wu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <vector>
#include "../3rdparty/angles.h"

namespace kalman_hybird_lib {

/**
 * @brief Unscented Kalman Filter (UKF) implementation with support for angular dimensions.
 * 
 * This filter uses the Unscented Transform to approximate the prediction and update steps 
 * in non-linear state estimation problems. It supports automatic noise matrix updates and
 * handling of angle wrapping in selected state or measurement dimensions.
 */
template<int N_X, int N_Z, class PredictFunc, class MeasureFunc>
class UnscentedKalmanFilter {
public:
    using MatrixXX = Eigen::Matrix<double, N_X, N_X>;
    using MatrixZX = Eigen::Matrix<double, N_Z, N_X>;
    using MatrixXZ = Eigen::Matrix<double, N_X, N_Z>;
    using MatrixZZ = Eigen::Matrix<double, N_Z, N_Z>;
    using MatrixX1 = Eigen::Matrix<double, N_X, 1>;
    using MatrixZ1 = Eigen::Matrix<double, N_Z, 1>;

    using UpdateQFunc = std::function<MatrixXX()>;
    using UpdateRFunc = std::function<MatrixZZ(const MatrixZ1&)>;

    /**
     * @brief Construct a new Unscented Kalman Filter object.
     * 
     * @param f         Process model function.
     * @param h         Measurement model function.
     * @param u_q       Function to compute or update the process noise covariance Q.
     * @param u_r       Function to compute or update the measurement noise covariance R.
     * @param P0        Initial covariance matrix.
     * @param alpha     UKF scaling parameter (usually small, e.g., 1e-3).
     * @param beta      UKF prior knowledge of distribution (2 for Gaussian).
     * @param kappa     UKF secondary scaling parameter (0 is common).
     */
    explicit UnscentedKalmanFilter(
        const PredictFunc& f,
        const MeasureFunc& h,
        const UpdateQFunc& u_q,
        const UpdateRFunc& u_r,
        const MatrixXX& P0,
        double alpha = 1e-3,
        double beta = 2.0,
        double kappa = 0.0
    ) noexcept;

    /**
     * @brief Set the initial state vector.
     * 
     * @param x0 Initial state.
     */
    void setState(const MatrixX1& x0) noexcept;

    /**
     * @brief Specify which state or measurement dimensions represent angles.
     * These will be wrapped using shortest angular distance.
     * 
     * @param dims Indices of angle dimensions.
     */
    void setAngleDims(const std::vector<int>& dims);

    /**
     * @brief Get the predicted (prior) covariance matrix.
     */
    const MatrixXX& getPriorCovariance() const noexcept;

    /**
     * @brief Get the updated (posterior) covariance matrix.
     */
    const MatrixXX& getPosteriorCovariance() const noexcept;

    /**
     * @brief Perform the prediction step of the UKF.
     * 
     * @return Predicted state vector.
     */
    MatrixX1 predict() noexcept;

    /**
     * @brief Perform the update step of the UKF given a new measurement.
     * 
     * @param z Observation vector.
     * @return Updated state estimate.
     */
    MatrixX1 update(const MatrixZ1& z) noexcept;

private:
    PredictFunc f;                      ///< Process model
    MeasureFunc h;                      ///< Measurement model
    UpdateQFunc update_Q;               ///< Process noise generator
    UpdateRFunc update_R;               ///< Measurement noise generator

    double lambda = 0, gamma = 0;       ///< UKF scaling parameters
    std::array<double, 2 * N_X + 1> weights_mean {}; ///< Weights for mean
    std::array<double, 2 * N_X + 1> weights_cov {};  ///< Weights for covariance

    Eigen::Matrix<double, N_X, 2 * N_X + 1> Xsig;      ///< Sigma points
    Eigen::Matrix<double, N_X, 2 * N_X + 1> Xsig_pred; ///< Predicted sigma points

    MatrixXX Q = MatrixXX::Zero();      ///< Process noise covariance
    MatrixZZ R = MatrixZZ::Zero();      ///< Measurement noise covariance
    MatrixXZ K = MatrixXZ::Zero();      ///< Kalman gain

    MatrixXX P_pri = MatrixXX::Identity(); ///< Predicted covariance
    MatrixXX P_post = MatrixXX::Identity(); ///< Posterior covariance
    MatrixX1 x_pri = MatrixX1::Zero();      ///< Predicted state
    MatrixX1 x_post = MatrixX1::Zero();     ///< Posterior state

    std::vector<int> angle_dims_;       ///< Angle dimension indices

    /**
     * @brief Generate sigma points from the current state and covariance.
     * 
     * @param x          State vector.
     * @param P          Covariance matrix.
     * @param Xsig_out   Output sigma points matrix.
     */
    void generateSigmaPoints(
        const MatrixX1& x,
        const MatrixXX& P,
        Eigen::Matrix<double, N_X, 2 * N_X + 1>& Xsig_out
    );
};

}; // namespace kalman_hybird_lib

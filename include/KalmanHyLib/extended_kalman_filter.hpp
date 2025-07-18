// Copyright Chen Jun 2023. Licensed under the MIT License.
// Copyright xinyang 2021.
//
// Additional modifications and features by Chengfu Zou, Labor. Licensed under
// Apache License 2.0.
//
// Copyright (C) FYT Vision Group. All rights reserved.
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
#include <ceres/jet.h>
#include <functional>
#include "../3rdparty/angles.h"
namespace kalman_hybird_lib {
/**
 * @brief Extended Kalman Filter (EKF) implementation using Ceres Jet for automatic differentiation.
 * 
 * @tparam N_X        Dimension of the state vector.
 * @tparam N_Z        Dimension of the measurement vector.
 * @tparam PredicFunc Functor type for the process model f: x_{k+1} = f(x_k).
 * @tparam MeasureFunc Functor type for the measurement model h: z_k = h(x_k).
 */
template<int N_X, int N_Z, class PredicFunc, class MeasureFunc>
class ExtendedKalmanFilter {
public:
    ExtendedKalmanFilter() = default;

    // Alias for square matrices and vectors of appropriate dimensions
    using MatrixXX = Eigen::Matrix<double, N_X, N_X>;
    using MatrixZX = Eigen::Matrix<double, N_Z, N_X>;
    using MatrixXZ = Eigen::Matrix<double, N_X, N_Z>;
    using MatrixZZ = Eigen::Matrix<double, N_Z, N_Z>;
    using MatrixX1 = Eigen::Matrix<double, N_X, 1>;
    using MatrixZ1 = Eigen::Matrix<double, N_Z, 1>;

    // Functor types for updating process noise Q and measurement noise R
    using UpdateQFunc = std::function<MatrixXX()>;
    using UpdateRFunc = std::function<MatrixZZ(const MatrixZ1& z)>;

    /**
      * @brief Constructor initializing models, noise updaters, and prior covariance.
      *
      * @param f      Process model functor.
      * @param h      Measurement model functor.
      * @param u_q    Function to produce process noise covariance Q.
      * @param u_r    Function to produce measurement noise covariance R based on measurement z.
      * @param P0     Initial posterior covariance matrix.
      */
    explicit ExtendedKalmanFilter(
        const PredicFunc& f,
        const MeasureFunc& h,
        const UpdateQFunc& u_q,
        const UpdateRFunc& u_r,
        const MatrixXX& P0
    ) noexcept:
        f(f),
        h(h),
        update_Q(u_q),
        update_R(u_r),
        P_post(P0) {
        F.setZero(); // Initialize process Jacobian
        H.setZero(); // Initialize measurement Jacobian
    }

    /**
      * @brief Set the filter's initial state estimate.
      * @param x0 Initial state vector.
      */
    void setState(const MatrixX1& x0) noexcept {
        x_post = x0;
    }

    /**
      * @brief Override the process model functor.
      */
    void setPredictFunc(const PredicFunc& f) noexcept {
        this->f = f;
    }

    /**
      * @brief Override the measurement model functor.
      */
    void setMeasureFunc(const MeasureFunc& h) noexcept {
        this->h = h;
    }

    /**
      * @brief Set the number of Gauss-Newton iterations in the update step.
      * @param num Number of iterations.
      */
    void setIterationNum(int num) {
        iteration_num_ = num;
    }

    /**
      * @brief Specify which measurement dimensions represent angles.
      *        These will be normalized using the shortest angular distance.
      * @param dims Indices of angle components in measurement vector.
      */
    void setAngleDims(const std::vector<int>& dims) {
        angle_dims_ = dims;
    }

    /**
     * @brief Get the covariance matrix after prediction.
     */
    const MatrixXX& getPriorCovariance() const noexcept {
        return P_pri;
    }

    /**
     * @brief Get the covariance matrix after update.
     */
    const MatrixXX& getPosteriorCovariance() const noexcept {
        return P_post;
    }

    /**
    * @brief Get the L2 norm of the last measurement residual (after update).
    */
    double getResidualNorm() const noexcept {
        return last_residual_.norm();
    }

    /**
      * @brief Perform the prediction step of EKF.
      * @return Predicted state vector (x_prior).
      */
    MatrixX1 predict() noexcept {
        // Convert state to Ceres Jet for auto-diff
        ceres::Jet<double, N_X> x_e_jet[N_X];
        for (int i = 0; i < N_X; ++i) {
            x_e_jet[i].a = x_post[i]; // value
            x_e_jet[i].v.setZero(); // derivative vector
            x_e_jet[i].v[i] = 1.0; // set seed for Jacobian
        }

        // Evaluate process model
        ceres::Jet<double, N_X> x_p_jet[N_X];
        f(x_e_jet, x_p_jet);

        // Extract predicted state and Jacobian F
        for (int i = 0; i < N_X; ++i) {
            x_pri[i] = std::isfinite(x_p_jet[i].a) ? x_p_jet[i].a : 0.0;
            F.block(i, 0, 1, N_X) = x_p_jet[i].v.transpose();
        }

        // Compute process noise
        Q = update_Q();

        // Predict covariance: P_prior = F * P_post * F^T + Q
        P_pri = F * P_post * F.transpose() + Q;
        // Symmetrize
        P_pri = 0.5 * (P_pri + P_pri.transpose());

        // Update posterior state for next cycle
        x_post = x_pri;
        return x_pri;
    }

    /**
      * @brief Perform the update (correction) step of EKF given measurement z.
      *        Uses iterative Gauss-Newton if iteration_num_ > 1.
      * @param z Measurement vector at current time step.
      * @return Updated state estimate (x_post).
      */
    MatrixX1 update(const MatrixZ1& z) noexcept {
        // Start from the prior/posterior state
        MatrixX1 x_iter = x_post;

        // Optional iterative refinement
        for (int iter = 0; iter < iteration_num_; ++iter) {
            // Build Ceres Jet for current guess x_iter
            ceres::Jet<double, N_X> x_p_jet[N_X];
            for (int i = 0; i < N_X; ++i) {
                x_p_jet[i].a = x_iter[i];
                x_p_jet[i].v.setZero();
                x_p_jet[i].v[i] = 1.0;
            }

            // Evaluate measurement model
            ceres::Jet<double, N_X> z_p_jet[N_Z];
            h(x_p_jet, z_p_jet);

            // Extract predicted measurement and Jacobian H
            MatrixZ1 z_pri;
            for (int i = 0; i < N_Z; ++i) {
                z_pri[i] = std::isfinite(z_p_jet[i].a) ? z_p_jet[i].a : 0.0;
                H.block(i, 0, 1, N_X) = z_p_jet[i].v.transpose();
            }

            // Compute measurement noise
            R = update_R(z);

            // Innovation covariance S = H P_prior H^T + R
            MatrixZZ S = H * P_pri * H.transpose() + R;
            // Add small term for numerical stability
            S += small_noise_ * MatrixZZ::Identity();

            // Compute Kalman gain K = P_prior H^T S^{-1}
            K = P_pri * H.transpose() * S.inverse();

            // Compute residual and handle angle wrapping
            MatrixZ1 residual = z - z_pri;
            for (int idx: angle_dims_) {
                residual[idx] = angles::shortest_angular_distance(z_pri[idx], z[idx]);
            }
            // Clamp and sanitize residual
            for (int i = 0; i < N_Z; ++i) {
                if (!std::isfinite(residual[i]))
                    residual[i] = 0.0;
                residual[i] = std::clamp(residual[i], -1e2, 1e2);
            }

            // Update estimate: x_new = x_iter + K * residual
            MatrixX1 x_new = x_iter + K * residual;
            // Ensure finite
            for (int i = 0; i < N_X; ++i) {
                if (!std::isfinite(x_new[i]))
                    x_new[i] = x_iter[i];
            }

            x_iter = x_new;
            last_residual_ = residual;
        }

        // Final posterior state
        x_post = x_iter;
        for (int i = 0; i < N_X; ++i) {
            if (!std::isfinite(x_post[i]))
                x_post[i] = 0.0;
        }

        // Updated covariance: P_post = (I - K H) P_prior
        P_post = (MatrixXX::Identity() - K * H) * P_pri;
        // Symmetrize
        P_post = 0.5 * (P_post + P_post.transpose());

        return x_post;
    }

private:
    PredicFunc f; // Process model function
    MeasureFunc h; // Measurement model function
    UpdateQFunc update_Q; // Function to update process noise covariance
    UpdateRFunc update_R; // Function to update measurement noise covariance

    MatrixXX F = MatrixXX::Zero(); // Process Jacobian
    MatrixZX H = MatrixZX::Zero(); // Measurement Jacobian
    MatrixXX Q = MatrixXX::Zero(); // Process noise covariance
    MatrixZZ R = MatrixZZ::Zero(); // Measurement noise covariance

    MatrixXX P_pri = MatrixXX::Identity(); // Prior covariance
    MatrixXX P_post = MatrixXX::Identity(); // Posterior covariance
    MatrixXZ K = MatrixXZ::Zero(); // Kalman gain

    MatrixX1 x_pri = MatrixX1::Zero(); // Prior state
    MatrixX1 x_post = MatrixX1::Zero(); // Posterior state

    MatrixZ1 last_residual_ = MatrixZ1::Zero();

    std::vector<int> angle_dims_; // Indices of angular measurement dims
    int iteration_num_ = 1; // GN iterations in update
    double small_noise_ = 1e-6;
};
}

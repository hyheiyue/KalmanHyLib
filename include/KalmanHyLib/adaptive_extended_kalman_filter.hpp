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
#include <ceres/jet.h>
#include <functional>
#include <stdexcept>
#include <vector>
#include "../3rdparty/angles.h"
namespace kalmanLib
{
/**
 * @brief Adaptive Extended Kalman Filter (AEKF) with residual-based noise estimation.
 *
 * This filter supports partial adaptation by blending the predicted process noise and measurement noise
 * with prior noise levels. It also handles nonlinear measurement and process models via Ceres Jet auto-differentiation.
 *
 * @tparam N_X         Dimension of the state vector.
 * @tparam N_Z         Dimension of the measurement vector.
 * @tparam PredicFunc  Functor type for the process model: x_{k+1} = f(x_k).
 * @tparam MeasureFunc Functor type for the measurement model: z_k = h(x_k).
 */
template<int N_X, int N_Z, class PredicFunc, class MeasureFunc>
class AdaptiveExtendedKalmanFilter {
public:
    using MatrixXX = Eigen::Matrix<double, N_X, N_X>;
    using MatrixZX = Eigen::Matrix<double, N_Z, N_X>;
    using MatrixXZ = Eigen::Matrix<double, N_X, N_Z>;
    using MatrixZZ = Eigen::Matrix<double, N_Z, N_Z>;
    using MatrixX1 = Eigen::Matrix<double, N_X, 1>;
    using MatrixZ1 = Eigen::Matrix<double, N_Z, 1>;

    /// @brief Function type that returns the prior process noise covariance Q.
    using UpdateQFunc = std::function<MatrixXX()>;
    /// @brief Function type that returns the measurement noise covariance R given a measurement.
    using UpdateRFunc = std::function<MatrixZZ(const MatrixZ1&)>;

    /**
     * @brief Default constructor.
     */
    AdaptiveExtendedKalmanFilter() = default;

    /**
     * @brief Parameterized constructor.
     * @param f     Process model functor.
     * @param h     Measurement model functor.
     * @param u_q   Function to obtain prior process noise covariance Q.
     * @param u_r   Function to obtain prior measurement noise covariance R given z.
     * @param P0    Initial posterior covariance.
     */
    explicit AdaptiveExtendedKalmanFilter(
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
        F.setZero();
        H.setZero();
    }

    /**
     * @brief Set the filter's initial state.
     * @param x0 Initial state vector.
     */
    void setState(const MatrixX1& x0) noexcept {
        x_post = x0;
    }

    /**
     * @brief Update the process model functor.
     * @param f New process model.
     */
    void setPredictFunc(const PredicFunc& f) noexcept {
        this->f = f;
    }

    /**
     * @brief Update the measurement model functor.
     * @param h New measurement model.
     */
    void setMeasureFunc(const MeasureFunc& h) noexcept {
        this->h = h;
    }

    /**
     * @brief Set the number of iterations for the update step.
     * @param num Iteration count.
     */
    void setIterationNum(int num) {
        iteration_num_ = num;
    }

    /**
     * @brief Specify which state dimensions represent angles.
     * @param dims Indices of angle dimensions for wrapping.
     */
    void setAngleDims(const std::vector<int>& dims) {
        angle_dims_ = dims;
    }

    /**
     * @brief Set a small noise floor added to Q and R for stability.
     * @param n Noise floor value.
     */
    void setSmallnoise(double n) {
        small_noise_ = n;
    }

    /**
     * @brief Enable or disable adaptive process noise estimation.
     * @param enable True to enable adaptive Q.
     */
    void enableAdaptiveQ(bool enable) {
        adaptive_Q_enabled = enable;
    }

    /**
     * @brief Enable or disable adaptive measurement noise estimation.
     * @param enable True to enable adaptive R.
     */
    void enableAdaptiveR(bool enable) {
        adaptive_R_enabled = enable;
    }

    /**
     * @brief Set the blending factor for residual-based R adaptation.
     * @param a Weight between 0 and 1.
     */
    void setResidualAlpha(double a) {
        alpha = std::clamp(a, 0.0, 1.0);
    }

    /**
     * @brief Set the blending factor for Q adaptation.
     * @param beta Weight between 0 and 1.
     */
    void setAdaptiveQRatio(double beta) {
        beta_Q = std::clamp(beta, 0.0, 1.0);
    }

    /**
     * @brief Set the blending factor for R adaptation.
     * @param beta Weight between 0 and 1.
     */
    void setAdaptiveRRatio(double beta) {
        beta_R = std::clamp(beta, 0.0, 1.0);
    }

    /**
     * @brief Prediction step: propagate state and covariance.
     * @return Predicted state vector x_pri.
     */
    MatrixX1 predict() noexcept {
        // Auto-differentiate process model to compute F jacobian
        ceres::Jet<double, N_X> x_e_jet[N_X];
        for (int i = 0; i < N_X; ++i) {
            x_e_jet[i].a = x_post[i];
            x_e_jet[i].v.setZero();
            x_e_jet[i].v[i] = 1.0;
        }

        ceres::Jet<double, N_X> x_p_jet[N_X];
        f(x_e_jet, x_p_jet);

        for (int i = 0; i < N_X; ++i) {
            x_pri[i] = std::isfinite(x_p_jet[i].a) ? x_p_jet[i].a : 0.0;
            F.block(i, 0, 1, N_X) = x_p_jet[i].v.transpose();
        }

        // Adaptive process noise Q
        if (adaptive_Q_enabled) {
            process_noise_est_ = x_pri - x_post;
            process_noise_est_.unaryExpr([](double v) { return std::isfinite(v) ? v : 0.0; });
            MatrixXX Q_adapt = process_noise_est_ * process_noise_est_.transpose();
            MatrixXX Q_prior = update_Q();
            Q = beta_Q * Q_adapt + (1.0 - beta_Q) * Q_prior;
            Q += small_noise_ * MatrixXX::Identity();
        } else {
            Q = update_Q();
        }

        // Covariance propagation
        P_pri = F * P_post * F.transpose() + Q;
        P_pri = 0.5 * (P_pri + P_pri.transpose());

        x_post = x_pri;
        return x_pri;
    }

    /**
     * @brief Update step: incorporate measurement and refine state.
     * @param z Measurement vector.
     * @return Updated state vector x_post.
     */
    MatrixX1 update(const MatrixZ1& z) noexcept {
        MatrixX1 x_iter = x_post;

        for (int iter = 0; iter < iteration_num_; ++iter) {
            // Auto-differentiate measurement model to compute H jacobian
            ceres::Jet<double, N_X> x_p_jet[N_X];
            for (int i = 0; i < N_X; ++i) {
                x_p_jet[i].a = x_iter[i];
                x_p_jet[i].v.setZero();
                x_p_jet[i].v[i] = 1.0;
            }

            ceres::Jet<double, N_X> z_p_jet[N_Z];
            h(x_p_jet, z_p_jet);

            MatrixZ1 z_pri;
            for (int i = 0; i < N_Z; ++i) {
                z_pri[i] = std::isfinite(z_p_jet[i].a) ? z_p_jet[i].a : 0.0;
                H.block(i, 0, 1, N_X) = z_p_jet[i].v.transpose();
            }

            // Compute residual and handle angular dimensions
            MatrixZ1 residual = z - z_pri;
            for (int idx: angle_dims_) {
                residual[idx] = angles::shortest_angular_distance(z_pri[idx], z[idx]);
            }
            residual.unaryExpr([](double v) {
                if (!std::isfinite(v))
                    return 0.0;
                return std::clamp(v, -1e2, 1e2);
            });
            last_residual_ = residual;

            // Adaptive measurement noise R
            if (adaptive_R_enabled) {
                MatrixZZ R_adapt = residual * residual.transpose();
                MatrixZZ R_prior = update_R(z);
                R = beta_R * R_adapt + (1.0 - beta_R) * R_prior;
                R += small_noise_ * MatrixZZ::Identity();
            } else {
                R = update_R(z);
            }

            // Kalman gain and state update
            MatrixZZ S = H * P_pri * H.transpose() + R;
            S += small_noise_ * MatrixZZ::Identity();
            K = P_pri * H.transpose() * S.inverse();

            MatrixX1 x_new = x_iter + K * residual;
            x_new.unaryExpr([&](double v) { return std::isfinite(v) ? v : x_iter; });
            x_iter = x_new;
        }

        // Finalize post-update state and covariance
        x_post = x_iter;
        x_post.unaryExpr([](double v) { return std::isfinite(v) ? v : 0.0; });
        P_post = (MatrixXX::Identity() - K * H) * P_pri;
        P_post = 0.5 * (P_post + P_post.transpose());
        return x_post;
    }

private:
    PredicFunc f; ///< Process model functor
    MeasureFunc h; ///< Measurement model functor
    UpdateQFunc update_Q; ///< Function to obtain Q_prior
    UpdateRFunc update_R; ///< Function to obtain R_prior(z)

    MatrixXX F = MatrixXX::Zero(); ///< State transition jacobian
    MatrixZX H = MatrixZX::Zero(); ///< Measurement jacobian
    MatrixXX Q = MatrixXX::Zero(); ///< Process noise covariance
    MatrixZZ R = MatrixZZ::Zero(); ///< Measurement noise covariance
    MatrixXX P_pri = MatrixXX::Identity(); ///< Prior covariance
    MatrixXX P_post = MatrixXX::Identity(); ///< Posterior covariance
    MatrixXZ K = MatrixXZ::Zero(); ///< Kalman gain
    MatrixX1 x_pri = MatrixX1::Zero(); ///< Prior state
    MatrixX1 x_post = MatrixX1::Zero(); ///< Posterior state

    MatrixZ1 last_residual_ = MatrixZ1::Zero(); ///< Last measurement residual
    MatrixX1 process_noise_est_ = MatrixX1::Zero(); ///< Estimated process noise

    std::vector<int> angle_dims_; ///< Indices of angular state components
    int iteration_num_ = 1; ///< Number of update iterations
    bool adaptive_Q_enabled = false; ///< Enable adaptive Q
    bool adaptive_R_enabled = false; ///< Enable adaptive R
    double alpha = 0.5; ///< Residual blending factor for R
    double beta_Q = 1.0; ///< Blending factor for Q adaptation
    double beta_R = 1.0; ///< Blending factor for R adaptation
    double small_noise_ = 1e-6; ///< Small noise floor for stability
};

}

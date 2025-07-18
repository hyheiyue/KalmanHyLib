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
 * @brief Unscented Kalman Filter (UKF) implementation with optional Gauss-Newton style iterative update.
 * 
 * This UKF uses the Unscented Transform to handle nonlinear process and measurement models.
 * An optional iterative update can refine the posterior state estimate via multiple measurement updates.
 * Supports handling of angular dimensions with shortest angular distance wrap-around.
 * 
 * @tparam N_X          Dimension of the state vector.
 * @tparam N_Z          Dimension of the measurement vector.
 * @tparam PredictFunc  Functor type for the process model: x_{k+1} = f(x_k).
 * @tparam MeasureFunc  Functor type for the measurement model: z_k = h(x_k).
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
     * @brief Construct the Unscented Kalman Filter.
     * 
     * @param f         Process model functor.
     * @param h         Measurement model functor.
     * @param u_q       Function to compute process noise covariance Q.
     * @param u_r       Function to compute measurement noise covariance R given measurement.
     * @param P0        Initial posterior covariance matrix.
     * @param alpha     UKF scaling parameter (default 1e-3).
     * @param beta      UKF prior knowledge parameter (default 2 for Gaussian).
     * @param kappa     UKF secondary scaling parameter (default 0).
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
    ) noexcept:
        f(f),
        h(h),
        update_Q(u_q),
        update_R(u_r),
        P_post(P0)
    {
        lambda = alpha * alpha * (N_X + kappa) - N_X;
        gamma = std::sqrt(N_X + lambda);

        weights_mean[0] = lambda / (N_X + lambda);
        weights_cov[0] = weights_mean[0] + (1 - alpha * alpha + beta);
        for (int i = 1; i < 2 * N_X + 1; ++i) {
            weights_mean[i] = weights_cov[i] = 1.0 / (2 * (N_X + lambda));
        }

        Xsig_pred.setZero();
    }

    /**
     * @brief Set the initial state estimate.
     * @param x0 Initial state vector.
     */
    void setState(const MatrixX1& x0) noexcept {
        x_post = x0;
    }

    /**
     * @brief Specify which indices in state or measurement vector represent angles.
     *        Those dimensions will be wrapped via shortest angular distance.
     * @param dims Vector of angle dimension indices.
     */
    void setAngleDims(const std::vector<int>& dims) {
        angle_dims_ = dims;
    }

    /**
     * @brief Set the number of Gauss-Newton style iterations during update.
     *        Minimum is 1 (standard UKF update).
     * @param num Number of iterations.
     */
    void setIterationNum(int num) {
        iteration_num_ = std::max(1, num);
    }

    /**
     * @brief Get the predicted (prior) covariance matrix.
     * @return Prior covariance.
     */
    const MatrixXX& getPriorCovariance() const noexcept {
        return P_pri;
    }

    /**
     * @brief Get the updated (posterior) covariance matrix.
     * @return Posterior covariance.
     */
    const MatrixXX& getPosteriorCovariance() const noexcept {
        return P_post;
    }

    /**
     * @brief Perform the prediction step of UKF.
     * @return Predicted (prior) state vector.
     */
    MatrixX1 predict() noexcept {
        Q = update_Q();
        generateSigmaPoints(x_post, P_post, Xsig);

        for (int i = 0; i < 2 * N_X + 1; ++i)
            Xsig_pred.col(i) = f(Xsig.col(i));

        x_pri.setZero();
        for (int i = 0; i < 2 * N_X + 1; ++i)
            x_pri += weights_mean[i] * Xsig_pred.col(i);

        P_pri.setZero();
        for (int i = 0; i < 2 * N_X + 1; ++i) {
            auto dx = Xsig_pred.col(i) - x_pri;
            P_pri += weights_cov[i] * dx * dx.transpose();
        }
        P_pri += Q;

        x_post = x_pri;
        return x_pri;
    }

    /**
     * @brief Perform the measurement update step with iterative refinement.
     * 
     * The update iteratively refines the posterior state estimate by repeatedly
     * generating sigma points around the current estimate and applying the measurement update.
     * 
     * @param z Measurement vector.
     * @return Updated (posterior) state vector.
     */
    MatrixX1 update(const MatrixZ1& z) noexcept {
        R = update_R(z);

        // Initialize iterative update state with prior mean
        MatrixX1 x_iter = x_pri;

        for (int iter = 0; iter < iteration_num_; ++iter) {
            // Generate sigma points around current estimate
            generateSigmaPoints(x_iter, P_pri, Xsig);

            // Predict measurement sigma points
            Eigen::Matrix<double, N_Z, 2 * N_X + 1> Zsig;
            for (int i = 0; i < 2 * N_X + 1; ++i)
                Zsig.col(i) = h(Xsig.col(i));

            // Calculate predicted measurement mean
            MatrixZ1 z_pred = MatrixZ1::Zero();
            for (int i = 0; i < 2 * N_X + 1; ++i)
                z_pred += weights_mean[i] * Zsig.col(i);

            // Calculate innovation covariance matrix S
            MatrixZZ S = MatrixZZ::Zero();
            for (int i = 0; i < 2 * N_X + 1; ++i) {
                MatrixZ1 dz = Zsig.col(i) - z_pred;
                for (int idx : angle_dims_)
                    dz[idx] = angles::shortest_angular_distance(z_pred[idx], Zsig.col(i)[idx]);
                S += weights_cov[i] * dz * dz.transpose();
            }
            S += R;

            // Calculate cross covariance matrix Tc
            MatrixXZ Tc = MatrixXZ::Zero();
            for (int i = 0; i < 2 * N_X + 1; ++i) {
                MatrixX1 dx = Xsig.col(i) - x_iter;
                MatrixZ1 dz = Zsig.col(i) - z_pred;
                for (int idx : angle_dims_)
                    dz[idx] = angles::shortest_angular_distance(z_pred[idx], Zsig.col(i)[idx]);
                Tc += weights_cov[i] * dx * dz.transpose();
            }

            // Calculate Kalman gain
            MatrixXZ K_iter = Tc * S.inverse();

            // Calculate residual (measurement innovation) with angle wrapping
            MatrixZ1 residual = z - z_pred;
            for (int idx : angle_dims_)
                residual[idx] = angles::shortest_angular_distance(z_pred[idx], z[idx]);
            for (int i = 0; i < N_Z; ++i) {
                if (!std::isfinite(residual[i]))
                    residual[i] = 0.0;
                residual[i] = std::clamp(residual[i], -1e2, 1e2);
            }

            // Update state estimate
            MatrixX1 x_new = x_iter + K_iter * residual;
            for (int i = 0; i < N_X; ++i) {
                if (!std::isfinite(x_new[i]))
                    x_new[i] = x_iter[i];
            }

            x_iter = x_new;
        }

        // Save final updated state
        x_post = x_iter;

        // Recompute final Kalman gain and covariance update
        generateSigmaPoints(x_post, P_pri, Xsig);
        Eigen::Matrix<double, N_Z, 2 * N_X + 1> Zsig_final;
        for (int i = 0; i < 2 * N_X + 1; ++i)
            Zsig_final.col(i) = h(Xsig.col(i));

        MatrixZ1 z_pred_final = MatrixZ1::Zero();
        for (int i = 0; i < 2 * N_X + 1; ++i)
            z_pred_final += weights_mean[i] * Zsig_final.col(i);

        MatrixZZ S_final = MatrixZZ::Zero();
        for (int i = 0; i < 2 * N_X + 1; ++i) {
            MatrixZ1 dz = Zsig_final.col(i) - z_pred_final;
            for (int idx : angle_dims_)
                dz[idx] = angles::shortest_angular_distance(z_pred_final[idx], Zsig_final.col(i)[idx]);
            S_final += weights_cov[i] * dz * dz.transpose();
        }
        S_final += R;

        MatrixXZ Tc_final = MatrixXZ::Zero();
        for (int i = 0; i < 2 * N_X + 1; ++i) {
            MatrixX1 dx = Xsig.col(i) - x_post;
            MatrixZ1 dz = Zsig_final.col(i) - z_pred_final;
            for (int idx : angle_dims_)
                dz[idx] = angles::shortest_angular_distance(z_pred_final[idx], Zsig_final.col(i)[idx]);
            Tc_final += weights_cov[i] * dx * dz.transpose();
        }

        K = Tc_final * S_final.inverse();

        // Update covariance
        P_post = P_pri - K * S_final * K.transpose();
        // Symmetrize covariance matrix
        P_post = 0.5 * (P_post + P_post.transpose());

        return x_post;
    }

private:
    PredictFunc f;             ///< Process model function
    MeasureFunc h;             ///< Measurement model function
    UpdateQFunc update_Q;      ///< Process noise covariance updater
    UpdateRFunc update_R;      ///< Measurement noise covariance updater

    double lambda = 0;         ///< UKF scaling parameter lambda
    double gamma = 0;          ///< Square root of (N_X + lambda)
    std::array<double, 2 * N_X + 1> weights_mean {};  ///< Sigma point weights for mean
    std::array<double, 2 * N_X + 1> weights_cov {};   ///< Sigma point weights for covariance

    Eigen::Matrix<double, N_X, 2 * N_X + 1> Xsig;      ///< Sigma points matrix
    Eigen::Matrix<double, N_X, 2 * N_X + 1> Xsig_pred; ///< Predicted sigma points matrix

    MatrixXX Q = MatrixXX::Zero();   ///< Process noise covariance
    MatrixXX P_pri = MatrixXX::Identity();   ///< Prior covariance
    MatrixXX P_post = MatrixXX::Identity();  ///< Posterior covariance

    MatrixZZ R = MatrixZZ::Zero();   ///< Measurement noise covariance
    MatrixXZ K = MatrixXZ::Zero();   ///< Kalman gain

    MatrixX1 x_pri = MatrixX1::Zero();  ///< Predicted (prior) state
    MatrixX1 x_post = MatrixX1::Zero(); ///< Updated (posterior) state

    std::vector<int> angle_dims_;  ///< Indices of angular dimensions to wrap

    int iteration_num_ = 1; ///< Number of iterations during update (>=1)

    /**
     * @brief Generate sigma points from state and covariance.
     * @param x State vector.
     * @param P Covariance matrix.
     * @param Xsig_out Output sigma points matrix (size N_X x (2*N_X+1)).
     */
    void generateSigmaPoints(
        const MatrixX1& x,
        const MatrixXX& P,
        Eigen::Matrix<double, N_X, 2 * N_X + 1>& Xsig_out
    ) {
        Eigen::Matrix<double, N_X, N_X> A = P.llt().matrixL();
        Xsig_out.col(0) = x;
        for (int i = 0; i < N_X; ++i) {
            Xsig_out.col(i + 1) = x + gamma * A.col(i);
            Xsig_out.col(i + 1 + N_X) = x - gamma * A.col(i);
        }
    }
};

} // namespace kalman_hybird_lib

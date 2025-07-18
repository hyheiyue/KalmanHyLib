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
#include <vector>
#include "../3rdparty/angles.h"

namespace kalman_hybird_lib {
/**
  * @brief Error-State Extended Kalman Filter with covariance accessors.
  *
  * @tparam N_X Dimension of the state vector.
  * @tparam N_Z Dimension of the measurement vector.
  * @tparam PredicFunc Functor type for the process model.
  * @tparam MeasureFunc Functor type for the measurement model.
  */
template<int N_X, int N_Z, class PredicFunc, class MeasureFunc>
class ErrorStateEKF {
public:
    using MatrixXX = Eigen::Matrix<double, N_X, N_X>;
    using MatrixZX = Eigen::Matrix<double, N_Z, N_X>;
    using MatrixXZ = Eigen::Matrix<double, N_X, N_Z>;
    using MatrixZZ = Eigen::Matrix<double, N_Z, N_Z>;
    using MatrixX1 = Eigen::Matrix<double, N_X, 1>;
    using MatrixZ1 = Eigen::Matrix<double, N_Z, 1>;

    using UpdateQFunc = std::function<MatrixXX()>;
    using UpdateRFunc = std::function<MatrixZZ(const MatrixZ1&)>;

    explicit ErrorStateEKF(
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
        P_delta(P0),
        P_delta_pri(P0) {
        F.setZero();
        H.setZero();
    }

    void setInjectFunc(std::function<void(const MatrixX1&, MatrixX1&)> inject_func) {
        inject_state = inject_func;
    }

    void setState(const MatrixX1& x0) noexcept {
        x_nominal = x0;
        delta_x.setZero();
    }

    void setPredictFunc(const PredicFunc& f) noexcept {
        this->f = f;
    }
    void setMeasureFunc(const MeasureFunc& h) noexcept {
        this->h = h;
    }
    void setIterationNum(int num) {
        iteration_num_ = num;
    }
    void setAngleDims(const std::vector<int>& dims) {
        angle_dims_ = dims;
    }
    /**
      * @brief Get prior error-state covariance (after predict, before update).
      * @return Reference to P_delta_pri.
      */
    const MatrixXX& getPriorCovariance() const noexcept {
        return P_delta_pri;
    }

    /**
     * @brief Get posterior error-state covariance (after update).
     * @return Reference to P_delta.
     */
    const MatrixXX& getPosteriorCovariance() const noexcept {
        return P_delta;
    }

    /**
     * @brief Get current nominal state.
     */
    const MatrixX1& getState() const noexcept {
        return x_nominal;
    }

    /**
    * @brief Get the L2 norm of the last measurement residual (after update).
    */
    double getResidualNorm() const noexcept {
        return last_residual_.norm();
    }

    /**
      * @brief Prediction step: propagate nominal state and error covariance.
      * @return Predicted nominal state.
      */
    MatrixX1 predict() noexcept {
        // Auto-diff for process model
        ceres::Jet<double, N_X> x_jet[N_X];
        for (int i = 0; i < N_X; ++i) {
            x_jet[i].a = x_nominal[i];
            x_jet[i].v.setZero();
            x_jet[i].v[i] = 1.0;
        }
        ceres::Jet<double, N_X> x_pred_jet[N_X];
        f(x_jet, x_pred_jet);

        MatrixX1 x_pri;
        for (int i = 0; i < N_X; ++i) {
            x_pri[i] = x_pred_jet[i].a;
            F.block(i, 0, 1, N_X) = x_pred_jet[i].v.transpose();
        }

        // Update error covariance prior
        Q = update_Q();
        P_delta_pri = F * P_delta * F.transpose() + Q;
        P_delta_pri = 0.5 * (P_delta_pri + P_delta_pri.transpose());

        // Set for next cycle
        P_delta = P_delta_pri;
        x_nominal = x_pri;
        delta_x.setZero();

        return x_pri;
    }

    /**
      * @brief Update step: correct nominal state with measurement.
      * @param z Measurement vector.
      * @return Updated nominal state.
      */
    MatrixX1 update(const MatrixZ1& z) noexcept {
        MatrixX1 delta_iter = delta_x;
        MatrixXX P_iter = P_delta;

        for (int iter = 0; iter < iteration_num_; ++iter) {
            // Inject error into nominal
            MatrixX1 x_full = x_nominal;
            if (inject_state)
                inject_state(delta_iter, x_full);

            // Auto-diff for measurement model
            ceres::Jet<double, N_X> x_jet[N_X];
            for (int i = 0; i < N_X; ++i) {
                x_jet[i].a = x_full[i];
                x_jet[i].v.setZero();
                x_jet[i].v[i] = 1.0;
            }
            ceres::Jet<double, N_X> z_jet[N_Z];
            h(x_jet, z_jet);

            MatrixZ1 z_pred;
            for (int i = 0; i < N_Z; ++i) {
                z_pred[i] = z_jet[i].a;
                H.block(i, 0, 1, N_X) = z_jet[i].v.transpose();
            }

            R = update_R(z);
            MatrixZZ S = H * P_iter * H.transpose() + R;
            S += small_noise_ * MatrixZZ::Identity();
            MatrixXZ K = P_iter * H.transpose() * S.inverse();

            MatrixZ1 residual = z - z_pred;
            for (int idx: angle_dims_) {
                residual[idx] = angles::shortest_angular_distance(z_pred[idx], z[idx]);
            }
            for (int i = 0; i < N_Z; ++i) {
                residual[i] = std::clamp(residual[i], -1e2, 1e2);
            }

            delta_iter += K * residual;
            P_iter = (MatrixXX::Identity() - K * H) * P_iter;
            P_iter = 0.5 * (P_iter + P_iter.transpose());
            last_residual_ = residual;
        }

        // Final injection
        if (inject_state)
            inject_state(delta_iter, x_nominal);
        delta_x.setZero();
        P_delta = P_iter;

        return x_nominal;
    }

private:
    PredicFunc f;
    MeasureFunc h;
    UpdateQFunc update_Q;
    UpdateRFunc update_R;

    MatrixXX F = MatrixXX::Zero();
    MatrixZX H = MatrixZX::Zero();
    MatrixXX Q = MatrixXX::Zero();
    MatrixZZ R = MatrixZZ::Zero();

    MatrixX1 x_nominal = MatrixX1::Zero();
    MatrixX1 delta_x = MatrixX1::Zero();
    MatrixXX P_delta = MatrixXX::Identity();
    MatrixXX P_delta_pri = MatrixXX::Identity();

    MatrixZ1 last_residual_ = MatrixZ1::Zero();

    std::vector<int> angle_dims_;
    int iteration_num_ = 1;
    double small_noise_ = 1e-6;
    std::function<void(const MatrixX1&, MatrixX1&)> inject_state;
};

}

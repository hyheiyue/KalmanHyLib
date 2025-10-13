#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <ceres/jet.h>
#include <cmath>
#include <deque>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

namespace kalman_hybird_lib {
/**
  * @brief Error-State Extended Kalman Filter with covariance accessors and online NIS/NEES anomaly detection.
  *
  * @tparam N_X Dimension of the state vector.
  * @tparam N_Z Dimension of the measurement vector.
  * @tparam PredicFunc Functor type for the process model.
  * @tparam MeasureFunc Functor type for the measurement model.
  */
template<int N_X, int N_Z, class PredicFunc, class MeasureFunc>
class ErrorStateEKF {
public:
    ErrorStateEKF() = default;
    using MatrixXX = Eigen::Matrix<double, N_X, N_X>;
    using MatrixZX = Eigen::Matrix<double, N_Z, N_X>;
    using MatrixXZ = Eigen::Matrix<double, N_X, N_Z>;
    using MatrixZZ = Eigen::Matrix<double, N_Z, N_Z>;
    using MatrixX1 = Eigen::Matrix<double, N_X, 1>;
    using MatrixZ1 = Eigen::Matrix<double, N_Z, 1>;

    using UpdateQFunc = std::function<MatrixXX()>;
    using UpdateRFunc = std::function<MatrixZZ(const MatrixZ1&)>;
    using InjectFunc = std::function<void(const MatrixX1&, MatrixX1&)>;
    using ResidualFunc = std::function<MatrixZ1(const MatrixZ1& z_pred, const MatrixZ1& z_meas)>;

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

        // default anomaly detection params
        window_size_ = 50;
        recent_fail_rate_threshold_ = 0.4;
        nis_threshold_.reset();
        nees_threshold_.reset();

        residual_func_ = [](const MatrixZ1& z_pred, const MatrixZ1& z_meas) -> MatrixZ1 {
            return z_meas - z_pred;
        };
    }

    void setInjectFunc(const InjectFunc& inject_func) {
        inject_state = inject_func;
    }

    void setState(const MatrixX1& x0) noexcept {
        x_nominal = x0;
        delta_x.setZero();
        x_nominal_pri.setZero();
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

    /**
      * @brief 设置自定义残差计算函数。
      * @param func 函数签名：MatrixZ1(const MatrixZ1& z_pred, const MatrixZ1& z_meas)
      *             返回值是残差向量（通常是 z_meas - z_pred 的某种变体）。
      *             若未设置，则使用默认实现（元素相减）。
      */
    void setResidualFunc(const ResidualFunc& func) {
        residual_func_ = func;
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
    void setUpdateQ(const UpdateQFunc& u_q) {
        this->update_Q = u_q;
    }

    void setUpdateR(const UpdateRFunc& u_r) {
        this->update_R = u_r;
    }
    // ----------------- anomaly detection API -----------------
    void setNisThreshold(double t) {
        nis_threshold_ = t;
    }
    void clearNisThreshold() {
        nis_threshold_.reset();
    }
    void setNeesThreshold(double t) {
        nees_threshold_ = t;
    }
    void clearNeesThreshold() {
        nees_threshold_.reset();
    }

    void setWindowSize(size_t w) {
        window_size_ = w;
        while (recent_nis_failures_.size() > window_size_)
            recent_nis_failures_.pop_front();
    }

    void setRecentFailRateThreshold(double rate) {
        recent_fail_rate_threshold_ = rate;
    }

    double lastNis() const noexcept {
        return last_nis_;
    }
    double lastNees() const noexcept {
        return last_nees_;
    }
    int totalChecks() const noexcept {
        return total_count_;
    }
    int nisFailureCount() const noexcept {
        return nis_count_;
    }
    int neesFailureCount() const noexcept {
        return nees_count_;
    }

    double recentNisFailureRate() const noexcept {
        if (recent_nis_failures_.empty())
            return 0.0;
        int sum = std::accumulate(recent_nis_failures_.begin(), recent_nis_failures_.end(), 0);
        return static_cast<double>(sum) / static_cast<double>(recent_nis_failures_.size());
    }

    bool isRecentlyInconsistent() const noexcept {
        return recentNisFailureRate() >= recent_fail_rate_threshold_;
    }
    // ----------------- end API -----------------

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

        // Set for next cycle: store prior nominal for NEES later
        x_nominal_pri = x_nominal;
        // update buffers
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
        MatrixZZ S_last = MatrixZZ::Zero(); // store last S for NIS
        MatrixZ1 residual_last = MatrixZ1::Zero();

        double prev_res_norm = std::numeric_limits<double>::max();
        MatrixXZ K_last;

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
            K_last = K;

            MatrixZ1 residual = residual_func_(z_pred, z);

            for (int i = 0; i < N_Z; ++i) {
                if (!std::isfinite(residual[i]))
                    residual[i] = 0.0;
                residual[i] = std::clamp(residual[i], -1e2, 1e2);
            }

            double alpha = 1.0;
            double cur_res_norm = residual.norm();
            if (cur_res_norm > prev_res_norm) {
                alpha = 0.5; // 阻尼
            }

            delta_iter += alpha * K * residual;

            double old_res_norm = prev_res_norm;
            prev_res_norm = cur_res_norm;

            last_residual_ = residual;
            S_last = S;
            residual_last = residual;

            if (cur_res_norm < 1e-4 || std::abs(old_res_norm - cur_res_norm) < 1e-6) {
                break;
            }
        }

        // ====== 最终注入和协方差更新 (Joseph形式) ======
        if (inject_state)
            inject_state(delta_iter, x_nominal);
        delta_x.setZero();

        P_delta = (MatrixXX::Identity() - K_last * H) * P_iter
                * (MatrixXX::Identity() - K_last * H).transpose()
            + K_last * R * K_last.transpose();
        P_delta = 0.5 * (P_delta + P_delta.transpose()); // 对称化

        // ----------------- Anomaly detection (NIS / NEES) -----------------
        double nis = std::numeric_limits<double>::quiet_NaN();
        double nees = std::numeric_limits<double>::quiet_NaN();
        try {
            Eigen::VectorXd tmp = S_last.ldlt().solve(residual_last);
            nis = static_cast<double>(residual_last.transpose() * tmp);
        } catch (...) {
            nis = std::numeric_limits<double>::quiet_NaN();
        }

        try {
            MatrixX1 dx = x_nominal - x_nominal_pri;
            Eigen::VectorXd tmp2 = P_delta.ldlt().solve(dx);
            nees = static_cast<double>(dx.transpose() * tmp2);
        } catch (...) {
            nees = std::numeric_limits<double>::quiet_NaN();
        }

        last_nis_ = nis;
        last_nees_ = nees;
        total_count_++;

        bool nis_failed = false;
        bool nees_failed = false;
        if (nis_threshold_.has_value() && std::isfinite(nis)) {
            nis_failed = (nis > nis_threshold_.value());
            if (nis_failed)
                nis_count_++;
        }
        if (nees_threshold_.has_value() && std::isfinite(nees)) {
            nees_failed = (nees > nees_threshold_.value());
            if (nees_failed)
                nees_count_++;
        }

        if (std::isfinite(nis)) {
            recent_nis_failures_.push_back(nis_failed ? 1 : 0);
            if (recent_nis_failures_.size() > window_size_) {
                recent_nis_failures_.pop_front();
            }
        }

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
    MatrixX1 x_nominal_pri = MatrixX1::Zero(); // store prior nominal for NEES
    MatrixX1 delta_x = MatrixX1::Zero();
    MatrixXX P_delta = MatrixXX::Identity();
    MatrixXX P_delta_pri = MatrixXX::Identity();

    MatrixZ1 last_residual_ = MatrixZ1::Zero();

    int iteration_num_ = 1;
    double small_noise_ = 1e-6;
    InjectFunc inject_state;

    // 残差函数成员（默认为 z_meas - z_pred）
    ResidualFunc residual_func_;

    // anomaly detection members
    size_t window_size_ = 50;
    std::deque<int> recent_nis_failures_;
    double recent_fail_rate_threshold_ = 0.4;

    std::optional<double> nis_threshold_;
    std::optional<double> nees_threshold_;

    int nis_count_ = 0;
    int nees_count_ = 0;
    int total_count_ = 0;

    double last_nis_ = std::numeric_limits<double>::quiet_NaN();
    double last_nees_ = std::numeric_limits<double>::quiet_NaN();
};

} // namespace kalman_hybird_lib

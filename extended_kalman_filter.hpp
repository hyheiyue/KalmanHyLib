#pragma once


#include <Eigen/Dense>
#include <ceres/jet.h>
#include <cmath>
#include <deque>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>

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

    // Residual function type: user provides residual = f(z_pred, z_meas)
    using ResidualFunc = std::function<MatrixZ1(const MatrixZ1& z_pred, const MatrixZ1& z_meas)>;

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

        // default values for anomaly detection
        window_size_ = 50;
        recent_fail_rate_threshold_ = 0.4;
        // thresholds not set -> optional empty (user should set proper chi-square thresholds if desired)
        nis_threshold_.reset();
        nees_threshold_.reset();

        // default residual: simple difference z_meas - z_pred
        residual_func_ = [](const MatrixZ1& z_pred, const MatrixZ1& z_meas) -> MatrixZ1 {
            return z_meas - z_pred;
        };
    }

    // ---- state / model setters ----
    void setState(const MatrixX1& x0) noexcept {
        x_post = x0;
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

    // new: allow user to set custom residual computation
    void setResidualFunc(const ResidualFunc& func) {
        residual_func_ = func;
    }

    const MatrixXX& getPriorCovariance() const noexcept {
        return P_pri;
    }

    const MatrixXX& getPosteriorCovariance() const noexcept {
        return P_post;
    }

    double getResidualNorm() const noexcept {
        return last_residual_.norm();
    }

    void setUpdateQ(const UpdateQFunc& u_q) {
        this->update_Q = u_q;
    }

    void setUpdateR(const UpdateRFunc& u_r) {
        this->update_R = u_r;
    }

    // ---- anomaly detection setters/getters ----

    // Set explicit thresholds (e.g. chi-square critical values)
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
        // trim deque if necessary
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

    // recent failure rate in [0,1], returns 0 if no samples
    double recentNisFailureRate() const noexcept {
        if (recent_nis_failures_.empty())
            return 0.0;
        int sum = std::accumulate(recent_nis_failures_.begin(), recent_nis_failures_.end(), 0);
        return static_cast<double>(sum) / static_cast<double>(recent_nis_failures_.size());
    }

    // returns true if recent failure rate exceeds configured threshold
    bool isRecentlyInconsistent() const noexcept {
        return recentNisFailureRate() >= recent_fail_rate_threshold_;
    }

    // perform the prediction step
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

    // ---- update step with iterative GN + anomaly detection ----
    MatrixX1 update(const MatrixZ1& z) noexcept {
        // Start from the prior/posterior state
        MatrixX1 x_iter = x_post;
        MatrixZ1 z_pri; // predicted measurement from last iter
        MatrixZZ S; // innovation covariance from last iter
        MatrixXZ K_local; // Kalman gain from last iter

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
            for (int i = 0; i < N_Z; ++i) {
                z_pri[i] = std::isfinite(z_p_jet[i].a) ? z_p_jet[i].a : 0.0;
                H.block(i, 0, 1, N_X) = z_p_jet[i].v.transpose();
            }

            // Compute measurement noise
            R = update_R(z);

            // Innovation covariance S = H P_prior H^T + R
            S = H * P_pri * H.transpose() + R;
            // Add small term for numerical stability
            S += small_noise_ * MatrixZZ::Identity();

            // Compute Kalman gain K = P_prior H^T S^{-1}
            K = P_pri * H.transpose() * S.inverse();
            K_local = K; // save

            // Compute residual using user-supplied residual_func_ (default: z - z_pri)
            MatrixZ1 residual = residual_func_(z_pri, z);

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
        P_post = (MatrixXX::Identity() - K_local * H) * P_pri;
        // Symmetrize
        P_post = 0.5 * (P_post + P_post.transpose());

        // ----------------- Anomaly detection (NIS / NEES) -----------------
        // residual from last iter is last_residual_ and S from last iter is S
        // compute NIS using stable solver
        double nis = std::numeric_limits<double>::quiet_NaN();
        double nees = std::numeric_limits<double>::quiet_NaN();
        try {
            // NIS = residual^T * S^{-1} * residual
            // use LDLT solve for numerical stability
            Eigen::VectorXd tmp = S.ldlt().solve(last_residual_);
            nis = static_cast<double>(last_residual_.transpose() * tmp);

            // NEES = (x_post - x_pri)^T * P_post^{-1} * (x_post - x_pri)
            MatrixX1 dx = x_post - x_pri;
            Eigen::VectorXd tmp2 = P_post.ldlt().solve(dx);
            nees = static_cast<double>(dx.transpose() * tmp2);
        } catch (...) {
            nis = std::numeric_limits<double>::quiet_NaN();
            nees = std::numeric_limits<double>::quiet_NaN();
        }

        last_nis_ = nis;
        last_nees_ = nees;
        total_count_++;

        // evaluate against thresholds if set
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

        // push recent nis failure (1/0) into deque for sliding-window rate
        if (std::isfinite(nis)) {
            recent_nis_failures_.push_back(nis_failed ? 1 : 0);
            if (recent_nis_failures_.size() > window_size_) {
                recent_nis_failures_.pop_front();
            }
        }

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

    // angle_dims_ removed per request

    int iteration_num_ = 1; // GN iterations in update
    double small_noise_ = 1e-6;

    // residual function (default = z_meas - z_pred)
    ResidualFunc residual_func_;

    // ----------------- anomaly detection members -----------------
    size_t window_size_ = 50;
    std::deque<int> recent_nis_failures_; // holds 0/1 flags (most recent at back)
    double recent_fail_rate_threshold_ = 0.4; // if recent fail rate >= this -> flagged

    std::optional<double> nis_threshold_; // if has_value -> compare nis > threshold
    std::optional<double> nees_threshold_; // if has_value -> compare nees > threshold

    int nis_count_ = 0;
    int nees_count_ = 0;
    int total_count_ = 0;

    double last_nis_ = std::numeric_limits<double>::quiet_NaN();
    double last_nees_ = std::numeric_limits<double>::quiet_NaN();
};

} // namespace kalman_hybird_lib

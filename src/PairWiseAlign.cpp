#include "PairWiseAlign.h"
#include <Acceleration.h>
#include <limits>
#define SAME_THRESHOLD 1e-6
#include <type_traits>

namespace PairWiseAlign::Align
{
template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
almost_equal(T x, T y, int ulp)
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::fabs(x - y) <= std::numeric_limits<T>::epsilon() * std::fabs(x + y) * ulp
        // unless the result is subnormal
        || std::fabs(x - y) < std::numeric_limits<T>::min();
}


AffineMatrixN
MYICP::LogMatrix(const AffineMatrixN &T)
{
    Eigen::RealSchur<AffineMatrixN> schur(T);
    AffineMatrixN U = schur.matrixU();
    AffineMatrixN R = schur.matrixT();
    std::vector<bool> selected(N, true);
    MatrixNN mat_B = MatrixNN::Zero(N, N);
    MatrixNN mat_V = MatrixNN::Identity(N, N);

    for (int i = 0; i < N; i++)
    {
        if (selected[i] && fabs(R(i, i) - 1) > SAME_THRESHOLD)
        {
            int pair_second = -1;
            for (int j = i + 1; j < N; j++)
            {
                if (fabs(R(j, j) - R(i, i)) < SAME_THRESHOLD)
                {
                    pair_second = j;
                    selected[j] = false;
                    break;
                }
            }
            if (pair_second > 0)
            {
                selected[i] = false;
                R(i, i) = R(i, i) < -1 ? -1 : R(i, i);
                double theta = acos(R(i, i));
                if (R(i, pair_second) < 0)
                {
                    theta = -theta;
                }
                mat_B(i, pair_second) += theta;
                mat_B(pair_second, i) += -theta;
                mat_V(i, pair_second) += -theta / 2;
                mat_V(pair_second, i) += theta / 2;
                double coeff = 1 - (theta * R(i, pair_second)) / (2 * (1 - R(i, i)));
                mat_V(i, i) += -coeff;
                mat_V(pair_second, pair_second) += -coeff;
            }
        }
    }

    AffineMatrixN LogTrim = AffineMatrixN::Zero();
    LogTrim.block(0, 0, N, N) = mat_B;
    LogTrim.block(0, N, N, 1) = mat_V * R.block(0, N, N, 1);
    AffineMatrixN res = U * LogTrim * U.transpose();
    return res;
}

inline Vector6
MYICP::RotToEuler(const AffineNd &T)
{
    Vector6 res;
    res.head(3) = T.rotation().eulerAngles(0, 1, 2);
    res.tail(3) = T.translation();
    return res;
}

inline AffineMatrixN
MYICP::EulerToRot(const Vector6 &v)
{
    MatrixNN s(Eigen::AngleAxis<Scalar>(v(0), Vector3::UnitX())
                   * Eigen::AngleAxis<Scalar>(v(1), Vector3::UnitY())
                   * Eigen::AngleAxis<Scalar>(v(2), Vector3::UnitZ()));

    AffineMatrixN m = AffineMatrixN::Zero();
    m.block(0, 0, 3, 3) = s;
    m(3, 3) = 1;
    m.col(3).head(3) = v.tail(3);
    return m;
}
inline Vector6
MYICP::LogToVec(const Eigen::Matrix4d &LogT)
{
    Vector6 res;
    res[0] = -LogT(1, 2);
    res[1] = LogT(0, 2);
    res[2] = -LogT(0, 1);
    res[3] = LogT(0, 3);
    res[4] = LogT(1, 3);
    res[5] = LogT(2, 3);
    return res;
}

inline AffineMatrixN
MYICP::VecToLog(const Vector6 &v)
{
    AffineMatrixN m = AffineMatrixN::Zero();
    m << 0, -v[2], v[1], v[3],
        v[2], 0, -v[0], v[4],
        -v[1], v[0], 0, v[5],
        0, 0, 0, 0;
    return m;
}

double
MYICP::FindKnearestMed(const KDtree &kdtree,
                       const MatrixNX &X, int nk)
{
    Eigen::VectorXd X_nearest(X.cols());
#pragma omp parallel for
    for (int i = 0; i < X.cols(); i++)
    {
        int *id = new int[nk];
        double *dist = new double[nk];
        kdtree.query(X.col(i).data(), nk, id, dist);
        Eigen::VectorXd k_dist = Eigen::Map<Eigen::VectorXd>(dist, nk);
        igl::median(k_dist.tail(nk - 1), X_nearest[i]);
        delete[]id;
        delete[]dist;
    }

    //用每个点自身最近邻的七个点距离的中位数,然后在使用所有的点的该值的中位数求根号
    double med;
    igl::median(X_nearest, med);
    return sqrt(med);
}
/// Find self normal edge median of point cloud
double
MYICP::FindKnearestNormMed(const KDtree &kdtree, const Eigen::Matrix3Xd &X, int nk, const Eigen::Matrix3Xd &norm_x)
{
    Eigen::VectorXd X_nearest(X.cols());
#pragma omp parallel for
    for (int i = 0; i < X.cols(); i++)
    {
        int *id = new int[nk];
        double *dist = new double[nk];
        kdtree.query(X.col(i).data(), nk, id, dist);
        Eigen::VectorXd k_dist = Eigen::Map<Eigen::VectorXd>(dist, nk);
        for (int s = 1; s < nk; s++)
        {
            k_dist[s] = std::abs((X.col(id[s]) - X.col(id[0])).dot(norm_x.col(id[0])));
        }
        igl::median(k_dist.tail(nk - 1), X_nearest[i]);  //其自身周边六个点中,距离其平面的距离,可以度量该点的平滑程度,以及该点所处平面法向量的可信度
        delete[]id;
        delete[]dist;
    }
    double med;
    igl::median(X_nearest, med);
    return med;
}

template<typename Derived1, typename Derived2, typename Derived3>
AffineNd
MYICP::point_to_point(Eigen::MatrixBase<Derived1> &X,
                      Eigen::MatrixBase<Derived2> &Y,
                      const Eigen::MatrixBase<Derived3> &w)
{
    int dim = X.rows();
    /// Normalize weight vector
    Eigen::VectorXd w_normalized = w / w.sum();
    /// De-mean
    Eigen::VectorXd X_mean(dim), Y_mean(dim);
    for (int i = 0; i < dim; ++i)
    {
        X_mean(i) = (X.row(i).array() * w_normalized.transpose().array()).sum();
        Y_mean(i) = (Y.row(i).array() * w_normalized.transpose().array()).sum();
    }
    X.colwise() -= X_mean;
    Y.colwise() -= Y_mean;

    /// Compute transformation
    AffineNd transformation;
    MatrixXX sigma = X * w_normalized.asDiagonal() * Y.transpose();
    Eigen::JacobiSVD<MatrixXX> svd(sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0.0)
    {
        VectorN S = VectorN::Ones(dim);
        S(dim - 1) = -1.0;
        transformation.linear() = svd.matrixV() * S.asDiagonal() * svd.matrixU().transpose();
    }
    else
    {
        transformation.linear() = svd.matrixV() * svd.matrixU().transpose();
    }
    transformation.translation() = Y_mean - transformation.linear() * X_mean;
    /// Re-apply mean
    X.colwise() += X_mean;
    Y.colwise() += Y_mean;
    /// Return transformation
    return transformation;
}

template<typename Derived1, typename Derived2, typename Derived3, typename Derived4, typename Derived5>
Eigen::Affine3d
MYICP::point_to_plane(Eigen::MatrixBase<Derived1> &X,
                      Eigen::MatrixBase<Derived2> &Y,
                      const Eigen::MatrixBase<Derived3> &Norm,
                      const Eigen::MatrixBase<Derived4> &w,
                      const Eigen::MatrixBase<Derived5> &u)
{
    typedef Eigen::Matrix<double, 6, 6> Matrix66;
    typedef Eigen::Matrix<double, 6, 1> Vector6;
    typedef Eigen::Block<Matrix66, 3, 3> Block33;
    /// Normalize weight vector
    Eigen::VectorXd w_normalized = w / w.sum();
    /// De-mean
    Eigen::Vector3d X_mean;
    for (int i = 0; i < 3; ++i)
        X_mean(i) = (X.row(i).array() * w_normalized.transpose().array()).sum();
    X.colwise() -= X_mean;
    Y.colwise() -= X_mean;
    /// Prepare LHS and RHS
    Matrix66 LHS = Matrix66::Zero();
    Vector6 RHS = Vector6::Zero();
    Block33 TL = LHS.topLeftCorner<3, 3>();
    Block33 TR = LHS.topRightCorner<3, 3>();
    Block33 BR = LHS.bottomRightCorner<3, 3>();
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(3, X.cols());

#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < X.cols(); i++)
        {
            C.col(i) = X.col(i).cross(Norm.col(i));
        }
#pragma omp sections nowait
        {
#pragma omp section
            for (int i = 0; i < X.cols(); i++)
                TL.selfadjointView<Eigen::Upper>().rankUpdate(C.col(i), w(i));
#pragma omp section
            for (int i = 0; i < X.cols(); i++)
                TR += (C.col(i) * Norm.col(i).transpose()) * w(i);
#pragma omp section
            for (int i = 0; i < X.cols(); i++)
                BR.selfadjointView<Eigen::Upper>().rankUpdate(Norm.col(i), w(i));
#pragma omp section
            for (int i = 0; i < C.cols(); i++)
            {
                double dist_to_plane = -((X.col(i) - Y.col(i)).dot(Norm.col(i)) - u(i)) * w(i);
                RHS.head<3>() += C.col(i) * dist_to_plane;
                RHS.tail<3>() += Norm.col(i) * dist_to_plane;
            }
        }
    }
    LHS = LHS.selfadjointView<Eigen::Upper>();
    /// Compute transformation
    Eigen::Affine3d transformation;
    Eigen::LDLT<Matrix66> ldlt(LHS);
    RHS = ldlt.solve(RHS);
    transformation = Eigen::AngleAxisd(RHS(0), Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(RHS(1), Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(RHS(2), Eigen::Vector3d::UnitZ());
    transformation.translation() = RHS.tail<3>();

    /// Apply transformation
    /// Re-apply mean
    X.colwise() += X_mean;
    Y.colwise() += X_mean;
    transformation.translation() += X_mean - transformation.linear() * X_mean;
    /// Return transformation
    return transformation;
}

template<typename Derived1, typename Derived2, typename Derived3, typename Derived4>
double
MYICP::point_to_plane_gaussnewton(const Eigen::MatrixBase<Derived1> &X,
                                  const Eigen::MatrixBase<Derived2> &Y,
                                  const Eigen::MatrixBase<Derived3> &norm_y,
                                  const Eigen::MatrixBase<Derived4> &w,
                                  Matrix44 Tk, Vector6 &dir)
{
    typedef Eigen::Matrix<double, 6, 6> Matrix66;
    typedef Eigen::Matrix<double, 12, 6> Matrix126;
    typedef Eigen::Matrix<double, 9, 3> Matrix93;
    typedef Eigen::Block<Matrix126, 9, 3> Block93;
    typedef Eigen::Block<Matrix126, 3, 3> Block33;
    typedef Eigen::Matrix<double, 12, 1> Vector12;
    typedef Eigen::Matrix<double, 9, 1> Vector9;
    typedef Eigen::Matrix<double, 4, 2> Matrix42;
    /// Normalize weight vector
    Eigen::VectorXd w_normalized = w / w.sum();
    /// Prepare LHS and RHS
    Matrix66 LHS = Matrix66::Zero();
    Vector6 RHS = Vector6::Zero();

    Vector6 log_T = LogToVec(LogMatrix(Tk));
    Matrix33 B = VecToLog(log_T).block(0, 0, 3, 3);
    double a = log_T[0];
    double b = log_T[1];
    double c = log_T[2];
    Matrix33 R = Tk.block(0, 0, 3, 3);
    Vector3 t = Tk.block(0, 3, 3, 1);
    Vector3 u = log_T.tail(3);

    Matrix93 dbdw = Matrix93::Zero();
    dbdw(1, 2) = dbdw(5, 0) = dbdw(6, 1) = -1;
    dbdw(2, 1) = dbdw(3, 2) = dbdw(7, 0) = 1;
    Matrix93 db2dw = Matrix93::Zero();
    db2dw(3, 1) = db2dw(4, 0) = db2dw(6, 2) = db2dw(8, 0) = a;
    db2dw(0, 1) = db2dw(1, 0) = db2dw(7, 2) = db2dw(8, 1) = b;
    db2dw(0, 2) = db2dw(2, 0) = db2dw(4, 2) = db2dw(5, 1) = c;
    db2dw(1, 1) = db2dw(2, 2) = -2 * a;
    db2dw(3, 0) = db2dw(5, 2) = -2 * b;
    db2dw(6, 0) = db2dw(7, 1) = -2 * c;
    double theta = std::sqrt(a * a + b * b + c * c);
    double st = sin(theta), ct = cos(theta);

    Matrix42 coeff = Matrix42::Zero();
    if (theta > SAME_THRESHOLD)
    {
        coeff << st / theta, (1 - ct) / (theta * theta),
            (theta * ct - st) / (theta * theta * theta), (theta * st - 2 * (1 - ct)) / pow(theta, 4),
            (1 - ct) / (theta * theta), (theta - st) / pow(theta, 3),
            (theta * st - 2 * (1 - ct)) / pow(theta, 4), (theta * (1 - ct) - 3 * (theta - st)) / pow(theta, 5);
    }
    else
        coeff(0, 0) = 1;

    Matrix93 tempB3;
    tempB3.block<3, 3>(0, 0) = a * B;
    tempB3.block<3, 3>(3, 0) = b * B;
    tempB3.block<3, 3>(6, 0) = c * B;
    Matrix33 B2 = B * B;
    Matrix93 temp2B3;
    temp2B3.block<3, 3>(0, 0) = a * B2;
    temp2B3.block<3, 3>(3, 0) = b * B2;
    temp2B3.block<3, 3>(6, 0) = c * B2;
    Matrix93 dRdw = coeff(0, 0) * dbdw + coeff(1, 0) * tempB3
        + coeff(2, 0) * db2dw + coeff(3, 0) * temp2B3;
    Vector9 dtdw = coeff(0, 1) * dbdw * u + coeff(1, 1) * tempB3 * u
        + coeff(2, 1) * db2dw * u + coeff(3, 1) * temp2B3 * u;
    Matrix33 dtdu = Matrix33::Identity() + coeff(2, 0) * B + coeff(2, 1) * B2;

    Eigen::VectorXd rk(X.cols());
    Eigen::MatrixXd Jk(X.cols(), 6);
#pragma omp for
    for (int i = 0; i < X.cols(); i++)
    {
        Vector3 xi = X.col(i);
        Vector3 yi = Y.col(i);
        Vector3 ni = norm_y.col(i);
        double wi = sqrt(w_normalized[i]);

        Matrix33 dedR = wi * ni * xi.transpose();
        Vector3 dedt = wi * ni;

        Vector6 dedx;
        dedx(0) = (dedR.cwiseProduct(dRdw.block(0, 0, 3, 3))).sum()
            + dedt.dot(dtdw.head<3>());
        dedx(1) = (dedR.cwiseProduct(dRdw.block(3, 0, 3, 3))).sum()
            + dedt.dot(dtdw.segment<3>(3));
        dedx(2) = (dedR.cwiseProduct(dRdw.block(6, 0, 3, 3))).sum()
            + dedt.dot(dtdw.tail<3>());
        dedx(3) = dedt.dot(dtdu.col(0));
        dedx(4) = dedt.dot(dtdu.col(1));
        dedx(5) = dedt.dot(dtdu.col(2));

        Jk.row(i) = dedx.transpose();
        rk[i] = wi * ni.dot(R * xi - yi + t);
    }
    LHS = Jk.transpose() * Jk;
    RHS = -Jk.transpose() * rk;
    Eigen::CompleteOrthogonalDecomposition<Matrix66> cod_(LHS);
    dir = cod_.solve(RHS);
    double gTd = -RHS.dot(dir);
    return gTd;
}

void
MYICP::point_to_point(MatrixNX &X, MatrixNX &Y, VectorN &source_mean,
                      VectorN &target_mean, ICP::Parameters &par)
{
    /// Build kd-tree
    KDtree kdtree(Y);
    /// Buffers
    MatrixNX Q = MatrixNX::Zero(N, X.cols());
    VectorX W = VectorX::Zero(X.cols());
    AffineNd T;
    if (par.use_init)
        T.matrix() = par.init_trans;
    else
        T = AffineNd::Identity();
    MatrixXX To1 = T.matrix();
    MatrixXX To2 = T.matrix();
    int nPoints = X.cols();

    //Anderson Acc para
    Acceleration accelerator_;
    AffineNd SVD_T = T;
    double energy = .0, last_energy = std::numeric_limits<double>::max();

    //ground truth point clouds
    MatrixNX X_gt = X;
    if (par.has_groundtruth)
    {
        VectorN temp_trans = par.gt_trans.col(N).head(N);
        X_gt.colwise() += source_mean;
        X_gt = par.gt_trans.block(0, 0, N, N) * X_gt;
        X_gt.colwise() += temp_trans - target_mean;
    }

    //output para
    std::string file_out = par.out_path;
    std::vector<double> times, energys, gt_mses;
    double begin_time, end_time, run_time;
    double gt_mse = 0.0;

    // dynamic welsch paras
    double nu1 = 1, nu2 = 1;
    double begin_init = omp_get_wtime();

    //Find initial closest point
#pragma omp parallel for
    for (int i = 0; i < nPoints; ++i)
    {
        VectorN cur_p = T * X.col(i);
        Q.col(i) = Y.col(kdtree.closest(cur_p.data()));  //在Y中去寻找最接近 T * X.col(i) 的一个点
        W[i] = (cur_p - Q.col(i)).norm(); //W是一个残差向量
    }
    if (par.f == ICP::WELSCH)
    {
        //动态 welsch，用自身计算 k 最近点;
        nu2 = par.nu_end_k * FindKnearestMed(kdtree, Y, 7);   // ( 1.0/6.0 ) *
        double med1;
        igl::median(W, med1);  //找中位数
        nu1 = par.nu_begin_k * med1;
        nu1 = nu1 > nu2 ? nu1 : nu2;
    }
    double end_init = omp_get_wtime();
    double init_time = end_init - begin_init;

    //AA init
    accelerator_.init(par.anderson_m, (N + 1) * (N + 1), LogMatrix(T.matrix()).data());

    begin_time = omp_get_wtime();
    bool stop1 = false;
    while (!stop1)
    {
        /// run ICP
        int icp = 0;
        for (; icp < par.max_icp; ++icp)
        {
            bool accept_aa = false;
            energy = get_energy(par.f, W, nu1);
            if (par.use_AA)
            {
                if (energy < last_energy)
                {
                    last_energy = energy;
                    accept_aa = true;
                }
                else
                {
                    accelerator_.replace(LogMatrix(SVD_T.matrix()).data());
                    //Re-find the closest point
#pragma omp parallel for
                    for (int i = 0; i < nPoints; ++i)
                    {
                        VectorN cur_p = SVD_T * X.col(i);
                        Q.col(i) = Y.col(kdtree.closest(cur_p.data()));
                        W[i] = (cur_p - Q.col(i)).norm();
                    }
                    last_energy = get_energy(par.f, W, nu1);
                }
            }
            else
                last_energy = energy;

            end_time = omp_get_wtime();
            run_time = end_time - begin_time;
            if (par.has_groundtruth)
            {
                gt_mse = (T * X - X_gt).squaredNorm() / nPoints;
            }

            // save results
            energys.push_back(last_energy);
            times.push_back(run_time);
            gt_mses.push_back(gt_mse);

            if (par.print_energy)
                std::cout << "icp iter = " << icp << ", Energy = " << last_energy
                          << ", time = " << run_time << std::endl;

            robust_weight(par.f, W, nu1);
            // Rotation and translation update
            T = RigidMotionEstimator::point_to_point(X, Q, W);

            //Anderson Acc
            SVD_T = T;
            if (par.use_AA)
            {
                AffineMatrixN Trans =
                    (Eigen::Map<const AffineMatrixN>(accelerator_.compute(LogMatrix(T.matrix()).data()).data(),
                                                     N + 1,
                                                     N + 1)).exp();
                T.linear() = Trans.block(0, 0, N, N);
                T.translation() = Trans.block(0, N, N, 1);
            }

            // Find closest point
#pragma omp parallel for
            for (int i = 0; i < nPoints; ++i)
            {
                VectorN cur_p = T * X.col(i);
                Q.col(i) = Y.col(kdtree.closest(cur_p.data()));
                W[i] = (cur_p - Q.col(i)).norm();
            }
            /// Stopping criteria
            double stop2 = (T.matrix() - To2).norm();
            To2 = T.matrix();
            if (stop2 < par.stop)
            {
                break;
            }
        }
        if (par.f != ICP::WELSCH)
            stop1 = true;
        else
        {
            stop1 = fabs(nu1 - nu2) < SAME_THRESHOLD ? true : false;
            nu1 = nu1 * par.nu_alpha > nu2 ? nu1 * par.nu_alpha : nu2;
            if (par.use_AA)
            {
                accelerator_.reset(LogMatrix(T.matrix()).data());
                last_energy = std::numeric_limits<double>::max();
            }
        }
    }

    ///calc convergence energy
    last_energy = get_energy(par.f, W, nu1);
    X = T * X;
    gt_mse = (X - X_gt).squaredNorm() / nPoints;
    T.translation() += -T.rotation() * source_mean + target_mean;
    X.colwise() += target_mean;

    ///save convergence result
    par.convergence_energy = last_energy;
    par.convergence_gt_mse = gt_mse;
    par.res_trans = T.matrix();

    ///output
    if (par.print_output)
    {
        std::ofstream out_res(par.out_path);
        if (!out_res.is_open())
        {
            std::cout << "Can't open out file " << par.out_path << std::endl;
        }

        //output time and energy
        out_res.precision(16);
        for (int i = 0; i < times.size(); i++)
        {
            out_res << times[i] << " " << energys[i] << " " << gt_mses[i] << std::endl;
        }
        out_res.close();
        std::cout << " write res to " << par.out_path << std::endl;
    }
}

/// Reweighted ICP with point to plane
/// @param Source (one 3D point per column)
/// @param Target (one 3D point per column)
/// @param Target normals (one 3D normal per column)
/// @param Parameters
//    template <typename Derived1, typename Derived2, typename Derived3>
void
MYICP::point_to_plane(Eigen::Matrix3Xd &X,
                      Eigen::Matrix3Xd &Y, Eigen::Matrix3Xd &norm_x, Eigen::Matrix3Xd &norm_y,
                      Eigen::Vector3d &source_mean, Eigen::Vector3d &target_mean,
                      ICP::Parameters &par)
{
    /// Build kd-tree
    KDtree kdtree(Y);
    /// Buffers
    Eigen::Matrix3Xd Qp = Eigen::Matrix3Xd::Zero(3, X.cols());
    Eigen::Matrix3Xd Qn = Eigen::Matrix3Xd::Zero(3, X.cols());
    Eigen::VectorXd W = Eigen::VectorXd::Zero(X.cols());
    Eigen::Matrix3Xd ori_X = X;
    AffineNd T;
    if (par.use_init)
        T.matrix() = par.init_trans;
    else
        T = AffineNd::Identity();
    AffineMatrixN To1 = T.matrix();
    X = T * X;

    Eigen::Matrix3Xd X_gt = X;
    if (par.has_groundtruth)
    {
        Eigen::Vector3d temp_trans = par.gt_trans.block(0, 3, 3, 1);
        X_gt = ori_X;
        X_gt.colwise() += source_mean;
        X_gt = par.gt_trans.block(0, 0, 3, 3) * X_gt;
        X_gt.colwise() += temp_trans - target_mean;
    }

    std::vector<double> times, energys, gt_mses;
    double begin_time, end_time, run_time;
    double gt_mse = 0.0;

    ///dynamic welsch, calc k-nearest points with itself;
    double begin_init = omp_get_wtime();

    //Anderson Acc para
    Acceleration accelerator_;
    AffineNd LG_T = T;
    double energy = 0.0, prev_res = std::numeric_limits<double>::max(), res = 0.0;


    // Find closest point
#pragma omp parallel for
    for (int i = 0; i < X.cols(); ++i)
    {
        int id = kdtree.closest(X.col(i).data());
        Qp.col(i) = Y.col(id);
        Qn.col(i) = norm_y.col(id);
        W[i] = std::abs(Qn.col(i).transpose() * (X.col(i) - Qp.col(i)));
    }
    double end_init = omp_get_wtime();
    double init_time = end_init - begin_init;

    begin_time = omp_get_wtime();
    int total_iter = 0;
    double test_total_time = 0.0;
    bool stop1 = false;
    while (!stop1)
    {
        /// ICP
        for (int icp = 0; icp < par.max_icp; ++icp)
        {
            total_iter++;

            bool accept_aa = false;
            energy = get_energy(par.f, W, par.p);
            end_time = omp_get_wtime();
            run_time = end_time - begin_time;
            energys.push_back(energy);
            times.push_back(run_time);
            Eigen::VectorXd test_w = (X - Qp).colwise().norm();
            if (par.has_groundtruth)
            {
                gt_mse = (X - X_gt).squaredNorm() / X.cols();
            }
            gt_mses.push_back(gt_mse);

            /// Compute weights
            robust_weight(par.f, W, par.p);
            /// Rotation and translation update
            T = point_to_plane(X, Qp, Qn, W, Eigen::VectorXd::Zero(X.cols())) * T;
            /// Find closest point
#pragma omp parallel for
            for (int i = 0; i < X.cols(); i++)
            {
                X.col(i) = T * ori_X.col(i);
                int id = kdtree.closest(X.col(i).data());
                Qp.col(i) = Y.col(id);
                Qn.col(i) = norm_y.col(id);
                W[i] = std::abs(Qn.col(i).transpose() * (X.col(i) - Qp.col(i)));
            }

            if (par.print_energy)
                std::cout << "icp iter = " << total_iter << ", gt_mse = " << gt_mse
                          << ", energy = " << energy << std::endl;

            /// Stopping criteria
            double stop2 = (T.matrix() - To1).norm();
            To1 = T.matrix();
            if (stop2 < par.stop)
                break;
        }
        stop1 = true;
    }

    par.res_trans = T.matrix();

    ///calc convergence energy
    W = (Qn.array() * (X - Qp).array()).colwise().sum().abs().transpose();
    energy = get_energy(par.f, W, par.p);
    gt_mse = (X - X_gt).squaredNorm() / X.cols();
    T.translation().noalias() += -T.rotation() * source_mean + target_mean;
    X.colwise() += target_mean;
    norm_x = T.rotation() * norm_x;

    ///save convergence result
    par.convergence_energy = energy;
    par.convergence_gt_mse = gt_mse;
    par.res_trans = T.matrix();

    ///output
    if (par.print_output)
    {
        std::ofstream out_res(par.out_path);
        if (!out_res.is_open())
        {
            std::cout << "Can't open out file " << par.out_path << std::endl;
        }

        ///output time and energy
        out_res.precision(16);
        for (int i = 0; i < total_iter; i++)
        {
            out_res << times[i] << " " << energys[i] << " " << gt_mses[i] << std::endl;
        }
        out_res.close();
        std::cout << " write res to " << par.out_path << std::endl;
    }
}

/// Reweighted ICP with point to plane
/// @param Source (one 3D point per column)
/// @param Target (one 3D point per column)
/// @param Target normals (one 3D normal per column)
/// @param Parameters
//    template <typename Derived1, typename Derived2, typename Derived3>
void
MYICP::point_to_plane_GN(Eigen::Matrix3Xd &X,
                         Eigen::Matrix3Xd &Y, Eigen::Matrix3Xd &norm_x, Eigen::Matrix3Xd &norm_y,
                         Eigen::Vector3d &source_mean, Eigen::Vector3d &target_mean,
                         ICP::Parameters &par)
{
    /// Build kd-tree
    KDtree kdtree(Y);

    /// Buffers
    Eigen::Matrix3Xd Qp = Eigen::Matrix3Xd::Zero(3, X.cols());
    Eigen::Matrix3Xd Qn = Eigen::Matrix3Xd::Zero(3, X.cols());
    Eigen::VectorXd W = Eigen::VectorXd::Zero(X.cols());
    Eigen::Matrix3Xd ori_X = X;


    AffineNd T;
    if (par.use_init)
        T.matrix() = par.init_trans;
    else
        T = AffineNd::Identity();
    AffineMatrixN To1 = T.matrix();
    X = T * X;

    Eigen::Matrix3Xd X_gt = X;  //3 * X的矩阵

    std::vector<double> times, energys, gt_mses;
    double begin_time, end_time, run_time;
    double gt_mse;

    ///dynamic welsch, calc k-nearest points with itself;
    double nu1 = 1, nu2 = 1;
    double begin_init = omp_get_wtime();

    //Anderson Acc para
    Acceleration accelerator_;
    Vector6 LG_T;
    Vector6 Dir;
    //add time test
    double energy = 0.0, prev_energy = std::numeric_limits<double>::max();
    if (par.use_AA)
    {
        Eigen::Matrix4d log_T = LogMatrix(T.matrix());
        LG_T = LogToVec(log_T);
        accelerator_.init(par.anderson_m, 6, LG_T.data());
    }

    // Find closest point
#pragma omp parallel for
    for (int i = 0; i < X.cols(); ++i)
    {
        int id = kdtree.closest(X.col(i).data());
        Qp.col(i) = Y.col(id);
        Qn.col(i) = norm_y.col(id);
        W[i] = std::abs(Qn.col(i).transpose() * (X.col(i) - Qp.col(i)));
    }

    if (par.f == ICP::WELSCH)
    {
        double med1;
        igl::median(W, med1);
        nu1 = par.nu_begin_k * med1;   //当前残差中位数
        nu2 = par.nu_end_k * FindKnearestNormMed(kdtree, Y, 7, norm_y); //自己与自己匹配的残差的中位数,可以作为迭代停止的一种标志
        nu1 = nu1 > nu2 ? nu1 : nu2;
    }
    double end_init = omp_get_wtime();
    double init_time = end_init - begin_init;

    begin_time = omp_get_wtime();
    int total_iter = 0;
    double test_total_time = 0.0;
    bool stop1 = false;
    par.max_icp = 6;
    while (!stop1)
    {
        par.max_icp = std::min(par.max_icp + 1, 10);  //迭代次数不要超过10
        /// ICP
        for (int icp = 0; icp < par.max_icp; ++icp)
        {
            total_iter++;
            int n_linsearch = 0;
            energy = get_energy(par.f, W, nu1);  //残差转能量函数
            if (par.use_AA)
            {
                if (energy < prev_energy)
                {
                    prev_energy = energy;
                }
                else
                {
                    // line search
                    double alpha = 0.0;
                    Vector6 new_t = LG_T;
                    Eigen::VectorXd lowest_W = W;
                    Eigen::Matrix3Xd lowest_Qp = Qp;
                    Eigen::Matrix3Xd lowest_Qn = Qn;
                    Eigen::Affine3d lowest_T = T;
                    n_linsearch++;
                    alpha = 1;
                    new_t = LG_T + alpha * Dir;
                    T.matrix() = VecToLog(new_t).exp();
                    /// Find closest point
                    //如果当前迭代残差增加,说明上一次的迭代失效,则使用安德森加速进行线性搜索
                    // 更新新的位姿后再一次更新匹配关系
#pragma omp parallel for
                    for (int i = 0; i < X.cols(); i++)
                    {
                        X.col(i) = T * ori_X.col(i);
                        int id = kdtree.closest(X.col(i).data());
                        Qp.col(i) = Y.col(id);
                        Qn.col(i) = norm_y.col(id);
                        W[i] = std::abs(Qn.col(i).transpose() * (X.col(i) - Qp.col(i)));
                    }
                    //看当前测试是否满足残差下降
                    double test_energy = get_energy(par.f, W, nu1);
                    if (test_energy < energy)
                    {
                        // 满足下降,则更新安德森参数
                        accelerator_.reset(new_t.data());
                        energy = test_energy;
                    }
                    else
                    {
                        Qp = lowest_Qp;
                        Qn = lowest_Qn;
                        W = lowest_W;
                        T = lowest_T;
                    }
                    prev_energy = energy;
                }
            }
            else
            {
                prev_energy = energy;
            }

            end_time = omp_get_wtime();
            run_time = end_time - begin_time;
            energys.push_back(prev_energy);
            times.push_back(run_time);
            if (par.has_groundtruth)
            {
                gt_mse = (X - X_gt).squaredNorm() / X.cols();
            }
            gt_mses.push_back(gt_mse);

            /// Compute weights nu1的调控是基于初始位姿的
            robust_weight(par.f, W, nu1);
            /// Rotation and translation update
            point_to_plane_gaussnewton(ori_X, Qp, Qn, W, T.matrix(), Dir);
            LG_T = LogToVec(LogMatrix(T.matrix()));
            LG_T += Dir;
            T.matrix() = VecToLog(LG_T).exp();

            // Anderson acc
            if (par.use_AA)
            {
                Vector6 AA_t;
                AA_t = accelerator_.compute(LG_T.data());
                T.matrix() = VecToLog(AA_t).exp();
            }
            if (par.print_energy)
                std::cout << "icp iter = " << total_iter << ", gt_mse = " << gt_mse
                          << ", nu1 = " << nu1 << ", acept_aa= " << n_linsearch
                          << ", energy = " << prev_energy << std::endl;

            /// Find closest point
#pragma omp parallel for
            for (int i = 0; i < X.cols(); i++)
            {
                X.col(i) = T * ori_X.col(i);
                int id = kdtree.closest(X.col(i).data());
                Qp.col(i) = Y.col(id);
                Qn.col(i) = norm_y.col(id);
                W[i] = std::abs(Qn.col(i).transpose() * (X.col(i) - Qp.col(i)));
            }

            /// Stopping criteria
            double stop2 = (T.matrix() - To1).norm();
            To1 = T.matrix();
            if (stop2 < par.stop)
                break;
        }

        if (par.f == ICP::WELSCH)
        {
            stop1 = fabs(nu1 - nu2) < SAME_THRESHOLD ? true : false;
            nu1 = nu1 * par.nu_alpha > nu2 ? nu1 * par.nu_alpha : nu2;
            if (par.use_AA)
            {
                accelerator_.reset(LogToVec(LogMatrix(T.matrix())).data());
                prev_energy = std::numeric_limits<double>::max();
            }
        }
        else
            stop1 = true;
    }

    par.res_trans = T.matrix();

    ///calc convergence energy
    W = (Qn.array() * (X - Qp).array()).colwise().sum().abs().transpose();
    energy = get_energy(par.f, W, nu1);
    gt_mse = (X - X_gt).squaredNorm() / X.cols();
    T.translation().noalias() += -T.rotation() * source_mean + target_mean;
    X.colwise() += target_mean;
    norm_x = T.rotation() * norm_x;

    ///save convergence result
    par.convergence_energy = energy;
    par.convergence_gt_mse = gt_mse;
    par.res_trans = T.matrix();

    ///output
    if (par.print_output)
    {
        std::ofstream out_res(par.out_path);
        if (!out_res.is_open())
        {
            std::cout << "Can't open out file " << par.out_path << std::endl;
        }

        ///output time and energy
        out_res.precision(16);
        for (int i = 0; i < total_iter; i++)
        {
            out_res << times[i] << " " << energys[i] << " " << gt_mses[i] << std::endl;
        }
        out_res.close();
        std::cout << " write res to " << par.out_path << std::endl;
    }
}

MYICP::MYICP(){}
MYICP::~MYICP(){}
}

namespace PairWiseAlign::AAICP{
///aaicp
///////////////////////////////////////////////////////////////////////////////////////////
Vector6 Matrix42Vector6 (const Matrix4 m)
{
    Vector6 v;
    Matrix3 s = m.block(0,0,3,3);
    v.head(3) = s.eulerAngles(0, 1, 2);
    v.tail(3) = m.col(3).head(3);
    return v;
}

///////////////////////////////////////////////////////////////////////////////////////////
Matrix4 Vector62Matrix4 (const Vector6 v)
{
    Matrix3 s (Eigen::AngleAxis<Scalar>(v(0), Vector3::UnitX())
                   * Eigen::AngleAxis<Scalar>(v(1), Vector3::UnitY())
                   * Eigen::AngleAxis<Scalar>(v(2), Vector3::UnitZ()));
    Matrix4 m = Matrix4::Zero();
    m.block(0,0,3,3) = s;
    m(3,3) = 1;
    m.col(3).head(3) = v.tail(3);
    return m;
}

///////////////////////////////////////////////////////////////////////////////////////////
int alphas_cond (VectorX alphas)
{
    double alpha_limit_min_ = -10;
    double alpha_limit_max_ = 10;
    return alpha_limit_min_ < alphas.minCoeff() && alphas.maxCoeff() < alpha_limit_max_ && alphas(alphas.size()-1) > 0;
}

///////////////////////////////////////////////////////////////////////////////////////////
VectorX get_alphas_lstsq (const Matrix6X f)
{
    Matrix6X A = f.leftCols(f.cols()-1);
    A *= -1;
    A += f.rightCols(1) * VectorX::Constant(f.cols()-1, 1).transpose();
    VectorX sol = A.colPivHouseholderQr().solve(f.rightCols(1));
    sol.conservativeResize(sol.size()+1);
    sol[sol.size()-1] = 0;
    sol[sol.size()-1] = 1-sol.sum();
    return sol;
}

///////////////////////////////////////////////////////////////////////////////////////////
VectorX get_next_u (const Matrix6X u, const Matrix6X g, const Matrix6X f, std::vector<double> & save_alphas)
{
    int i = 1;
    double beta_ = 1.0;
    save_alphas.clear();
    Vector6 sol = ((1-beta_)*u.col(u.cols()-1) + beta_*g.col(g.cols()-1));
    VectorX sol_alphas(1);
    sol_alphas << 1;

    i = 2;
    for (; i <= f.cols(); i++)
    {
        VectorX alphas = get_alphas_lstsq(f.rightCols(i));
        if (!alphas_cond(alphas))
        {
            break;
        }
        sol = (1-beta_)*u.rightCols(i)*alphas + beta_*g.rightCols(i)*alphas;
        sol_alphas = alphas;
    }
    for(int i= 0; i<sol_alphas.rows(); i++)
    {
        save_alphas.push_back(sol_alphas[i]);
    }
    return sol;
}


void point_to_point_aaicp(Eigen::Matrix<Scalar, 3, Eigen::Dynamic> & X,
                          Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& Y,
                          Eigen::Matrix<Scalar, 3, 1>& source_mean,
                          Eigen::Matrix<Scalar, 3, 1>& target_mean,
                          ICP::Parameters& par) {
    /// Build kd-tree
    _nanoflann::KDTreeAdaptor<Eigen::Matrix<Scalar, 3, Eigen::Dynamic>, 3, _nanoflann::metric_L2_Simple> kdtree(Y);
    /// Buffers
    Eigen::Matrix3Xd Q = Eigen::Matrix3Xd::Zero(3, X.cols());
    Eigen::VectorXd W = Eigen::VectorXd::Zero(X.cols());
    Eigen::Matrix3Xd ori_X = X;

    double prev_energy = std::numeric_limits<double>::max(), energy = std::numeric_limits<double>::max();
    Eigen::Affine3d T;
    if(par.use_init)
    {
        T.linear() = par.init_trans.block(0,0,3,3);
        T.translation() = par.init_trans.block(0,3,3,1);
    }
    else
        T = Eigen::Affine3d::Identity();
    Eigen::Matrix3Xd X_gt = X;
    ///stop criterion paras
    MatrixXX To1 = T.matrix();
    MatrixXX To2 = T.matrix();

    ///AA paras
    Matrix6X u(6,0), g(6,0), f(6,0);
    Vector6 u_next, u_k;
    Matrix4 transformation = Matrix4::Identity();
    Matrix4 final_transformation = Matrix4::Identity();

    ///output para
    std::vector<double> times, energys, gt_mses;
    double gt_mse;
    double begin_time, end_time, run_time;
    begin_time = omp_get_wtime();

    ///output coeffs
    std::vector<std::vector<double>> coeffs;
    coeffs.clear();
    std::vector<double> alphas;

    X = T * X;
    ///groud truth target point cloud
    if(par.has_groundtruth)
    {
        Eigen::Vector3d temp_trans = par.gt_trans.block(0, 3, 3, 1);
        X_gt = ori_X;
        X_gt.colwise() += source_mean;
        X_gt = par.gt_trans.block(0, 0, 3, 3) * X_gt;
        X_gt.colwise() += temp_trans - target_mean;
    }

    ///begin ICP
    int icp = 0;
    for (; icp<par.max_icp; ++icp)
    {
        bool accept_aa = false;
        int nPoints = X.cols();
        end_time = omp_get_wtime();
        run_time = end_time - begin_time;
        /// Find closest point
#pragma omp parallel for
        for (int i = 0; i<nPoints; ++i) {
            Q.col(i) = Y.col(kdtree.closest(X.col(i).data()));
        }

        ///calc time

        times.push_back(run_time);
        if(par.has_groundtruth)
        {
            gt_mse = pow((X - X_gt).norm(),2) / nPoints;
        }
        gt_mses.push_back(gt_mse);

        /// Computer rotation and translation
        /// Compute weights
        W = (X - Q).colwise().norm();
        robust_weight(par.f, W, par.p);

        /// Rotation and translation update
        T = RigidMotionEstimator::point_to_point(X, Q, W) * T;
        final_transformation = T.matrix();

        ///Anderson acceleration
        if(icp)
        {
            Vector6 g_k = Matrix42Vector6(transformation * final_transformation);
            // Calculate energy
            W = (X - Q).colwise().norm();
            energy = get_energy(par.f, W, par.p);

            ///The first heuristic
            if ((energy - prev_energy)/prev_energy > par.error_overflow_threshold_) {
                u_next = u_k = g.rightCols(1);
                prev_energy = std::numeric_limits<double>::max();
                u = u.rightCols(2);
                g = g.rightCols(1);
                f = f.rightCols(1);
            }
            else
            {
                prev_energy = energy;

                g.conservativeResize(g.rows(),g.cols()+1);
                g.col(g.cols()-1) = g_k;

                Vector6 f_k = g_k - u_k;
                f.conservativeResize(f.rows(),f.cols()+1);
                f.col(f.cols()-1) = f_k;

                u_next = get_next_u(u, g, f, alphas);
                u.conservativeResize(u.rows(),u.cols()+1);
                u.col(u.cols()-1) = u_next;

                u_k = u_next;
                accept_aa = true;
            }
        }
            ///init
        else
        {
            // Calculate energy
            W = (X - Q).colwise().norm();
            prev_energy = get_energy(par.f, W, par.p);
            Vector6 u0 = Matrix42Vector6(Matrix4::Identity());
            u.conservativeResize(u.rows(),u.cols()+1);
            u.col(0)=u0;

            Vector6 u1 = Matrix42Vector6(transformation * final_transformation);
            g.conservativeResize(g.rows(),g.cols()+1);
            g.col(0)=u1;

            u.conservativeResize(u.rows(),u.cols()+1);

            u.col(1)=u1;

            f.conservativeResize(f.rows(),f.cols()+1);
            f.col(0)=u1 - u0;

            u_next = u1;
            u_k = u1;

            energy = prev_energy;
        }

        transformation = Vector62Matrix4(u_next)*(final_transformation.inverse());
        final_transformation = Vector62Matrix4(u_next);
        X = final_transformation.block(0,0,3,3) * ori_X;
        Vector3 trans = final_transformation.block(0,3,3,1);
        X.colwise() += trans;

        energys.push_back(energy);

        if (par.print_energy)
            std::cout << "icp iter = " << icp << ", Energy = " << energy << ", gt_mse = " << gt_mse<< std::endl;

        /// Stopping criteria
        double stop2 = (final_transformation - To2).norm();
        To2 = final_transformation;
        if (stop2 < par.stop && icp) break;
    }

    W = (X - Q).colwise().norm();
    double last_energy = get_energy(par.f, W, par.p);
    gt_mse = pow((X - X_gt).norm(),2) / X.cols();

    final_transformation.block(0,3,3,1) += -final_transformation.block(0, 0, 3, 3)*source_mean + target_mean;
    X.colwise() += target_mean;

    ///save convergence result
    par.convergence_energy = last_energy;
    par.convergence_gt_mse = gt_mse;
    par.convergence_iter = icp;
    par.res_trans = final_transformation;

    ///output
    if (par.print_output)
    {
        std::ofstream out_res(par.out_path);
        if (!out_res.is_open())
        {
            std::cout << "Can't open out file " << par.out_path << std::endl;
        }
        ///output time and energy
        out_res.precision(16);
        for (int i = 0; i<icp; i++)
        {
            out_res << times[i] << " " << energys[i] <<" " << gt_mses[i] << std::endl;
        }
        out_res.close();
    }
}
}

namespace PairWiseAlign::SICP
{
/// Shrinkage operator (Automatic loop unrolling using template)
template<unsigned int I>
inline double
shrinkage(double mu, double n, double p, double s)
{
    return shrinkage<I - 1>(mu, n, p, 1.0 - (p / mu) * std::pow(n, p - 2.0) * std::pow(s, p - 1.0));
}
template<>
inline double
shrinkage<0>(double, double, double, double s) { return s; }
/// 3D Shrinkage for point-to-point
template<unsigned int I>
inline void
shrink(Eigen::Matrix3Xd &Q, double mu, double p)
{
    double Ba = std::pow((2.0 / mu) * (1.0 - p), 1.0 / (2.0 - p));
    double ha = Ba + (p / mu) * std::pow(Ba, p - 1.0);
#pragma omp parallel for
    for (int i = 0; i < Q.cols(); ++i)
    {
        double n = Q.col(i).norm();
        double w = 0.0;
        if (n > ha)
            w = shrinkage<I>(mu, n, p, (Ba / n + 1.0) / 2.0);
        Q.col(i) *= w;
    }
}
/// 1D Shrinkage for point-to-plane
template<unsigned int I>
inline void
shrink(Eigen::VectorXd &y, double mu, double p)
{
    double Ba = std::pow((2.0 / mu) * (1.0 - p), 1.0 / (2.0 - p));
    double ha = Ba + (p / mu) * std::pow(Ba, p - 1.0);
#pragma omp parallel for
    for (int i = 0; i < y.rows(); ++i)
    {
        double n = std::abs(y(i));
        double s = 0.0;
        if (n > ha)
            s = shrinkage<I>(mu, n, p, (Ba / n + 1.0) / 2.0);
        y(i) *= s;
    }
}
/// Sparse ICP with point to point
/// @param Source (one 3D point per column)
/// @param Target (one 3D point per column)
/// @param Parameters
void
point_to_point(Eigen::Matrix<Scalar, 3, Eigen::Dynamic> & X,
               Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& Y,
               Eigen::Matrix<Scalar, 3, 1>& source_mean,
               Eigen::Matrix<Scalar, 3, 1>& target_mean,
               Parameters& par)
{
    /// Build kd-tree
    _nanoflann::KDTreeAdaptor<Eigen::Matrix<Scalar, 3, Eigen::Dynamic>, 3, _nanoflann::metric_L2_Simple> kdtree(Y);
    /// Buffers
    Eigen::Matrix3Xd Q = Eigen::Matrix3Xd::Zero(3, X.cols());
    Eigen::Matrix3Xd Z = Eigen::Matrix3Xd::Zero(3, X.cols());
    Eigen::Matrix3Xd C = Eigen::Matrix3Xd::Zero(3, X.cols());
    Eigen::Matrix3Xd ori_X = X;
    Eigen::Affine3d T(par.init_trans);
    Eigen::Matrix3Xd X_gt;
    int nPoints = X.cols();
    X = T * X;
    Eigen::Matrix3Xd Xo1 = X;
    Eigen::Matrix3Xd Xo2 = X;
    double gt_mse = 0.0, run_time;
    double begin_time, end_time;
    std::vector<double> gt_mses, times;

    if (par.has_groundtruth)
    {
        Eigen::Vector3d temp_trans = par.gt_trans.block(0, 3, 3, 1);
        X_gt = ori_X;
        X_gt.colwise() += source_mean;
        X_gt = par.gt_trans.block(0, 0, 3, 3) * X_gt;
        X_gt.colwise() += temp_trans - target_mean;
    }

    begin_time = omp_get_wtime();

    /// ICP
    int icp;
    for (icp = 0; icp < par.max_icp; ++icp)
    {
        /// Find closest point
#pragma omp parallel for
        for (int i = 0; i < X.cols(); ++i)
        {
            Q.col(i) = Y.col(kdtree.closest(X.col(i).data()));
        }

        end_time = omp_get_wtime();
        run_time = end_time - begin_time;
        ///calc mse and gt_mse
        if (par.has_groundtruth)
        {
            gt_mse = (X - X_gt).squaredNorm() / nPoints;
        }
        times.push_back(run_time);
        gt_mses.push_back(gt_mse);
//            if(par.print_icpn)
//                std::cout << "iter = " << icp << ", time = " << run_time << ", mse = " << mse << ", gt_mse = " << gt_mse << std::endl;

        /// Computer rotation and translation
        double mu = par.mu;
        for (int outer = 0; outer < par.max_outer; ++outer)
        {
            double dual = 0.0;
            for (int inner = 0; inner < par.max_inner; ++inner)
            {
                /// Z update (shrinkage)
                Z = X - Q + C / mu;
                shrink<3>(Z, mu, par.p);
                /// Rotation and translation update
                Eigen::Matrix3Xd U = Q + Z - C / mu;
                Eigen::Affine3d cur_T = RigidMotionEstimator::point_to_point(X, U);
                X = cur_T * X;
                T = cur_T * T;
                /// Stopping criteria
                dual = pow((X - Xo1).norm(), 2) / nPoints;
                Xo1 = X;
                if (dual < par.stop)
                    break;
            }
            /// C update (lagrange multipliers)
            Eigen::Matrix3Xd P = X - Q - Z;
            if (!par.use_penalty)
                C.noalias() += mu * P;
            /// mu update (penalty)
            if (mu < par.max_mu)
                mu *= par.alpha;
            /// Stopping criteria
            double primal = P.colwise().norm().maxCoeff();
            if (primal < par.stop && dual < par.stop)
                break;
        }

        /// Stopping criteria
        double stop = (X - Xo2).colwise().norm().maxCoeff();
        Xo2 = X;
        if (stop < par.stop)
            break;
    }
    if (par.has_groundtruth)
        gt_mse = (X - X_gt).squaredNorm() / nPoints;

    if (par.print_icpn)
    {
        std::ofstream out_res(par.out_path);
        for (int i = 0; i < times.size(); i++)
        {
            out_res << times[i] << " " << gt_mses[i] << std::endl;
        }
        out_res.close();
    }

    T.translation().noalias() += -T.rotation() * source_mean + target_mean;
    X.colwise() += target_mean;

    ///save convergence result
    par.convergence_gt_mse = gt_mse;
    par.convergence_iter = icp;
    par.res_trans = T.matrix();
}
/// Sparse ICP with point to plane
/// @param Source (one 3D point per column)
/// @param Target (one 3D point per column)
/// @param Target normals (one 3D normal per column)
/// @param Parameters
void
point_to_plane(Eigen::Matrix<Scalar, 3, Eigen::Dynamic> &X,
               Eigen::Matrix<Scalar, 3, Eigen::Dynamic> &Y,
               Eigen::Matrix<Scalar, 3, Eigen::Dynamic> &N,
               Eigen::Matrix<Scalar, 3, 1> source_mean,
               Eigen::Matrix<Scalar, 3, 1> target_mean,
               Parameters &par)
{
    /// Build kd-tree
    _nanoflann::KDTreeAdaptor<Eigen::Matrix<Scalar, 3, Eigen::Dynamic>, 3, _nanoflann::metric_L2_Simple> kdtree(Y);
    /// Buffers
    Eigen::Matrix3Xd Qp = Eigen::Matrix3Xd::Zero(3, X.cols());
    Eigen::Matrix3Xd Qn = Eigen::Matrix3Xd::Zero(3, X.cols());
    Eigen::VectorXd Z = Eigen::VectorXd::Zero(X.cols());
    Eigen::VectorXd C = Eigen::VectorXd::Zero(X.cols());
    Eigen::Matrix3Xd ori_X = X;
    Eigen::Affine3d T(par.init_trans);
    Eigen::Matrix3Xd X_gt;
    int nPoints = X.cols();
    X = T * X;
    Eigen::Matrix3Xd Xo1 = X;
    Eigen::Matrix3Xd Xo2 = X;
    double gt_mse = 0.0, run_time;
    double begin_time, end_time;
    std::vector<double> gt_mses, times;

    if (par.has_groundtruth)
    {
        Eigen::Vector3d temp_trans = par.gt_trans.block(0, 3, 3, 1);
        X_gt = ori_X;
        X_gt.colwise() += source_mean;
        X_gt = par.gt_trans.block(0, 0, 3, 3) * X_gt;
        X_gt.colwise() += temp_trans - target_mean;
    }

    begin_time = omp_get_wtime();

    /// ICP
    int icp;
    int total_iters = 0;
    for (icp = 0; icp < par.max_icp; ++icp)
    {

        /// Find closest point
#pragma omp parallel for
        for (int i = 0; i < X.cols(); ++i)
        {
            int id = kdtree.closest(X.col(i).data());
            Qp.col(i) = Y.col(id);
            Qn.col(i) = N.col(id);
        }

        end_time = omp_get_wtime();
        run_time = end_time - begin_time;
        ///calc mse and gt_mse
        if (par.has_groundtruth)
        {
            gt_mse = (X - X_gt).squaredNorm() / nPoints;
        }
        times.push_back(run_time);
        gt_mses.push_back(gt_mse);

        if (par.print_icpn)
            std::cout << "iter = " << icp << ", time = " << run_time << ", gt_mse = " << gt_mse << std::endl;

        /// Computer rotation and translation
        double mu = par.mu;
        for (int outer = 0; outer < par.max_outer; ++outer)
        {
            double dual = 0.0;
            for (int inner = 0; inner < par.max_inner; ++inner)
            {
                total_iters++;
                /// Z update (shrinkage)
                Z = (Qn.array() * (X - Qp).array()).colwise().sum().transpose() + C.array() / mu;
                shrink<3>(Z, mu, par.p);
                /// Rotation and translation update
                Eigen::VectorXd U = Z - C / mu;
                T = RigidMotionEstimator::point_to_plane(X, Qp, Qn, Eigen::VectorXd::Ones(X.cols()), U) * T;
                /// Stopping criteria
                dual = (X - Xo1).colwise().norm().maxCoeff();
                Xo1 = X;
                if (dual < par.stop)
                    break;
            }
            /// C update (lagrange multipliers)
            Eigen::VectorXd P = (Qn.array() * (X - Qp).array()).colwise().sum().transpose() - Z.array();
            if (!par.use_penalty)
                C.noalias() += mu * P;
            /// mu update (penalty)
            if (mu < par.max_mu)
                mu *= par.alpha;
            /// Stopping criteria
            double primal = P.array().abs().maxCoeff();
            if (primal < par.stop && dual < par.stop)
                break;
        }
        /// Stopping criteria
        double stop = (X - Xo2).colwise().norm().maxCoeff();
        Xo2 = X;
        if (stop < par.stop)
            break;
    }
    if (par.has_groundtruth)
    {
        gt_mse = (X - X_gt).squaredNorm() / nPoints;
    }
    if (par.print_icpn)
    {
        std::ofstream out_res(par.out_path);
        for (int i = 0; i < times.size(); i++)
        {
            out_res << times[i] << " " << gt_mses[i] << std::endl;
        }
        out_res.close();
    }

    T.translation() += -T.rotation() * source_mean + target_mean;
    X.colwise() += target_mean;
    ///save convergence result
    par.convergence_gt_mse = gt_mse;
    par.convergence_iter = icp;
    par.res_trans = T.matrix();
    par.total_iters = total_iters;
}
}
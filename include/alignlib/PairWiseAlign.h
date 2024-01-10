#ifndef GWALIGN_H
#define GWALIGN_H
#include "ICP.h"
#include <unsupported/Eigen/MatrixFunctions>

#include "median.h"
#include <limits>
#define SAME_THRESHOLD 1e-6
#include <type_traits>
namespace PairWiseAlign::Align
{

typedef double Scalar;

typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> MatrixNX;

typedef Eigen::Matrix<Scalar, 3, 3> MatrixNN;

typedef Eigen::Matrix<Scalar, 3 + 1, 3 + 1> AffineMatrixN;

typedef Eigen::Transform<Scalar, 3, Eigen::Affine> AffineNd;

typedef Eigen::Matrix<Scalar, 3, 1> VectorN;

typedef _nanoflann::KDTreeAdaptor<MatrixNX, 3, _nanoflann::metric_L2_Simple> KDtree;

typedef Eigen::Matrix<Scalar, 6, 1> Vector6;

template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
almost_equal(T x, T y, int ulp);
//template<int N>

class MYICP
{
public:
    long N = 3;

    double test_total_construct_time = .0;
    double test_total_solve_time = .0;
    int test_total_iters = 0;
    Eigen::Matrix4f init_s2t;

    MYICP();
    ~MYICP();

private:
    AffineMatrixN
    LogMatrix(const AffineMatrixN &T);

    inline Vector6
    RotToEuler(const AffineNd &T);

    inline AffineMatrixN
    EulerToRot(const Vector6 &v);
    inline Vector6
    LogToVec(const Eigen::Matrix4d &LogT);

    inline AffineMatrixN
    VecToLog(const Vector6 &v);

    double
    FindKnearestMed(const KDtree &kdtree,
                    const MatrixNX &X, int nk);
    /// Find self normal edge median of point cloud
    double
    FindKnearestNormMed(const KDtree &kdtree, const Eigen::Matrix3Xd &X, int nk, const Eigen::Matrix3Xd &norm_x);

    template<typename Derived1, typename Derived2, typename Derived3>
    AffineNd
    point_to_point(Eigen::MatrixBase<Derived1> &X,
                   Eigen::MatrixBase<Derived2> &Y,
                   const Eigen::MatrixBase<Derived3> &w);

    template<typename Derived1, typename Derived2, typename Derived3, typename Derived4, typename Derived5>
    Eigen::Affine3d
    point_to_plane(Eigen::MatrixBase<Derived1> &X,
                   Eigen::MatrixBase<Derived2> &Y,
                   const Eigen::MatrixBase<Derived3> &Norm,
                   const Eigen::MatrixBase<Derived4> &w,
                   const Eigen::MatrixBase<Derived5> &u);

    template<typename Derived1, typename Derived2, typename Derived3, typename Derived4>
    double
    point_to_plane_gaussnewton(const Eigen::MatrixBase<Derived1> &X,
                               const Eigen::MatrixBase<Derived2> &Y,
                               const Eigen::MatrixBase<Derived3> &norm_y,
                               const Eigen::MatrixBase<Derived4> &w,
                               Matrix44 Tk, Vector6 &dir);

public:
    void
    point_to_point(MatrixNX &X, MatrixNX &Y, VectorN &source_mean,
                   VectorN &target_mean, ICP::Parameters &par);

    /// Reweighted ICP with point to plane
    /// @param Source (one 3D point per column)
    /// @param Target (one 3D point per column)
    /// @param Target normals (one 3D normal per column)
    /// @param Parameters
    //    template <typename Derived1, typename Derived2, typename Derived3>
    void
    point_to_plane(Eigen::Matrix3Xd &X,
                   Eigen::Matrix3Xd &Y, Eigen::Matrix3Xd &norm_x, Eigen::Matrix3Xd &norm_y,
                   Eigen::Vector3d &source_mean, Eigen::Vector3d &target_mean,
                   ICP::Parameters &par);

    /// Reweighted ICP with point to plane
    /// @param Source (one 3D point per column)
    /// @param Target (one 3D point per column)
    /// @param Target normals (one 3D normal per column)
    /// @param Parameters
    //    template <typename Derived1, typename Derived2, typename Derived3>
    void
    point_to_plane_GN(Eigen::Matrix3Xd &X,
                      Eigen::Matrix3Xd &Y, Eigen::Matrix3Xd &norm_x, Eigen::Matrix3Xd &norm_y,
                      Eigen::Vector3d &source_mean, Eigen::Vector3d &target_mean,
                      ICP::Parameters &par);
};

}


namespace PairWiseAlign::AAICP{
typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
typedef Eigen::Matrix<Scalar, 6, 1> Vector6;
typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
typedef Eigen::Matrix<Scalar, 6, Eigen::Dynamic> Matrix6X;

///aaicp
///////////////////////////////////////////////////////////////////////////////////////////
Vector6 Matrix42Vector6 (const Matrix4 m);

///////////////////////////////////////////////////////////////////////////////////////////
Matrix4 Vector62Matrix4 (const Vector6 v);

///////////////////////////////////////////////////////////////////////////////////////////
int alphas_cond (VectorX alphas);

///////////////////////////////////////////////////////////////////////////////////////////
VectorX get_alphas_lstsq (const Matrix6X f);

///////////////////////////////////////////////////////////////////////////////////////////
VectorX get_next_u (const Matrix6X u, const Matrix6X g, const Matrix6X f, std::vector<double> & save_alphas);


void point_to_point_aaicp(Eigen::Matrix<Scalar, 3, Eigen::Dynamic> & X,
                          Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& Y,
                          Eigen::Matrix<Scalar, 3, 1>& source_mean,
                          Eigen::Matrix<Scalar, 3, 1>& target_mean,
                          ICP::Parameters& par);
}

namespace PairWiseAlign::SICP
{
struct Parameters
{
    bool use_penalty = false; /// if use_penalty then penalty method else ADMM or ALM (see max_inner)
    double p = 1.0;           /// p norm
    double mu = 10.0;         /// penalty weight
    double alpha = 1.2;       /// penalty increase factor
    double max_mu = 1e5;      /// max penalty
    int max_icp = 100;        /// max ICP iteration
    int max_outer = 100;      /// max outer iteration
    int max_inner = 1;        /// max inner iteration. If max_inner=1 then ADMM else ALM
    double stop = 1e-5;       /// stopping criteria
    bool print_icpn = false;  /// (debug) print ICP iteration
    Eigen::Matrix4d init_trans = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d gt_trans = Eigen::Matrix4d::Identity();
    bool has_groundtruth = false;
    int convergence_iter = 0;
    double convergence_mse = 0.0;
    double convergence_gt_mse = 0.0;
    Eigen::Matrix4d res_trans = Eigen::Matrix4d::Identity();
    std::string file_err = "";
    std::string out_path = "";
    int total_iters = 0;
};
/// Shrinkage operator (Automatic loop unrolling using template)
template<unsigned int I>
inline double
shrinkage(double mu, double n, double p, double s);
template<>
inline double
shrinkage<0>(double, double, double, double s);
/// 3D Shrinkage for point-to-point
template<unsigned int I>
inline void
shrink(Eigen::Matrix3Xd &Q, double mu, double p);
/// 1D Shrinkage for point-to-plane
template<unsigned int I>
inline void
shrink(Eigen::VectorXd &y, double mu, double p);
/// Sparse ICP with point to point
/// @param Source (one 3D point per column)
/// @param Target (one 3D point per column)
/// @param Parameters
void
point_to_point(Eigen::Matrix<Scalar, 3, Eigen::Dynamic> & X,
               Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& Y,
               Eigen::Matrix<Scalar, 3, 1>& source_mean,
               Eigen::Matrix<Scalar, 3, 1>& target_mean,
               Parameters& par);
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
               Parameters &par);
}

///////////////////////////////////////////////////////////////////////////////
/// ICP implementation using ADMM/ALM/Penalty method

#endif

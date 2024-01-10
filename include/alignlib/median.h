#ifndef MEDIAN_H
#define MEDIAN_H
#include <Eigen/Dense>
#include <vector>
namespace igl
{
    template <typename DerivedM>
void matrix_to_list(const Eigen::DenseBase<DerivedM>& M,
                    std::vector<typename DerivedM::Scalar> &V)
{
    using namespace std;
    V.resize(M.size());
    // loop over cols then rows
    for(int j =0; j<M.cols();j++)
    {
        for(int i = 0; i < M.rows();i++)
        {
            V[i+j*M.rows()] = M(i,j);
        }
    }
}
  template <typename DerivedV, typename mType>
  bool median(
    const Eigen::MatrixBase<DerivedV> & V, mType & m)
  {
    using namespace std;
    if(V.size() == 0)
    {
      return false;
    }
    vector<typename DerivedV::Scalar> vV;
    matrix_to_list(V,vV);
    // http://stackoverflow.com/a/1719155/148668
    size_t n = vV.size()/2;
    nth_element(vV.begin(),vV.begin()+n,vV.end());
    if(vV.size()%2==0)
    {
      nth_element(vV.begin(),vV.begin()+n-1,vV.end());
      m = 0.5*(vV[n]+vV[n-1]);
    }else
    {
      m = vV[n];
    }
    return true;
  }
}
#endif

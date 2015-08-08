/*
 * TriHidderLayer.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_TriHidderLayer_H_
#define SRC_TriHidderLayer_H_
#include "tensor.h"
#include "Utiltensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class TriHidderLayer {

public:

  Tensor<xpu, 2, double> _W1;
  Tensor<xpu, 2, double> _W2;
  Tensor<xpu, 2, double> _W3;
  Tensor<xpu, 2, double> _b;

  Tensor<xpu, 2, double> _gradW1;
  Tensor<xpu, 2, double> _gradW2;
  Tensor<xpu, 2, double> _gradW3;
  Tensor<xpu, 2, double> _gradb;

  Tensor<xpu, 2, double> _eg2W1;
  Tensor<xpu, 2, double> _eg2W2;
  Tensor<xpu, 2, double> _eg2W3;
  Tensor<xpu, 2, double> _eg2b;

  bool _bUseB;

  int _funcType; // 0: tanh, 1: sigmod, 2: f(x)=x

public:
  TriHidderLayer(){}

  inline void initial(int nOSize, int nISize1, int nISize2, int nISize3, bool bUseB=true, int seed = 0, int funcType = 0) {
     double bound = sqrt(6.0 / (nOSize + nISize1 + nISize2 + nISize3 + 1));

     _W1 = NewTensor<xpu>(Shape2(nOSize, nISize1), 0.0);
     _gradW1 = NewTensor<xpu>(Shape2(nOSize, nISize1), 0.0);
     _eg2W1 = NewTensor<xpu>(Shape2(nOSize, nISize1), 0.0);

     _W2 = NewTensor<xpu>(Shape2(nOSize, nISize2), 0.0);
     _gradW2 = NewTensor<xpu>(Shape2(nOSize, nISize2), 0.0);
     _eg2W2 = NewTensor<xpu>(Shape2(nOSize, nISize2), 0.0);

     _W3 = NewTensor<xpu>(Shape2(nOSize, nISize3), 0.0);
     _gradW3 = NewTensor<xpu>(Shape2(nOSize, nISize3), 0.0);
     _eg2W3 = NewTensor<xpu>(Shape2(nOSize, nISize3), 0.0);

     _b = NewTensor<xpu>(Shape2(1, nOSize), 0.0);
     _gradb = NewTensor<xpu>(Shape2(1, nOSize), 0.0);
     _eg2b = NewTensor<xpu>(Shape2(1, nOSize), 0.0);


     Random<xpu, double> rnd(seed);
     rnd.SampleUniform(&_W1, -1.0 * bound, 1.0 *bound);
     rnd.SampleUniform(&_W2, -1.0 * bound, 1.0 *bound);
     rnd.SampleUniform(&_W3, -1.0 * bound, 1.0 *bound);
     rnd.SampleUniform(&_b, -1.0 * bound, 1.0 *bound);



     _bUseB = bUseB;
     _funcType = funcType;
   }

   inline void initial(const Tensor<xpu, 2, double>& W1, const Tensor<xpu, 2, double>& W2,
       const Tensor<xpu, 2, double>& W3, const Tensor<xpu, 2, double>& b,
       bool bUseB=true, int funcType = 0) {
     static int nOSize, nISize1, nISize2, nISize3;
     nOSize = W1.size(0);
     nISize1 = W1.size(1);
     nISize2 = W2.size(1);
     nISize3 = W3.size(1);


     _W1 = NewTensor<xpu>(Shape2(nOSize, nISize1), 0.0);
     _gradW1 = NewTensor<xpu>(Shape2(nOSize, nISize1), 0.0);
     _eg2W1 = NewTensor<xpu>(Shape2(nOSize, nISize1), 0.0);
     Copy(_W1, W1);

     _W2 = NewTensor<xpu>(Shape2(nOSize, nISize2), 0.0);
     _gradW2 = NewTensor<xpu>(Shape2(nOSize, nISize2), 0.0);
     _eg2W2 = NewTensor<xpu>(Shape2(nOSize, nISize2), 0.0);
     Copy(_W2, W2);

     _W3 = NewTensor<xpu>(Shape2(nOSize, nISize3), 0.0);
     _gradW3 = NewTensor<xpu>(Shape2(nOSize, nISize3), 0.0);
     _eg2W3 = NewTensor<xpu>(Shape2(nOSize, nISize3), 0.0);
     Copy(_W3, W3);

     _b = NewTensor<xpu>(Shape2(1, nOSize), 0.0);
     _gradb = NewTensor<xpu>(Shape2(1, nOSize), 0.0);
     _eg2b = NewTensor<xpu>(Shape2(1, nOSize), 0.0);

     if(bUseB)Copy(_b, b);

     _bUseB = bUseB;
     _funcType = funcType;
   }

   inline void release(){
     FreeSpace(&_W1); FreeSpace(&_gradW1); FreeSpace(&_eg2W1);
     FreeSpace(&_W2); FreeSpace(&_gradW2); FreeSpace(&_eg2W2);
     FreeSpace(&_W3); FreeSpace(&_gradW3); FreeSpace(&_eg2W3);
     FreeSpace(&_b); FreeSpace(&_gradb); FreeSpace(&_eg2b);
   }


  virtual ~TriHidderLayer() {
    // TODO Auto-generated destructor stub
  }

  inline double squarenormAll()
  {
    double result = squarenorm(_gradW1);
    result += squarenorm(_gradW2);
    result += squarenorm(_gradW3);
    if(_bUseB)
    {
      result += squarenorm(_gradb);
    }

    return result;
  }

  inline void scaleGrad(double scale)
  {
    _gradW1 = _gradW1 * scale;
    _gradW2 = _gradW2 * scale;
    _gradW3 = _gradW3 * scale;
    if(_bUseB)
    {
      _gradb = _gradb * scale;
    }
  }

public:
  inline void ComputeForwardScore(const Tensor<xpu, 2, double> &x1, const Tensor<xpu, 2, double> &x2,
      const Tensor<xpu, 2, double> &x3, Tensor<xpu, 2, double> &y)
  {
    y = dot(x1, _W1.T());
    y += dot(x2, _W2.T());
    y += dot(x3, _W3.T());
    if(_bUseB)y = y + _b;
    if(_funcType == 0)y = F<nl_tanh>(y);
    else if(_funcType == 1) y = F<nl_sigmoid>(y);
  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(const Tensor<xpu, 2, double>& x1, const Tensor<xpu, 2, double>& x2,
      const Tensor<xpu, 2, double>& x3, const Tensor<xpu, 2, double>& y, const Tensor<xpu, 2, double>& ly,
      Tensor<xpu, 2, double>& lx1, Tensor<xpu, 2, double>& lx2, Tensor<xpu, 2, double>& lx3)
  {
    //_gradW
    Tensor<xpu, 2, double> deri_yx(Shape2(y.size(0), y.size(1))), cly(Shape2(y.size(0), y.size(1)));
    AllocSpace(&deri_yx); AllocSpace(&cly);
    if(_funcType == 0)
     {
       deri_yx = F<nl_dtanh>(y);
       cly = ly * deri_yx;
     }
     else if(_funcType == 1)
     {
       deri_yx = F<nl_dsigmoid>(y);
       cly = ly * deri_yx;
     }
     else
     {
       //cly = ly;
       Copy(cly, ly);
     }
    //_gradW
    _gradW1 += dot(cly.T(), x1);
    _gradW2 += dot(cly.T(), x2);
    _gradW3 += dot(cly.T(), x3);

    //_gradb
    if(_bUseB)_gradb += cly;

    //lx
    lx1 = dot(cly, _W1);
    lx2 = dot(cly, _W2);
    lx3 = dot(cly, _W3);

    FreeSpace(&deri_yx); FreeSpace(&cly);
  }

  inline void randomprint(int num)
  {
    static int nOSize, nISize1, nISize2, nISize3;
    nOSize = _W1.size(0);
    nISize1 = _W1.size(1);
    nISize2 = _W2.size(1);
    nISize3 = _W3.size(1);
    int count = 0;
    while(count < num)
    {
      int idx1 = rand()%nOSize;
      int idy1 = rand()%nISize1;
      int idx2 = rand()%nOSize;
      int idy2 = rand()%nISize2;
      int idx3 = rand()%nOSize;
      int idy3 = rand()%nISize3;


      std::cout << "_W1[" << idx1 << "," << idy1 << "]=" << _W1[idx1][idy1] << " ";
      std::cout << "_W2[" << idx2 << "," << idy2 << "]=" << _W2[idx2][idy2] << " ";
      std::cout << "_W3[" << idx3 << "," << idy3 << "]=" << _W3[idx3][idy3] << " ";

      if(_bUseB)
      {
          int idz = rand()%nOSize;
          std::cout << "_b[0][" << idz << "]=" << _b[0][idz] << " ";
      }
      count++;
    }

    std::cout << std::endl;
  }

  inline void updateAdaGrad(double regularizationWeight, double adaAlpha, double adaEps)
  {
    _gradW1 = _gradW1 + _W1 * regularizationWeight;
    _eg2W1 = _eg2W1 + _gradW1 * _gradW1;
    _W1 = _W1 - _gradW1 * adaAlpha / F<nl_sqrt>(_eg2W1 + adaEps);

    _gradW2 = _gradW2 + _W2 * regularizationWeight;
    _eg2W2 = _eg2W2 + _gradW2 * _gradW2;
    _W2 = _W2 - _gradW2 * adaAlpha / F<nl_sqrt>(_eg2W2 + adaEps);

    _gradW3 = _gradW3 + _W3 * regularizationWeight;
    _eg2W3 = _eg2W3 + _gradW3 * _gradW3;
    _W3 = _W3 - _gradW3 * adaAlpha / F<nl_sqrt>(_eg2W3 + adaEps);

    if(_bUseB)
    {
      _gradb = _gradb + _b * regularizationWeight;
      _eg2b = _eg2b + _gradb * _gradb;
      _b = _b - _gradb * adaAlpha / F<nl_sqrt>(_eg2b + adaEps);
    }


    clearGrad();
  }

  inline void clearGrad()
  {
    _gradW1 = 0;
    _gradW2 = 0;
    _gradW3 = 0;
    if(_bUseB)_gradb = 0;
  }
};

#endif /* SRC_TriHidderLayer_H_ */

/*
 * UniHidderLayer.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_UniHidderLayer_H_
#define SRC_UniHidderLayer_H_
#include "tensor.h"
#include "Utiltensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class UniHidderLayer {

public:

  Tensor<xpu, 2, double> _W;
  Tensor<xpu, 2, double> _b;

  Tensor<xpu, 2, double> _gradW;
  Tensor<xpu, 2, double> _gradb;

  Tensor<xpu, 2, double> _eg2W;
  Tensor<xpu, 2, double> _eg2b;

  bool _bUseB;

  int _funcType; // 0: tanh, 1: sigmod, 2: f(x)=x

public:
  UniHidderLayer(){}

  inline void initial(int nOSize, int nISize, bool bUseB=true, int seed = 0, int funcType = 0) {
     double bound = sqrt(6.0 / (nOSize + nISize+1));

     _W = NewTensor<xpu>(Shape2(nOSize, nISize), 0.0);
     _gradW = NewTensor<xpu>(Shape2(nOSize, nISize), 0.0);
     _eg2W = NewTensor<xpu>(Shape2(nOSize, nISize), 0.0);


     _b = NewTensor<xpu>(Shape2(1, nOSize), 0.0);
     _gradb = NewTensor<xpu>(Shape2(1, nOSize), 0.0);
     _eg2b = NewTensor<xpu>(Shape2(1, nOSize), 0.0);


     Random<xpu, double> rnd(seed);
     rnd.SampleUniform(&_W, -1.0 * bound, 1.0 *bound);
     rnd.SampleUniform(&_b, -1.0 * bound, 1.0 *bound);


     _bUseB = bUseB;
     _funcType = funcType;
   }

   inline void initial(const Tensor<xpu, 2, double>& W, const Tensor<xpu, 2, double>& b,
		   bool bUseB=true, int funcType = 0) {
     static int nOSize, nISize;
     nOSize = W.size(0);
     nISize = W.size(1);


     _W = NewTensor<xpu>(Shape2(nOSize, nISize), 0.0);
     _gradW = NewTensor<xpu>(Shape2(nOSize, nISize), 0.0);
     _eg2W = NewTensor<xpu>(Shape2(nOSize, nISize), 0.0);
     Copy(_W, W);

     _b = NewTensor<xpu>(Shape2(1, nOSize), 0.0);
     _gradb = NewTensor<xpu>(Shape2(1, nOSize), 0.0);
     _eg2b = NewTensor<xpu>(Shape2(1, nOSize), 0.0);

     if(bUseB)Copy(_b, b);

     _bUseB = bUseB;
     _funcType = funcType;
   }

   inline void release(){
     FreeSpace(&_W); FreeSpace(&_gradW); FreeSpace(&_eg2W);
     FreeSpace(&_b); FreeSpace(&_gradb); FreeSpace(&_eg2b);
   }


  virtual ~UniHidderLayer() {
    // TODO Auto-generated destructor stub
  }

  inline double squarenormAll()
  {
    double result = squarenorm(_gradW);


    if(_bUseB)
    {
    	result += squarenorm(_gradb);
    }

    return result;
  }

  inline void scaleGrad(double scale)
  {
    _gradW = _gradW * scale;
    if(_bUseB)
    {
      _gradb = _gradb * scale;
    }
  }

public:
  inline void ComputeForwardScore(const Tensor<xpu, 2, double> &x, Tensor<xpu, 2, double> &y)
  {
    y = dot(x, _W.T());
    if(_bUseB)y = y + _b;
    if(_funcType == 0)y = F<nl_tanh>(y);
    else if(_funcType == 1) y = F<nl_sigmoid>(y);
  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(const Tensor<xpu, 2, double>& x, const Tensor<xpu, 2, double>& y,
		  const Tensor<xpu, 2, double>& ly, Tensor<xpu, 2, double>& lx)
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
    _gradW += dot(cly.T(), x);

    //_gradb
    if(_bUseB)_gradb += cly;

    //lx
    lx = dot(cly, _W);

    FreeSpace(&deri_yx); FreeSpace(&cly);
  }

  inline void randomprint(int num)
  {
    static int nOSize, nISize;
    nOSize = _W.size(0);
    nISize = _W.size(1);
    int count = 0;
    while(count < num)
    {
      int idx = rand()%nOSize;
      int idy = rand()%nISize;


      std::cout << "_W[" << idx << "," << idy << "]=" << _W[idx][idy] << " ";

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
    _gradW = _gradW + _W * regularizationWeight;
    _eg2W = _eg2W + _gradW * _gradW;
    _W = _W - _gradW * adaAlpha / F<nl_sqrt>(_eg2W + adaEps);


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
    _gradW = 0;
    if(_bUseB)_gradb = 0;
  }
};

#endif /* SRC_UniHidderLayer_H_ */

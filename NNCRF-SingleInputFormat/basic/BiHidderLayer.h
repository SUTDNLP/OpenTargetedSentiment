/*
 * BiHidderLayer.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_BiHidderLayer_H_
#define SRC_BiHidderLayer_H_
#include "tensor.h"
#include "Utiltensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class BiHidderLayer {

public:

  Tensor<xpu, 2, double> _WL;
  Tensor<xpu, 2, double> _WR;
  Tensor<xpu, 2, double> _b;

  Tensor<xpu, 2, double> _gradWL;
  Tensor<xpu, 2, double> _gradWR;
  Tensor<xpu, 2, double> _gradb;

  Tensor<xpu, 2, double> _eg2WL;
  Tensor<xpu, 2, double> _eg2WR;
  Tensor<xpu, 2, double> _eg2b;

  bool _bUseB;

  int _funcType; // 0: tanh, 1: sigmod, 2: f(x)=x

public:
  BiHidderLayer(){}

  inline void initial(int nOSize, int nLISize, int nRISize, bool bUseB=true, int seed = 0, int funcType = 0) {
     double bound = sqrt(6.0 / (nOSize + nLISize + nRISize + 1));

     _WL = NewTensor<xpu>(Shape2(nOSize, nLISize), 0.0);
     _gradWL = NewTensor<xpu>(Shape2(nOSize, nLISize), 0.0);
     _eg2WL = NewTensor<xpu>(Shape2(nOSize, nLISize), 0.0);

     _WR = NewTensor<xpu>(Shape2(nOSize, nRISize), 0.0);
     _gradWR = NewTensor<xpu>(Shape2(nOSize, nRISize), 0.0);
     _eg2WR = NewTensor<xpu>(Shape2(nOSize, nRISize), 0.0);

     _b = NewTensor<xpu>(Shape2(1, nOSize), 0.0);
     _gradb = NewTensor<xpu>(Shape2(1, nOSize), 0.0);
     _eg2b = NewTensor<xpu>(Shape2(1, nOSize), 0.0);


     Random<xpu, double> rnd(seed);
     rnd.SampleUniform(&_WL, -1.0 * bound, 1.0 *bound);
     rnd.SampleUniform(&_WR, -1.0 * bound, 1.0 *bound);
     rnd.SampleUniform(&_b, -1.0 * bound, 1.0 *bound);



     _bUseB = bUseB;
     _funcType = funcType;
   }

   inline void initial(const Tensor<xpu, 2, double>& WL, const Tensor<xpu, 2, double>& WR, const Tensor<xpu, 2, double>& b,
		   bool bUseB=true, int funcType = 0) {
     static int nOSize, nLISize, nRISize;
     nOSize = WL.size(0);
     nLISize = WL.size(1);
     nRISize = WR.size(1);


     _WL = NewTensor<xpu>(Shape2(nOSize, nLISize), 0.0);
     _gradWL = NewTensor<xpu>(Shape2(nOSize, nLISize), 0.0);
     _eg2WL = NewTensor<xpu>(Shape2(nOSize, nLISize), 0.0);
     Copy(_WL, WL);

     _WR = NewTensor<xpu>(Shape2(nOSize, nRISize), 0.0);
     _gradWR = NewTensor<xpu>(Shape2(nOSize, nRISize), 0.0);
     _eg2WR = NewTensor<xpu>(Shape2(nOSize, nRISize), 0.0);
     Copy(_WR, WR);

     _b = NewTensor<xpu>(Shape2(1, nOSize), 0.0);
     _gradb = NewTensor<xpu>(Shape2(1, nOSize), 0.0);
     _eg2b = NewTensor<xpu>(Shape2(1, nOSize), 0.0);

     if(bUseB)Copy(_b, b);

     _bUseB = bUseB;
     _funcType = funcType;
   }

   inline void release(){
     FreeSpace(&_WL); FreeSpace(&_gradWL); FreeSpace(&_eg2WL);
     FreeSpace(&_WR); FreeSpace(&_gradWR); FreeSpace(&_eg2WR);
     FreeSpace(&_b); FreeSpace(&_gradb); FreeSpace(&_eg2b);
   }


  virtual ~BiHidderLayer() {
    // TODO Auto-generated destructor stub
  }

  inline double squarenormAll()
  {
    double result = squarenorm(_gradWL);
    result += squarenorm(_gradWR);
    if(_bUseB)
    {
    	result += squarenorm(_gradb);
    }

    return result;
  }

  inline void scaleGrad(double scale)
  {
    _gradWL = _gradWL * scale;
    _gradWR = _gradWR * scale;
    if(_bUseB)
    {
      _gradb = _gradb * scale;
    }
  }

public:
  inline void ComputeForwardScore(const Tensor<xpu, 2, double> &xl, const Tensor<xpu, 2, double> &xr, Tensor<xpu, 2, double> &y)
  {
    y = dot(xl, _WL.T());
    y += dot(xr, _WR.T());
    if(_bUseB)y = y + _b;
    if(_funcType == 0)y = F<nl_tanh>(y);
    else if(_funcType == 1) y = F<nl_sigmoid>(y);
  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(const Tensor<xpu, 2, double>& xl, const Tensor<xpu, 2, double>& xr,
      const Tensor<xpu, 2, double>& y, const Tensor<xpu, 2, double>& ly,
      Tensor<xpu, 2, double>& lxl, Tensor<xpu, 2, double>& lxr)
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
    _gradWL += dot(cly.T(), xl);
    _gradWR += dot(cly.T(), xr);

    //_gradb
    if(_bUseB)_gradb += cly;

    //lx
    lxl = dot(cly, _WL);
    lxr = dot(cly, _WR);

    FreeSpace(&deri_yx); FreeSpace(&cly);
  }

  inline void randomprint(int num)
  {
    static int nOSize, nLISize, nRISize;
    nOSize = _WL.size(0);
    nLISize = _WL.size(1);
    nRISize = _WR.size(1);
    int count = 0;
    while(count < num)
    {
      int idxl = rand()%nOSize;
      int idyl = rand()%nLISize;
      int idxr = rand()%nOSize;
      int idyr = rand()%nRISize;


      std::cout << "_WL[" << idxl << "," << idyl << "]=" << _WL[idxl][idyl] << " ";
      std::cout << "_WR[" << idxr << "," << idyr << "]=" << _WR[idxr][idyr] << " ";

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
    _gradWL = _gradWL + _WL * regularizationWeight;
    _eg2WL = _eg2WL + _gradWL * _gradWL;
    _WL = _WL - _gradWL * adaAlpha / F<nl_sqrt>(_eg2WL + adaEps);

    _gradWR = _gradWR + _WR * regularizationWeight;
    _eg2WR = _eg2WR + _gradWR * _gradWR;
    _WR = _WR - _gradWR * adaAlpha / F<nl_sqrt>(_eg2WR + adaEps);

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
    _gradWL = 0;
    _gradWR = 0;
    if(_bUseB)_gradb = 0;
  }
};

#endif /* SRC_BiHidderLayer_H_ */

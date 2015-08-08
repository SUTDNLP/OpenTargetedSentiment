/*
 * SparseUniHidderLayer.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_SparseUniHidderLayer_H_
#define SRC_SparseUniHidderLayer_H_
#include "tensor.h"
#include "Utiltensor.h"
#include "MyLib.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class SparseUniHidderLayer {

public:

  hash_set<int> _indexers;

  Tensor<xpu, 2, double> _W;
  Tensor<xpu, 2, double> _b;

  Tensor<xpu, 2, double> _gradW;
  Tensor<xpu, 2, double> _gradb;

  Tensor<xpu, 2, double> _eg2W;
  Tensor<xpu, 2, double> _eg2b;

  Tensor<xpu, 2, double> _ftW;

  bool _bUseB;

  int _funcType; // 0: tanh, 1: sigmod, 2: f(x)=x

public:

  SparseUniHidderLayer() {
  }

  inline void initial(int nOSize, int nISize, bool bUseB = true, int seed = 0, int funcType = 0) {
    double bound = sqrt(6.0 / (nOSize + nISize + 1));

    _W = NewTensor<xpu>(Shape2(nOSize, nISize), 0.0);
    _gradW = NewTensor<xpu>(Shape2(nOSize, nISize), 0.0);
    _eg2W = NewTensor<xpu>(Shape2(nOSize, nISize), 0.0);
    _ftW = NewTensor<xpu>(Shape2(nOSize, nISize), 1.0);

    _b = NewTensor<xpu>(Shape2(1, nOSize), 0.0);
    _gradb = NewTensor<xpu>(Shape2(1, nOSize), 0.0);
    _eg2b = NewTensor<xpu>(Shape2(1, nOSize), 0.0);

    Random<xpu, double> rnd(seed);
    rnd.SampleUniform(&_W, -1.0 * bound, 1.0 * bound);
    rnd.SampleUniform(&_b, -1.0 * bound, 1.0 * bound);

    _bUseB = bUseB;
    _funcType = funcType;
  }

  inline void initial(const Tensor<xpu, 2, double>& W, const Tensor<xpu, 2, double>& b, bool bUseB = true, int funcType = 0) {
    static int nOSize, nISize;
    nOSize = W.size(0);
    nISize = W.size(1);

    _W = NewTensor<xpu>(Shape2(nOSize, nISize), 0.0);
    _gradW = NewTensor<xpu>(Shape2(nOSize, nISize), 0.0);
    _eg2W = NewTensor<xpu>(Shape2(nOSize, nISize), 0.0);
    _ftW = NewTensor<xpu>(Shape2(nOSize, nISize), 1.0);
    Copy(_W, W);

    _b = NewTensor<xpu>(Shape2(1, nOSize), 0.0);
    _gradb = NewTensor<xpu>(Shape2(1, nOSize), 0.0);
    _eg2b = NewTensor<xpu>(Shape2(1, nOSize), 0.0);

    if (bUseB)
      Copy(_b, b);

    _bUseB = bUseB;
    _funcType = funcType;
  }

  inline void release() {
    FreeSpace(&_W);
    FreeSpace(&_gradW);
    FreeSpace(&_eg2W);
    FreeSpace(&_ftW);
    FreeSpace(&_b);
    FreeSpace(&_gradb);
    FreeSpace(&_eg2b);
  }

  virtual ~SparseUniHidderLayer() {
    // TODO Auto-generated destructor stub
  }

  inline double squarenormAll() {
    double result = 0;
    static hash_set<int>::iterator it;
    for (int idx = 0; idx < _gradW.size(0); idx++) {
      for (it = _indexers.begin(); it != _indexers.end(); ++it) {
        result += _gradW[idx][*it] * _gradW[idx][*it];
      }
    }

    if (_bUseB) {
      result += squarenorm(_gradb);
    }

    return result;
  }

  inline void scaleGrad(double scale) {
    static hash_set<int>::iterator it;
    for (int idx = 0; idx < _gradW.size(0); idx++) {
      for (it = _indexers.begin(); it != _indexers.end(); ++it) {
        _gradW[idx][*it] = _gradW[idx][*it] * scale;
      }
    }

    if (_bUseB) {
      _gradb = _gradb * scale;
    }
  }

public:
  void ComputeForwardScore(const std::vector<int>& x, Tensor<xpu, 2, double>& y) {
    static int featNum, featId, outDim;
    featNum = x.size();
    outDim = _W.size(0);
    y = 0.0;
    for (int idx = 0; idx < featNum; idx++) {
      featId = x[idx];
      for (int idy = 0; idy < outDim; idy++) {
        y[0][idy] += _W[idy][featId] / _ftW[idy][featId];
      }
    }

    if (_bUseB)
      y = y + _b;
    if (_funcType == 0)
      y = F<nl_tanh>(y);
    else if (_funcType == 1)
      y = F<nl_sigmoid>(y);

  }

  // loss is stopped at this layer, since the input is one-hold alike
  void ComputeBackwardLoss(const std::vector<int>& x, const Tensor<xpu, 2, double>& y, const Tensor<xpu, 2, double>& ly) {
    Tensor<xpu, 2, double> deri_yx(Shape2(y.size(0), y.size(1))), cly(Shape2(y.size(0), y.size(1)));
    AllocSpace(&deri_yx);
    AllocSpace(&cly);
    if (_funcType == 0) {
      deri_yx = F<nl_dtanh>(y);
      cly = ly * deri_yx;
    } else if (_funcType == 1) {
      deri_yx = F<nl_dsigmoid>(y);
      cly = ly * deri_yx;
    } else {
      //cly = ly;
      Copy(cly, ly);
    }
    //_gradW
    static int featNum, featId, outDim;
    featNum = x.size();
    outDim = _W.size(0);
    for (int idx = 0; idx < featNum; idx++) {
      featId = x[idx];
      _indexers.insert(featId);
      for (int idy = 0; idy < outDim; idy++) {
        _gradW[idy][featId] += cly[0][idy];
      }
    }

    if (_bUseB)
      _gradb = _gradb + cly;
  }

  void randomprint(int num) {
    static int nOSize, nISize;
    nOSize = _W.size(0);
    nISize = _W.size(1);
    int count = 0;
    while (count < num) {
      int idx = rand() % nOSize;
      int idy = rand() % nISize;

      std::cout << "_W[" << idx << "," << idy << "]=" << _W[idx][idy] << " ";

      if (_bUseB) {
        int idz = rand() % nOSize;
        std::cout << "_b[0][" << idz << "]=" << _b[0][idz] << " ";
      }
      count++;
    }

    std::cout << std::endl;
  }

  void updateAdaGrad(double regularizationWeight, double adaAlpha, double adaEps) {
    static int outDim;
    outDim = _W.size(0);
    static hash_set<int>::iterator it;

    for (it = _indexers.begin(); it != _indexers.end(); ++it) {
      int index = *it;
      for (int idx = 0; idx < outDim; idx++) {
        double _grad_wordEmb_ij = _gradW[idx][index] + regularizationWeight * _W[idx][index] / _ftW[idx][index];
        _eg2W[idx][index] += _grad_wordEmb_ij * _grad_wordEmb_ij;
        double tmp_normaize_alpha = sqrt(_eg2W[idx][index] + adaEps);
        double tmp_alpha = adaAlpha / tmp_normaize_alpha;

        double _ft_wordEmb_ij = _ftW[idx][index] * tmp_alpha * regularizationWeight;
        _ftW[idx][index] -= _ft_wordEmb_ij;
        _W[idx][index] -= tmp_alpha * _gradW[idx][index] / _ftW[idx][index];
      }
    }

    if (_bUseB) {
      _gradb = _gradb + _b * regularizationWeight;
      _eg2b = _eg2b + _gradb * _gradb;
      _b = _b - _gradb * adaAlpha / F<nl_sqrt>(_eg2b + adaEps);
    }

    clearGrad();
  }

  void clearGrad() {
    static int outDim;
    outDim = _W.size(0);
    static hash_set<int>::iterator it;

    for (it = _indexers.begin(); it != _indexers.end(); ++it) {
      int index = *it;
      for (int idx = 0; idx < outDim; idx++) {
        _gradW[idx][index] = 0.0;
      }
    }

    _indexers.clear();
    if (_bUseB)
      _gradb = 0.0;
  }
};

#endif /* SRC_SparseUniHidderLayer_H_ */

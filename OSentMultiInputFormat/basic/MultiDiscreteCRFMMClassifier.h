/*
 * MultiDiscreteCRFMMClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_MultiDiscreteCRFMMClassifier_H_
#define SRC_MultiDiscreteCRFMMClassifier_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Feature.h"
#include "N3L.h"
#include "MultiCRFLoss.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//A native neural network classfier using only word embeddings
template<typename xpu>
class MultiDiscreteCRFMMClassifier {
public:
  MultiDiscreteCRFMMClassifier() {
    _dropOut = 0.5;
  }
  ~MultiDiscreteCRFMMClassifier() {

  }

public:
  int _label1Size, _label1_o;
  int _label2Size, _label2_o;
  int _linearfeatSize;

  dtype _dropOut;
  Metric _eval1, _eval2;
  
  MultiCRFLoss<xpu> _crf_layer;
  SparseUniLayer<xpu> _layer_linear1;
  SparseUniLayer<xpu> _layer_linear2;

public:

  inline void init(int label1Size, int label2Size, int linearfeatSize) {
    _label1Size = label1Size;
    _label2Size = label2Size;
    _linearfeatSize = linearfeatSize;

    _crf_layer.initial(_label1Size, _label2Size, 70);
    
    _layer_linear1.initial(_label1Size, _linearfeatSize, false, 40, 2);
    _eval1.reset();

    _layer_linear2.initial(_label2Size, _linearfeatSize, false, 50, 2);
    _eval2.reset();
  }

  inline void release() {
    _layer_linear2.release();
    _layer_linear1.release();
    _crf_layer.release();
  }

  inline dtype process(const vector<Example>& examples, int iter) {
    _eval1.reset();
    _eval2.reset();

    int example_num = examples.size();
    dtype cost = 0.0;
    int offset = 0;

    for (int count = 0; count < example_num; count++) {
      const Example& example = examples[count];

      int seq_size = example.m_features.size();
      vector<Tensor<xpu, 2, dtype> > output1(seq_size), output1Loss(seq_size);
      vector<Tensor<xpu, 2, dtype> > output2(seq_size), output2Loss(seq_size);

      //initialize
      for (int idx = 0; idx < seq_size; idx++) {
        output1[idx] = NewTensor<xpu>(Shape2(1, _label1Size), 0.0);
        output1Loss[idx] = NewTensor<xpu>(Shape2(1, _label1Size), 0.0);
        output2[idx] = NewTensor<xpu>(Shape2(1, _label2Size), 0.0);
        output2Loss[idx] = NewTensor<xpu>(Shape2(1, _label2Size), 0.0);
      }

      //forward propagation
      vector<vector<int> > linear_features(seq_size);
      for (int idx = 0; idx < seq_size; idx++) {
        const Feature& feature = example.m_features[idx];
        srand(iter * example_num + count * seq_size + idx);
        linear_features[idx].clear();
        for (int idy = 0; idy < feature.linear_features.size(); idy++) {
          if (1.0 * rand() / RAND_MAX >= _dropOut) {
            linear_features[idx].push_back(feature.linear_features[idy]);
          }
        }
      }

      _layer_linear1.ComputeForwardScore(linear_features, output1);
      _layer_linear2.ComputeForwardScore(linear_features, output2);
      
      cost += _crf_layer.loss(output1, output2, example.m_label1s, example.m_label2s, output1Loss, output2Loss, _eval1, _eval2, example_num);

      // loss backward propagation
      _layer_linear1.ComputeBackwardLoss(linear_features, output1, output1Loss);
      _layer_linear2.ComputeBackwardLoss(linear_features, output2, output2Loss);

      //release
      for (int idx = 0; idx < seq_size; idx++) {
        FreeSpace(&(output1[idx]));
        FreeSpace(&(output1Loss[idx]));
        FreeSpace(&(output2[idx]));
        FreeSpace(&(output2Loss[idx]));
      }
    }

    if (_eval1.getAccuracy() < 0 || _eval2.getAccuracy() < 0) {
      std::cout << "strange" << std::endl;
    }

    return cost;
  }

  void predict(const vector<Feature>& features, vector<int>& result1s, vector<int>& result2s) {
    int seq_size = features.size();
    vector<Tensor<xpu, 2, dtype> > output1(seq_size);
    vector<Tensor<xpu, 2, dtype> > output2(seq_size);

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      output1[idx] = NewTensor<xpu>(Shape2(1, _label1Size), 0.0);
      output2[idx] = NewTensor<xpu>(Shape2(1, _label2Size), 0.0);
    }

    //forward propagation
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = features[idx];
      _layer_linear1.ComputeForwardScore(feature.linear_features, output1[idx]);
      _layer_linear2.ComputeForwardScore(feature.linear_features, output2[idx]);
    }

    _crf_layer.predict(output1, output2, result1s, result2s);

    //release
    for (int idx = 0; idx < seq_size; idx++) {
      FreeSpace(&(output1[idx]));
      FreeSpace(&(output2[idx]));
    }
  }


  void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
    _layer_linear1.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _layer_linear2.updateAdaGrad(nnRegular, adaAlpha, adaEps);    
    _crf_layer.updateAdaGrad(nnRegular, adaAlpha, adaEps);
  }

  void writeModel();

  void loadModel();


public:
  inline void resetEval() {
    _eval2.reset();
  }

  inline void setDropValue(dtype dropOut) {
    _dropOut = dropOut;
  }

  inline void setLabelO(int label1_o, int label2_o)
  {
    _crf_layer.setLabelO(label1_o, label2_o);
  }

};

#endif /* SRC_MultiDiscreteCRFMMClassifier_H_ */

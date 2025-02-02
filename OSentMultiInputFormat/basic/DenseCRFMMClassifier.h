/*
 * DenseCRFMMClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_DenseCRFMMClassifier_H_
#define SRC_DenseCRFMMClassifier_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Feature.h"
#include "N3L.h"
#include "SingleCRFLoss.h"



using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//A native neural network classfier using only word embeddings
template<typename xpu>
class DenseCRFMMClassifier {
public:
  DenseCRFMMClassifier() {
    _dropOut = 0.5;
  }
  ~DenseCRFMMClassifier() {

  }

public:
  LookupTable<xpu> _words;

  int _wordcontext, _wordwindow;
  int _wordSize;
  int _wordDim;
  int _inputSize;
  int _hiddenSize;
  
  SingleCRFLoss<xpu> _crf_layer;
  UniLayer<xpu> _olayer_linear2;
  UniLayer<xpu> _sharelayer_projected;

  int _label2Size, _label2_o;

  Metric _eval2;

  dtype _dropOut;

public:

  inline void init(const NRMat<dtype>& wordEmb, int wordcontext, int label2Size, int hiddenSize) {
    _wordcontext = wordcontext;
    _wordwindow = 2 * _wordcontext + 1;
    _wordSize = wordEmb.nrows();
    _wordDim = wordEmb.ncols();

    _inputSize = _wordDim * _wordwindow;
    _hiddenSize = hiddenSize;
    _label2Size = label2Size;

    _words.initial(wordEmb);

    _crf_layer.initial(_label2Size, 70);
    _sharelayer_projected.initial(_hiddenSize, _inputSize, true, 30, 0);
    _olayer_linear2.initial(_label2Size, _hiddenSize, false, 40, 2);
    _eval2.reset();
  }

  inline void release() {
    _words.release();
    _olayer_linear2.release();
    _sharelayer_projected.release();

    _crf_layer.release();
  }

  inline dtype process(const vector<Example>& examples, int iter) {
    _eval2.reset();

    int example_num = examples.size();
    dtype cost = 0.0;
    int offset = 0;
    for (int count = 0; count < example_num; count++) {
      const Example& example = examples[count];

      int seq_size = example.m_features.size();

      vector<Tensor<xpu, 2, dtype> > wordprime(seq_size), wordprimeLoss(seq_size), wordprimeMask(seq_size);  
      vector<Tensor<xpu, 2, dtype> > input(seq_size), inputLoss(seq_size);
      vector<Tensor<xpu, 2, dtype> > project(seq_size), projectLoss(seq_size);
      
      vector<Tensor<xpu, 2, dtype> > output2(seq_size), output2Loss(seq_size);

      //initialize
      for (int idx = 0; idx < seq_size; idx++) {
        wordprime[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
        wordprimeLoss[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
        wordprimeMask[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_one);
        input[idx] = NewTensor<xpu>(Shape2(1, _inputSize), d_zero);
        inputLoss[idx] = NewTensor<xpu>(Shape2(1, _inputSize), d_zero);
        project[idx] = NewTensor<xpu>(Shape2(1, _hiddenSize), d_zero);
        projectLoss[idx] = NewTensor<xpu>(Shape2(1, _hiddenSize), d_zero);
        
        output2[idx] = NewTensor<xpu>(Shape2(1, _label2Size), 0.0);
        output2Loss[idx] = NewTensor<xpu>(Shape2(1, _label2Size), 0.0);
      }

      //forward propagation
      //input setting, and linear setting
      for (int idx = 0; idx < seq_size; idx++) {
        const Feature& feature = example.m_features[idx];
        //linear features should not be dropped out

        srand(iter * example_num + count * seq_size + idx);

        const vector<int>& words = feature.words;
        _words.GetEmb(words[0], wordprime[idx]);

        dropoutcol(wordprimeMask[idx], _dropOut);
        wordprime[idx] = wordprime[idx] * wordprimeMask[idx];
      }

      windowlized(wordprime, input, _wordcontext);
      _sharelayer_projected.ComputeForwardScore(input, project);
      _olayer_linear2.ComputeForwardScore(project, output2);

      //compute delta
      cost += _crf_layer.loss(output2, example.m_label2s, output2Loss, _eval2, example_num);

      // loss backward propagation
      _olayer_linear2.ComputeBackwardLoss(project, output2, output2Loss, projectLoss);
      _sharelayer_projected.ComputeBackwardLoss(input, project, projectLoss, inputLoss);
       windowlized_backward(wordprimeLoss, inputLoss, _wordcontext);
              
      // word fine tune
      if (_words.bEmbFineTune()) {
        for (int idx = 0; idx < seq_size; idx++) {
          const Feature& feature = example.m_features[idx];
          const vector<int>& words = feature.words;
          wordprimeLoss[idx] = wordprimeLoss[idx] * wordprimeMask[idx];
          _words.EmbLoss(words[0], wordprimeLoss[idx]);
        }
      }

      //release
      for (int idx = 0; idx < seq_size; idx++) {
        FreeSpace(&(wordprime[idx]));
        FreeSpace(&(wordprimeLoss[idx]));
        FreeSpace(&(wordprimeMask[idx]));      	
        FreeSpace(&(input[idx]));
        FreeSpace(&(inputLoss[idx]));
        FreeSpace(&(project[idx]));
        FreeSpace(&(projectLoss[idx]));
        FreeSpace(&(output2[idx]));
        FreeSpace(&(output2Loss[idx]));
      }
    }

    if (_eval2.getAccuracy() < 0) {
      std::cout << "strange" << std::endl;
    }

    return cost;
  }

  void predict(const vector<Feature>& features, const vector<string>& result1s, vector<int>& result2s) {
    int seq_size = features.size();
    int offset = 0;

    vector<Tensor<xpu, 2, dtype> > wordprime(seq_size);  
    vector<Tensor<xpu, 2, dtype> > input(seq_size);
    vector<Tensor<xpu, 2, dtype> > project(seq_size);
           
    vector<Tensor<xpu, 2, dtype> > output2(seq_size);


    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      wordprime[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
      input[idx] = NewTensor<xpu>(Shape2(1, _inputSize), d_zero);
      project[idx] = NewTensor<xpu>(Shape2(1, _hiddenSize), d_zero);
      
      output2[idx] = NewTensor<xpu>(Shape2(1, _label2Size), 0.0);
    }

    //forward propagation
    //input setting, and linear setting
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = features[idx];
      //linear features should not be dropped out

      const vector<int>& words = feature.words;
      _words.GetEmb(words[0], wordprime[idx]);
    }

    windowlized(wordprime, input, _wordcontext);
    _sharelayer_projected.ComputeForwardScore(input, project);
    _olayer_linear2.ComputeForwardScore(project, output2);

    _crf_layer.predict(output2, result1s, result2s);

    //release
    for (int idx = 0; idx < seq_size; idx++) {
      FreeSpace(&(wordprime[idx]));      	
      FreeSpace(&(input[idx]));
      FreeSpace(&(project[idx]));
      FreeSpace(&(output2[idx]));
    }
  }

  void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
    _olayer_linear2.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _sharelayer_projected.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _crf_layer.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _words.updateAdaGrad(nnRegular, adaAlpha, adaEps);
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

  inline void setWordEmbFineTune(bool b_wordEmb_finetune) {
    _words.setEmbFineTune(b_wordEmb_finetune);
  }

  inline void setLabel2O(int label2_o)
  {
    _crf_layer.setLabelO(label2_o);
  }

};

#endif /* SRC_DenseCRFMMClassifier_H_ */

/*
 * MultiDcombCRFMMClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_MultiDcombCRFMMClassifier_H_
#define SRC_MultiDcombCRFMMClassifier_H_

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
class MultiDcombCRFMMClassifier {
public:
  MultiDcombCRFMMClassifier() {
    _dropOut = 0.5;
  }
  ~MultiDcombCRFMMClassifier() {

  }

public:
  LookupTable<xpu> _words;

  int _wordcontext, _wordwindow;
  int _wordSize;
  int _wordDim;
  int _inputSize;
  int _hiddenSize;
  int _linearfeatSize;
  int _linearHiddenSize;
      
  MultiCRFLoss<xpu> _crf_layer;
  UniLayer<xpu> _olayer_sparselinear1;
  UniLayer<xpu> _olayer_sparselinear2;
  SparseUniLayer<xpu> _sparselayer_linear;
  UniLayer<xpu> _olayer_denselinear1;
  UniLayer<xpu> _olayer_denselinear2;
  UniLayer<xpu> _sharelayer_projected;

  int _label1Size, _label1_o;
  int _label2Size, _label2_o;

  Metric _eval1, _eval2;

  dtype _dropOut;

public:

  inline void init(const NRMat<dtype>& wordEmb, int wordcontext, int label1Size, int label2Size, int hiddenSize,
      int linearHiddenSize, int linearfeatSize) {
    _wordcontext = wordcontext;
    _wordwindow = 2 * _wordcontext + 1;
    _wordSize = wordEmb.nrows();
    _wordDim = wordEmb.ncols();

    _inputSize = _wordDim * _wordwindow;
    _hiddenSize = hiddenSize;
    _linearfeatSize = linearfeatSize;
    _linearHiddenSize = linearHiddenSize;
    _label1Size = label1Size;
    _label2Size = label2Size;

    _words.initial(wordEmb);

    _sparselayer_linear.initial(_linearHiddenSize, _linearfeatSize, false, 10, 0); 
    _olayer_sparselinear1.initial(_label1Size, _linearHiddenSize, false, 20, 2); 
    _olayer_sparselinear2.initial(_label2Size, _linearHiddenSize, false, 30, 2);  
    _sharelayer_projected.initial(_hiddenSize, _inputSize, true, 40, 0);
    _olayer_denselinear1.initial(_label1Size, _hiddenSize, false, 50, 2);    
    _olayer_denselinear2.initial(_label2Size, _hiddenSize, false, 60, 2);
    _crf_layer.initial(_label1Size, _label2Size, 70);
    _eval1.reset();
    _eval2.reset();
  }

  inline void release() {
    _words.release();
    _olayer_denselinear1.release();
    _olayer_sparselinear1.release();
    _olayer_denselinear2.release();
    _olayer_sparselinear2.release();
    _sharelayer_projected.release();
    _sparselayer_linear.release();

    _crf_layer.release();
  }

  inline dtype process(const vector<Example>& examples, int iter) {
    _eval1.reset(); _eval2.reset();

    int example_num = examples.size();
    dtype cost = 0.0;
    int offset = 0;
    for (int count = 0; count < example_num; count++) {
      const Example& example = examples[count];

      int seq_size = example.m_features.size();

      vector<Tensor<xpu, 2, dtype> > wordprime(seq_size), wordprimeLoss(seq_size), wordprimeMask(seq_size);  
      vector<Tensor<xpu, 2, dtype> > input(seq_size), inputLoss(seq_size);
      vector<Tensor<xpu, 2, dtype> > project(seq_size), projectLoss(seq_size);
      
      vector<vector<int> > linear_features(seq_size);
      vector<Tensor<xpu, 2, dtype> > sparse(seq_size), sparseLoss(seq_size);
      
      vector<Tensor<xpu, 2, dtype> > denseout1(seq_size), sparseout1(seq_size);
      vector<Tensor<xpu, 2, dtype> > denseout2(seq_size), sparseout2(seq_size);      
            
      vector<Tensor<xpu, 2, dtype> > output1(seq_size), output1Loss(seq_size);
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
        
        sparse[idx] = NewTensor<xpu>(Shape2(1, _linearHiddenSize), 0.0);
        sparseLoss[idx] = NewTensor<xpu>(Shape2(1, _linearHiddenSize), 0.0);

        sparseout1[idx] = NewTensor<xpu>(Shape2(1, _label1Size), 0.0);
        denseout1[idx] = NewTensor<xpu>(Shape2(1, _label1Size), 0.0);
        output1[idx] = NewTensor<xpu>(Shape2(1, _label1Size), 0.0);
        output1Loss[idx] = NewTensor<xpu>(Shape2(1, _label1Size), 0.0);

        sparseout2[idx] = NewTensor<xpu>(Shape2(1, _label2Size), 0.0);
        denseout2[idx] = NewTensor<xpu>(Shape2(1, _label2Size), 0.0);
        output2[idx] = NewTensor<xpu>(Shape2(1, _label2Size), 0.0);
        output2Loss[idx] = NewTensor<xpu>(Shape2(1, _label2Size), 0.0);
      }

      //forward propagation
      //input setting, and linear setting
      for (int idx = 0; idx < seq_size; idx++) {
        const Feature& feature = example.m_features[idx];
        //linear features should not be dropped out

        srand(iter * example_num + count * seq_size + idx);

        linear_features[idx].clear();
        for (int idy = 0; idy < feature.linear_features.size(); idy++) {
          if (1.0 * rand() / RAND_MAX >= _dropOut) {
            linear_features[idx].push_back(feature.linear_features[idy]);
          }
        }

        const vector<int>& words = feature.words;
        _words.GetEmb(words[0], wordprime[idx]);

        dropoutcol(wordprimeMask[idx], _dropOut);
        wordprime[idx] = wordprime[idx] * wordprimeMask[idx];
      }

      windowlized(wordprime, input, _wordcontext);  
      
      _sharelayer_projected.ComputeForwardScore(input, project);
      _sparselayer_linear.ComputeForwardScore(linear_features, sparse); 
      
      _olayer_denselinear1.ComputeForwardScore(project, denseout1);
      _olayer_sparselinear1.ComputeForwardScore(sparse, sparseout1);            

      _olayer_denselinear2.ComputeForwardScore(project, denseout2);
      _olayer_sparselinear2.ComputeForwardScore(sparse, sparseout2);
              
      for (int idx = 0; idx < seq_size; idx++) {
        output1[idx] = denseout1[idx] + sparseout1[idx];
        output2[idx] = denseout2[idx] + sparseout2[idx];
      }

      cost += _crf_layer.loss(output1, output2, example.m_label1s, example.m_label2s, output1Loss, output2Loss, _eval1, _eval2, example_num);

      // loss backward propagation
      _olayer_sparselinear1.ComputeBackwardLoss(sparse, sparseout1, output1Loss, sparseLoss);
      _olayer_denselinear1.ComputeBackwardLoss(project, denseout1, output1Loss, projectLoss);

      _olayer_sparselinear2.ComputeBackwardLoss(sparse, sparseout2, output2Loss, sparseLoss);
      _olayer_denselinear2.ComputeBackwardLoss(project, denseout2, output2Loss, projectLoss);

      _sparselayer_linear.ComputeBackwardLoss(linear_features, sparse, sparseLoss);
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
        FreeSpace(&(sparse[idx]));
        FreeSpace(&(sparseLoss[idx]));
        FreeSpace(&(sparseout1[idx]));
        FreeSpace(&(denseout1[idx]));
        FreeSpace(&(output1[idx]));
        FreeSpace(&(output1Loss[idx]));
        FreeSpace(&(sparseout2[idx]));
        FreeSpace(&(denseout2[idx]));
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
    int offset = 0;
    vector<Tensor<xpu, 2, dtype> > wordprime(seq_size);  
    vector<Tensor<xpu, 2, dtype> > input(seq_size);
    vector<Tensor<xpu, 2, dtype> > project(seq_size);
    
    vector<Tensor<xpu, 2, dtype> > sparse(seq_size);
    
    vector<Tensor<xpu, 2, dtype> > denseout1(seq_size), sparseout1(seq_size);
    vector<Tensor<xpu, 2, dtype> > denseout2(seq_size), sparseout2(seq_size);      
          
    vector<Tensor<xpu, 2, dtype> > output1(seq_size);
    vector<Tensor<xpu, 2, dtype> > output2(seq_size);

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {        
      wordprime[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
      input[idx] = NewTensor<xpu>(Shape2(1, _inputSize), d_zero);
      project[idx] = NewTensor<xpu>(Shape2(1, _hiddenSize), d_zero);
      
      sparse[idx] = NewTensor<xpu>(Shape2(1, _linearHiddenSize), 0.0);

      sparseout1[idx] = NewTensor<xpu>(Shape2(1, _label1Size), 0.0);
      denseout1[idx] = NewTensor<xpu>(Shape2(1, _label1Size), 0.0);
      output1[idx] = NewTensor<xpu>(Shape2(1, _label1Size), 0.0);

      sparseout2[idx] = NewTensor<xpu>(Shape2(1, _label2Size), 0.0);
      denseout2[idx] = NewTensor<xpu>(Shape2(1, _label2Size), 0.0);
      output2[idx] = NewTensor<xpu>(Shape2(1, _label2Size), 0.0);
    }

    //forward propagation
    //input setting, and linear setting
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = features[idx];
      //linear features should not be dropped out
      _sparselayer_linear.ComputeForwardScore(feature.linear_features, sparse[idx]); 

      const vector<int>& words = feature.words;
      _words.GetEmb(words[0], wordprime[idx]);
    }

    windowlized(wordprime, input, _wordcontext);  
    _sharelayer_projected.ComputeForwardScore(input, project);    
    
    _olayer_denselinear1.ComputeForwardScore(project, denseout1);
    _olayer_sparselinear1.ComputeForwardScore(sparse, sparseout1);            

    _olayer_denselinear2.ComputeForwardScore(project, denseout2);
    _olayer_sparselinear2.ComputeForwardScore(sparse, sparseout2);
            
    for (int idx = 0; idx < seq_size; idx++) {
      output1[idx] = denseout1[idx] + sparseout1[idx];
      output2[idx] = denseout2[idx] + sparseout2[idx];
    }
    
    _crf_layer.predict(output1, output2, result1s, result2s);

    //release
    for (int idx = 0; idx < seq_size; idx++) {
      FreeSpace(&(wordprime[idx]));
      FreeSpace(&(input[idx]));
      FreeSpace(&(project[idx]));
      FreeSpace(&(sparse[idx]));
      FreeSpace(&(sparseout1[idx]));
      FreeSpace(&(denseout1[idx]));
      FreeSpace(&(output1[idx]));
      FreeSpace(&(sparseout2[idx]));
      FreeSpace(&(denseout2[idx]));
      FreeSpace(&(output2[idx]));
    }
  }


  void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
    _olayer_denselinear1.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_sparselinear1.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_denselinear2.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_sparselinear2.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _sparselayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
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

  inline void setLabelO(int label1_o, int label2_o)
  {
    _crf_layer.setLabelO(label1_o, label2_o);
  }

};

#endif /* SRC_MultiDcombCRFMMClassifier_H_ */

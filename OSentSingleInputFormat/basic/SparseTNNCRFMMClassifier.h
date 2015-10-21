/*
 * SparseTNNCRFMMClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_SparseTNNCRFMMClassifier_H_
#define SRC_SparseTNNCRFMMClassifier_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Feature.h"
#include "Metric.h"
#include "N3L.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//A native neural network classfier using only word embeddings
template<typename xpu>
class SparseTNNCRFMMClassifier {
public:
  SparseTNNCRFMMClassifier() {
    _dropOut = 0.5;
  }
  ~SparseTNNCRFMMClassifier() {

  }

public:
  LookupTable<xpu> _words;

  int _wordcontext, _wordwindow;
  int _wordSize;
  int _wordDim;

  int _hiddensize;
  int _inputsize, _token_representation_size;
  UniLayer<xpu> _olayer_linear;
  UniLayer<xpu> _tanh_project;
  MMCRFLoss<xpu> _crf_layer;

  int _labelSize;

  Metric _eval;

  dtype _dropOut;
// add sparse
  SparseUniLayer<xpu> _sparselayer_hidden;
  UniLayer<xpu> _sparselayer_out;
  int _linearfeatSize;
  int _linearHiddenSize;

public:

  inline void init(const NRMat<dtype>& wordEmb, int wordcontext, int labelSize,
      int hiddensize, int linearfeatSize, int linearHiddenSize) {
    _wordcontext = wordcontext;
    _wordwindow = 2 * _wordcontext + 1;
    _wordSize = wordEmb.nrows();
    _wordDim = wordEmb.ncols();

    _labelSize = labelSize;
    _hiddensize = hiddensize;
    _token_representation_size = _wordDim;
    _inputsize = _wordwindow * _token_representation_size;

    _words.initial(wordEmb);

    _tanh_project.initial(_hiddensize, _inputsize, true, 30, 0);
    _olayer_linear.initial(_labelSize, _hiddensize, false, 60, 2);
// add sparse and crf
    _linearfeatSize = linearfeatSize;
    _linearHiddenSize = linearHiddenSize;
    _sparselayer_hidden.initial(_linearHiddenSize, _linearfeatSize, true, 1000, 0);
    _sparselayer_out.initial(_labelSize, _linearHiddenSize, false, 1100, 2);
    _crf_layer.initial(_labelSize, 70);
  }

  inline void release() {
    _words.release();
    _olayer_linear.release();
    _tanh_project.release();
// add sparse and crf
    _sparselayer_hidden.release();
    _sparselayer_out.release();
    _crf_layer.release();

  }

  inline dtype process(const vector<Example>& examples, int iter) {
    _eval.reset();

    int example_num = examples.size();
    dtype cost = 0.0;
    int offset = 0;
    for (int count = 0; count < example_num; count++) {
      const Example& example = examples[count];

      int seq_size = example.m_features.size();
// add sparse
      vector<vector<int> > linear_features(seq_size);
      vector<Tensor<xpu, 2, dtype> > denseout(seq_size), denseoutLoss(seq_size);
      vector<Tensor<xpu, 2, dtype> > sparseproject(seq_size), sparseprojectLoss(seq_size);
      vector<Tensor<xpu, 2, dtype> > sparseout(seq_size), sparseoutLoss(seq_size);

      vector<Tensor<xpu, 2, dtype> > input(seq_size), inputLoss(seq_size);
      vector<Tensor<xpu, 2, dtype> > project(seq_size), projectLoss(seq_size);
      vector<Tensor<xpu, 2, dtype> > output(seq_size), outputLoss(seq_size);

      vector<Tensor<xpu, 2, dtype> > wordprime(seq_size), wordprimeLoss(seq_size), wordprimeMask(seq_size);
      vector<Tensor<xpu, 2, dtype> > wordrepresent(seq_size), wordrepresentLoss(seq_size);

      //initialize
      for (int idx = 0; idx < seq_size; idx++) {
        const Feature& feature = example.m_features[idx];

        wordprime[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
        wordprimeLoss[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
        wordprimeMask[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_one);
        wordrepresent[idx] = NewTensor<xpu>(Shape2(1, _token_representation_size), d_zero);
        wordrepresentLoss[idx] = NewTensor<xpu>(Shape2(1, _token_representation_size), d_zero);
        input[idx] = NewTensor<xpu>(Shape2(1, _inputsize), d_zero);
        inputLoss[idx] = NewTensor<xpu>(Shape2(1, _inputsize), d_zero);
        project[idx] = NewTensor<xpu>(Shape2(1, _hiddensize), d_zero);
        projectLoss[idx] = NewTensor<xpu>(Shape2(1, _hiddensize), d_zero);
        output[idx] = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
        outputLoss[idx] = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
        // add sparse
        sparseproject[idx] = NewTensor<xpu>(Shape2(1, _linearHiddenSize), d_zero);
        sparseprojectLoss[idx] = NewTensor<xpu>(Shape2(1, _linearHiddenSize), d_zero);
        sparseout[idx] = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
        sparseoutLoss[idx] = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
        denseout[idx] = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
        denseoutLoss[idx] = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
      }

      //forward propagation
      //input setting, and linear setting
      for (int idx = 0; idx < seq_size; idx++) {
        const Feature& feature = example.m_features[idx];
        //linear features should not be dropped out

        srand(iter * example_num + count * seq_size + idx);

// add sparse features
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

      for (int idx = 0; idx < seq_size; idx++) {
        wordrepresent[idx] += wordprime[idx];
      }

      windowlized(wordrepresent, input, _wordcontext);
      _tanh_project.ComputeForwardScore(input, project);
      _olayer_linear.ComputeForwardScore(project, denseout);

// add sparse
      _sparselayer_hidden.ComputeForwardScore(linear_features, sparseproject);
      _sparselayer_out.ComputeForwardScore(sparseproject, sparseout);
      for (int idx = 0; idx < seq_size; idx++) {
        output[idx] = denseout[idx] + sparseout[idx];
      }

      // get delta for each output
      cost += _crf_layer.loss(output, example.m_labels, outputLoss, _eval, example_num);

      // loss backward propagation
      // output
      _olayer_linear.ComputeBackwardLoss(project, denseout, outputLoss, projectLoss);
      // word combination
      _tanh_project.ComputeBackwardLoss(input, project, projectLoss, inputLoss);
           
      _sparselayer_out.ComputeBackwardLoss(sparseproject, sparseout, outputLoss, sparseprojectLoss);
      _sparselayer_hidden.ComputeBackwardLoss(linear_features, sparseproject, sparseprojectLoss);
      // word context
      windowlized_backward(wordrepresentLoss, inputLoss, _wordcontext);
      // decompose loss
      for (int idx = 0; idx < seq_size; idx++) {
        wordprimeLoss[idx] += wordrepresentLoss[idx];
      }
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
        FreeSpace(&(wordrepresent[idx]));
        FreeSpace(&(wordrepresentLoss[idx]));

        FreeSpace(&(input[idx]));
        FreeSpace(&(inputLoss[idx]));
        FreeSpace(&(project[idx]));
        FreeSpace(&(projectLoss[idx]));
        FreeSpace(&(output[idx]));
        FreeSpace(&(outputLoss[idx]));

        // add sparse
        FreeSpace(&(sparseproject[idx]));
        FreeSpace(&(sparseprojectLoss[idx]));
        FreeSpace(&(sparseout[idx]));
        FreeSpace(&(sparseoutLoss[idx]));
        FreeSpace(&(denseout[idx]));
        FreeSpace(&(denseoutLoss[idx]));
      }
    }

    if (_eval.getAccuracy() < 0) {
      std::cout << "strange" << std::endl;
    }

    return cost;
  }

  void predict(const vector<Feature>& features, vector<int>& results) {
    int seq_size = features.size();
    int offset = 0;

// add sparse
    vector<Tensor<xpu, 2, dtype> > denseout(seq_size);
    vector<Tensor<xpu, 2, dtype> > sparseproject(seq_size);
    vector<Tensor<xpu, 2, dtype> > sparseout(seq_size);

    vector<Tensor<xpu, 2, dtype> > input(seq_size);
    vector<Tensor<xpu, 2, dtype> > project(seq_size);
    vector<Tensor<xpu, 2, dtype> > output(seq_size);

    vector<Tensor<xpu, 2, dtype> > wordprime(seq_size);
    vector<Tensor<xpu, 2, dtype> > wordrepresent(seq_size);

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = features[idx];

      wordprime[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
      wordrepresent[idx] = NewTensor<xpu>(Shape2(1, _token_representation_size), d_zero);
      input[idx] = NewTensor<xpu>(Shape2(1, _inputsize), d_zero);
      project[idx] = NewTensor<xpu>(Shape2(1, _hiddensize), d_zero);
      output[idx] = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);

      // add sparse
      sparseproject[idx] = NewTensor<xpu>(Shape2(1, _linearHiddenSize), d_zero);
      sparseout[idx] = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
      denseout[idx] = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
    }

    //forward propagation
    //input setting, and linear setting
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = features[idx];
      //linear features should not be dropped out
      // add sparse features
      _sparselayer_hidden.ComputeForwardScore(feature.linear_features, sparseproject[idx]);

      const vector<int>& words = feature.words;
      _words.GetEmb(words[0], wordprime[idx]);
    }

    for (int idx = 0; idx < seq_size; idx++) {
      wordrepresent[idx] += wordprime[idx];
    }

    windowlized(wordrepresent, input, _wordcontext);
    _tanh_project.ComputeForwardScore(input, project);
    _olayer_linear.ComputeForwardScore(project, denseout);
    _sparselayer_out.ComputeForwardScore(sparseproject, sparseout);

    for (int idx = 0; idx < seq_size; idx++) {
      output[idx] = denseout[idx] + sparseout[idx];
    }

    // decode algorithm
    _crf_layer.predict(output, results);

    //release
    for (int idx = 0; idx < seq_size; idx++) {
      FreeSpace(&(wordprime[idx]));
      FreeSpace(&(wordrepresent[idx]));
      FreeSpace(&(input[idx]));
      FreeSpace(&(project[idx]));
      FreeSpace(&(output[idx]));
      // add sparse
      FreeSpace(&(sparseproject[idx]));
      FreeSpace(&(sparseout[idx]));
      FreeSpace(&(denseout[idx]));
    }
  }

  dtype computeScore(const Example& example) {
    int seq_size = example.m_features.size();
    int offset = 0;

    vector<Tensor<xpu, 2, dtype> > denseout(seq_size);
    vector<Tensor<xpu, 2, dtype> > sparseproject(seq_size);
    vector<Tensor<xpu, 2, dtype> > sparseout(seq_size);

    vector<Tensor<xpu, 2, dtype> > input(seq_size);
    vector<Tensor<xpu, 2, dtype> > project(seq_size);
    vector<Tensor<xpu, 2, dtype> > output(seq_size);

    vector<Tensor<xpu, 2, dtype> > wordprime(seq_size);
    vector<Tensor<xpu, 2, dtype> > wordrepresent(seq_size);

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = example.m_features[idx];

      wordprime[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
      wordrepresent[idx] = NewTensor<xpu>(Shape2(1, _token_representation_size), d_zero);

      input[idx] = NewTensor<xpu>(Shape2(1, _inputsize), d_zero);
      project[idx] = NewTensor<xpu>(Shape2(1, _hiddensize), d_zero);
      output[idx] = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
      // add sparse
      sparseproject[idx] = NewTensor<xpu>(Shape2(1, _linearHiddenSize), d_zero);
      sparseout[idx] = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
      denseout[idx] = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
    }

    //forward propagation
    //input setting, and linear setting
    for (int idx = 0; idx < seq_size; idx++) {
    	const Feature& feature = example.m_features[idx];
      //linear features should not be dropped out
      // add sparse features
      _sparselayer_hidden.ComputeForwardScore(feature.linear_features, sparseproject[idx]);

      const vector<int>& words = feature.words;
      _words.GetEmb(words[0], wordprime[idx]);
    }

    for (int idx = 0; idx < seq_size; idx++) {
      wordrepresent[idx] += wordprime[idx];
    }

    windowlized(wordrepresent, input, _wordcontext);
    _tanh_project.ComputeForwardScore(input, project);
    _olayer_linear.ComputeForwardScore(project, denseout);
    _sparselayer_out.ComputeForwardScore(sparseproject, sparseout);

    for (int idx = 0; idx < seq_size; idx++) {
      output[idx] = denseout[idx] + sparseout[idx];
    }

    // get delta for each output
    dtype cost = _crf_layer.cost(output, example.m_labels);

    //release
    for (int idx = 0; idx < seq_size; idx++) {
      FreeSpace(&(wordprime[idx]));
      FreeSpace(&(wordrepresent[idx]));
      FreeSpace(&(input[idx]));
      FreeSpace(&(project[idx]));
      FreeSpace(&(output[idx]));
      // add sparse
      FreeSpace(&(sparseproject[idx]));
      FreeSpace(&(sparseout[idx]));
      FreeSpace(&(denseout[idx]));
    }
    return cost;
  }

  void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
    _tanh_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    // add sparse and crf
    _sparselayer_hidden.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _sparselayer_out.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _crf_layer.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _words.updateAdaGrad(nnRegular, adaAlpha, adaEps);
  }

  void writeModel();

  void loadModel();

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter) {
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols;
    idRows.clear();
    idCols.clear();
    for (int i = 0; i < Wd.size(0); ++i)
      idRows.push_back(i);
    for (int idx = 0; idx < Wd.size(1); idx++)
      idCols.push_back(idx);

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());

    int check_i = idRows[0], check_j = idCols[0];

    dtype orginValue = Wd[check_i][check_j];

    Wd[check_i][check_j] = orginValue + 0.1;
    dtype lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossAdd += computeScore(oneExam);
    }

    Wd[check_i][check_j] = orginValue - 0.1;
    dtype lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossPlus += computeScore(oneExam);
    }

    dtype mockGrad = (lossAdd - lossPlus) / 0.2;
    mockGrad = mockGrad / examples.size();
    dtype computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;
  }

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter,
      const hash_set<int>& indexes, bool bRow = true) {
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols;
    idRows.clear();
    idCols.clear();
    static hash_set<int>::iterator it;
    if (bRow) {
      for (it = indexes.begin(); it != indexes.end(); ++it)
        idRows.push_back(*it);
      for (int idx = 0; idx < Wd.size(1); idx++)
        idCols.push_back(idx);
    } else {
      for (it = indexes.begin(); it != indexes.end(); ++it)
        idCols.push_back(*it);
      for (int idx = 0; idx < Wd.size(0); idx++)
        idRows.push_back(idx);
    }

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());

    int check_i = idRows[0], check_j = idCols[0];

    dtype orginValue = Wd[check_i][check_j];

    Wd[check_i][check_j] = orginValue + 0.1;
    dtype lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossAdd += computeScore(oneExam);
    }

    Wd[check_i][check_j] = orginValue - 0.1;
    dtype lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossPlus += computeScore(oneExam);
    }

    dtype mockGrad = (lossAdd - lossPlus) / 0.2;
    mockGrad = mockGrad / examples.size();
    dtype computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;

  }

  void checkgrads(const vector<Example>& examples, int iter) {
    checkgrad(examples, _olayer_linear._W, _olayer_linear._gradW, "_olayer_linear._W", iter);

    checkgrad(examples, _tanh_project._W, _tanh_project._gradW, "_tanh_project._W", iter);
    checkgrad(examples, _tanh_project._b, _tanh_project._gradb, "_tanh_project._b", iter);

    checkgrad(examples, _words._E, _words._gradE, "_words._E", iter, _words._indexers);
    // add sparse and crf
    checkgrad(examples, _sparselayer_hidden._W, _sparselayer_hidden._gradW, "_sparselayer_hidden._W", iter, _sparselayer_hidden._indexers, false);
    checkgrad(examples, _sparselayer_hidden._b, _sparselayer_hidden._gradb, "_sparselayer_hidden._b", iter);
    checkgrad(examples, _sparselayer_out._W, _sparselayer_out._gradW, "_sparselayer_out._W", iter);
    checkgrad(examples, _crf_layer._tagBigram, _crf_layer._grad_tagBigram, "_crf_layer._tagBigram", iter);
  }

public:
  inline void resetEval() {
    _eval.reset();
  }

  inline void setDropValue(dtype dropOut) {
    _dropOut = dropOut;
  }

  inline void setWordEmbFinetune(bool b_wordEmb_finetune) {
    _words.setEmbFineTune(b_wordEmb_finetune);
  }

};

#endif /* SRC_SparseTNNCRFMMClassifier_H_ */

/*
 * Sparse2TNNCRFMMClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_Sparse2TNNCRFMMClassifier_H_
#define SRC_Sparse2TNNCRFMMClassifier_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Feature.h"
#include "Metric.h"
#include "NRMat.h"
#include "MyLib.h"
#include "tensor.h"

#include "SparseUniHidderLayer.h"
#include "UniHidderLayer.h"
#include "BiHidderLayer.h"
#include "Utiltensor.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//A native neural network classfier using only word embeddings
template<typename xpu>
class Sparse2TNNCRFMMClassifier {
public:
  Sparse2TNNCRFMMClassifier() {
    _b_wordEmb_finetune = false;
    _dropOut = 0.5;
  }
  ~Sparse2TNNCRFMMClassifier() {

  }

public:
  Tensor<xpu, 2, double> _wordEmb;
  Tensor<xpu, 2, double> _grad_wordEmb;
  Tensor<xpu, 2, double> _eg2_wordEmb;
  Tensor<xpu, 2, double> _ft_wordEmb;
  hash_set<int> _indexers;

  int _wordcontext;
  int _wordSize;
  int _wordDim;
  bool _b_wordEmb_finetune;

  Tensor<xpu, 2, double> _tagBigram;
  Tensor<xpu, 2, double> _grad_tagBigram;
  Tensor<xpu, 2, double> _eg2_tagBigram;

  int _hiddensize;
  int _inputsize;
  UniHidderLayer<xpu> _olayer_linear;
  UniHidderLayer<xpu> _tanh_project;

  int _labelSize;

  Metric _eval;

  double _dropOut;

  SparseUniHidderLayer<xpu> _sparselayer_linear;
  UniHidderLayer<xpu> _olayer_sparselinear;
  int _linearfeatSize;
  int _linearHiddenSize;

public:

  inline void init(const NRMat<double>& wordEmb, int wordcontext, int labelSize, int hiddensize, int linearHiddenSize, int linearfeatSize) {
    _wordcontext = wordcontext;
    _wordSize = wordEmb.nrows();
    _wordDim = wordEmb.ncols();

    _labelSize = labelSize;
    _hiddensize = hiddensize;
    _inputsize = _wordcontext * _wordDim;

    _wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 0.0);
    _grad_wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 0.0);
    _eg2_wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 0.0);
    _ft_wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 1.0);
    assign(_wordEmb, wordEmb);
    for (int idx = 0; idx < _wordSize; idx++) {
      norm2one(_wordEmb, idx);
    }

    _tagBigram = NewTensor<xpu>(Shape2(_labelSize, _labelSize), 0.0);
    _grad_tagBigram = NewTensor<xpu>(Shape2(_labelSize, _labelSize), 0.0);
    _eg2_tagBigram = NewTensor<xpu>(Shape2(_labelSize, _labelSize), 0.0);

    Random<xpu, double> rnd(100);
    rnd.SampleUniform(&_tagBigram, -0.1, 0.1);

    _tanh_project.initial(_hiddensize, _inputsize, true, 3, 0);
    _olayer_linear.initial(_labelSize, _hiddensize, true, 4, 2);

    _linearfeatSize = linearfeatSize;
    _linearHiddenSize = linearHiddenSize;
    _sparselayer_linear.initial(_linearHiddenSize, _linearfeatSize, false, 5, 0);
    _olayer_sparselinear.initial(_labelSize, _linearHiddenSize, true, 6, 2);

  }

  inline void release() {
    FreeSpace(&_wordEmb);
    FreeSpace(&_grad_wordEmb);
    FreeSpace(&_eg2_wordEmb);
    FreeSpace(&_ft_wordEmb);
    _olayer_linear.release();
    _sparselayer_linear.release();
    _olayer_sparselinear.release();
    _tanh_project.release();

    FreeSpace(&_tagBigram);
    FreeSpace(&_grad_tagBigram);
    FreeSpace(&_eg2_tagBigram);

  }

  inline double process(const vector<Example>& examples, int iter) {
    _eval.reset();
    _indexers.clear();

    int example_num = examples.size();
    double cost = 0.0;
    int offset = 0;
    for (int count = 0; count < example_num; count++) {
      const Example& example = examples[count];

      int seq_size = example.m_features.size();
      vector<vector<int> > linear_features(seq_size);

      Tensor<xpu, 2, double> input[seq_size], inputLoss[seq_size];
      Tensor<xpu, 2, double> project[seq_size], projectLoss[seq_size];
      Tensor<xpu, 2, double> sparse[seq_size], sparseLoss[seq_size];
      Tensor<xpu, 2, double> denseout[seq_size], denseoutLoss[seq_size];
      Tensor<xpu, 2, double> sparseout[seq_size], sparseoutLoss[seq_size];
      Tensor<xpu, 2, double> output[seq_size], outputLoss[seq_size];

      Tensor<xpu, 2, double> inputcontext[seq_size][_wordcontext];
      Tensor<xpu, 2, double> inputcontextMask[seq_size][_wordcontext];
      Tensor<xpu, 2, double> inputcontextLoss[seq_size][_wordcontext];

      //initialize
      for (int idx = 0; idx < seq_size; idx++) {
        for (int idy = 0; idy < _wordcontext; idy++) {
          inputcontext[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
          inputcontextMask[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 1.0);
          inputcontextLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
        }
        input[idx] = NewTensor<xpu>(Shape2(1, _inputsize), 0.0);
        inputLoss[idx] = NewTensor<xpu>(Shape2(1, _inputsize), 0.0);
        project[idx] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
        projectLoss[idx] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
        sparse[idx] = NewTensor<xpu>(Shape2(1, _linearHiddenSize), 0.0);
        sparseLoss[idx] = NewTensor<xpu>(Shape2(1, _linearHiddenSize), 0.0);
        sparseout[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
        sparseoutLoss[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
        denseout[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
        denseoutLoss[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
        output[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
        outputLoss[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
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

        const vector<int>& ngram_words = feature.ngram_words;
        assert(ngram_words.size() == _wordcontext);

        for (int i = 0; i < _wordcontext; i++) {
          offset = ngram_words[i];
          inputcontext[idx][i][0] = _wordEmb[offset] / _ft_wordEmb[offset];
          for (int j = 0; j < _wordDim; j++) {
            if (1.0 * rand() / RAND_MAX >= _dropOut) {
              inputcontextMask[idx][i][0][j] = 1.0;
            } else {
              inputcontextMask[idx][i][0][j] = 0.0;
            }
          }
          inputcontext[idx][i] = inputcontext[idx][i] * inputcontextMask[idx][i];
        }

        offset = 0;
        for (int i = 0; i < _wordcontext; i++) {
          for (int j = 0; j < _wordDim; j++) {
            input[idx][0][offset] = inputcontext[idx][i][0][j];
            offset++;
          }
        }
      }

      for (int idx = 0; idx < seq_size; idx++) {
        _tanh_project.ComputeForwardScore(input[idx], project[idx]);
        _olayer_linear.ComputeForwardScore(project[idx], denseout[idx]);
        _sparselayer_linear.ComputeForwardScore(linear_features[idx], sparse[idx]);
        _olayer_sparselinear.ComputeForwardScore(sparse[idx], sparseout[idx]);
        output[idx] = denseout[idx] + sparseout[idx];
      }

      // get delta for each output
      // viterbi algorithm
      NRVec<int> goldlabels(seq_size);
      goldlabels = -1;
      for (int idx = 0; idx < seq_size; idx++) {
        for (int i = 0; i < _labelSize; ++i) {
          if (example.m_labels[idx][i] == 1) {
            goldlabels[idx] = i;
          }
        }
      }

      NRMat<double> maxscores(seq_size, _labelSize);
      NRMat<int> maxlastlabels(seq_size, _labelSize);
      double goldScore = 0.0;
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx == 0)
          goldScore = output[idx][0][goldlabels[idx]];
        else
          goldScore += output[idx][0][goldlabels[idx]] + _tagBigram[goldlabels[idx - 1]][goldlabels[idx]];
        double delta = 0.2;
        for (int i = 0; i < _labelSize; ++i) {
          // can be changed with probabilities in future work
          if (idx == 0) {
            maxscores[idx][i] = output[idx][0][i];
            if (goldlabels[idx] != i)
              maxscores[idx][i] = maxscores[idx][i] + delta;
            maxlastlabels[idx][i] = -1;
          } else {
            int maxlastlabel = 0;
            double maxscore = _tagBigram[0][i] + output[idx][0][i] + maxscores[idx - 1][0];
            for (int j = 1; j < _labelSize; ++j) {
              double curscore = _tagBigram[j][i] + output[idx][0][i] + maxscores[idx - 1][j];
              if (curscore > maxscore) {
                maxlastlabel = j;
                maxscore = curscore;
              }
            }
            maxscores[idx][i] = maxscore;
            if (goldlabels[idx] != i)
              maxscores[idx][i] = maxscores[idx][i] + delta;
            maxlastlabels[idx][i] = maxlastlabel;

          }
        }
      }

      NRVec<int> optLabels(seq_size);
      optLabels = 0;
      double maxScore = maxscores[seq_size - 1][0];
      for (int i = 1; i < _labelSize; ++i) {
        if (maxscores[seq_size - 1][i] > maxScore) {
          maxScore = maxscores[seq_size - 1][i];
          optLabels[seq_size - 1] = i;
        }
      }

      for (int idx = seq_size - 2; idx >= 0; idx--) {
        optLabels[idx] = maxlastlabels[idx + 1][optLabels[idx + 1]];
      }

      bool bcorrect = true;
      for (int idx = 0; idx < seq_size; idx++) {
        if (goldlabels[idx] == -1)
          continue;
        _eval.overall_label_count++;
        if (optLabels[idx] == goldlabels[idx]) {
          _eval.correct_label_count++;
        } else {
          bcorrect = false;
        }
      }

      double curcost = bcorrect ? 0.0 : 1.0;
      //double curcost = maxScore - goldScore;
      curcost = curcost / example_num;

      for (int idx = 0; idx < seq_size; idx++) {
        if (goldlabels[idx] == -1)
          continue;
        if (optLabels[idx] != goldlabels[idx]) {
          outputLoss[idx][0][optLabels[idx]] = curcost;
          outputLoss[idx][0][goldlabels[idx]] = -curcost;
          cost += curcost;
        }
        if (idx > 0 && goldlabels[idx - 1] >= 0) {
          _grad_tagBigram[optLabels[idx - 1]][optLabels[idx]] += curcost;
          _grad_tagBigram[goldlabels[idx - 1]][goldlabels[idx]] -= curcost;
        }
      }

      // loss backward propagation
      for (int idx = 0; idx < seq_size; idx++) {
        _olayer_linear.ComputeBackwardLoss(project[idx], denseout[idx], outputLoss[idx], projectLoss[idx]);
        _olayer_sparselinear.ComputeBackwardLoss(sparse[idx], sparseout[idx], outputLoss[idx], sparseLoss[idx]);
        _sparselayer_linear.ComputeBackwardLoss(linear_features[idx], sparse[idx], sparseLoss[idx]);
        _tanh_project.ComputeBackwardLoss(input[idx], project[idx], projectLoss[idx], inputLoss[idx]);

        offset = 0;
        for (int i = 0; i < _wordcontext; i++) {
          for (int j = 0; j < _wordDim; j++) {
            inputcontextLoss[idx][i][0][j] = inputLoss[idx][0][offset];
            offset++;
          }
        }

        if (_b_wordEmb_finetune) {
          const Feature& feature = example.m_features[idx];
          const vector<int>& ngram_words = feature.ngram_words;
          assert(ngram_words.size() == _wordcontext);
          for (int i = 0; i < _wordcontext; i++) {
            offset = ngram_words[i];
            inputcontextLoss[idx][i] = inputcontextLoss[idx][i] * inputcontextMask[idx][i];
            _grad_wordEmb[offset] += inputcontextLoss[idx][i][0];
            _indexers.insert(offset);
          }
        }
      }

      //release
      for (int idx = 0; idx < seq_size; idx++) {
        for (int idy = 0; idy < _wordcontext; idy++) {
          FreeSpace(&(inputcontext[idx][idy]));
          FreeSpace(&(inputcontextMask[idx][idy]));
          FreeSpace(&(inputcontextLoss[idx][idy]));
        }

        FreeSpace(&(input[idx]));
        FreeSpace(&(inputLoss[idx]));
        FreeSpace(&(project[idx]));
        FreeSpace(&(projectLoss[idx]));
        FreeSpace(&(sparse[idx]));
        FreeSpace(&(sparseLoss[idx]));
        FreeSpace(&(sparseout[idx]));
        FreeSpace(&(sparseoutLoss[idx]));
        FreeSpace(&(denseout[idx]));
        FreeSpace(&(denseoutLoss[idx]));
        FreeSpace(&(output[idx]));
        FreeSpace(&(outputLoss[idx]));
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

    Tensor<xpu, 2, double> input[seq_size];
    Tensor<xpu, 2, double> project[seq_size];
    Tensor<xpu, 2, double> sparse[seq_size];
    Tensor<xpu, 2, double> sparseout[seq_size];
    Tensor<xpu, 2, double> denseout[seq_size];
    Tensor<xpu, 2, double> output[seq_size];

    Tensor<xpu, 2, double> inputcontext[seq_size][_wordcontext];

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      for (int idy = 0; idy < _wordcontext; idy++) {
        inputcontext[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
      }

      input[idx] = NewTensor<xpu>(Shape2(1, _inputsize), 0.0);
      project[idx] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
      sparse[idx] = NewTensor<xpu>(Shape2(1, _linearHiddenSize), 0.0);
      sparseout[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      denseout[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      output[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
    }

    //forward propagation
    //input setting, and linear setting
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = features[idx];
      //linear features should not be dropped out

      const vector<int>& ngram_words = feature.ngram_words;
      assert(ngram_words.size() == _wordcontext);

      for (int i = 0; i < _wordcontext; i++) {
        offset = ngram_words[i];
        inputcontext[idx][i][0] = _wordEmb[offset] / _ft_wordEmb[offset];
      }

      offset = 0;
      for (int i = 0; i < _wordcontext; i++) {
        for (int j = 0; j < _wordDim; j++) {
          input[idx][0][offset] = inputcontext[idx][i][0][j];
          offset++;
        }
      }
    }

    for (int idx = 0; idx < seq_size; idx++) {
      _tanh_project.ComputeForwardScore(input[idx], project[idx]);
      _olayer_linear.ComputeForwardScore(project[idx], denseout[idx]);
      _sparselayer_linear.ComputeForwardScore(features[idx].linear_features, sparse[idx]);
      _olayer_sparselinear.ComputeForwardScore(sparse[idx], sparseout[idx]);
      output[idx] = denseout[idx] + sparseout[idx];
    }

    // decode algorithm
    // viterbi algorithm
    NRMat<double> maxscores(seq_size, _labelSize);
    NRMat<int> maxlastlabels(seq_size, _labelSize);

    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < _labelSize; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          maxscores[idx][i] = output[idx][0][i];
          maxlastlabels[idx][i] = -1;
        } else {
          int maxlastlabel = 0;
          double maxscore = _tagBigram[0][i] + output[idx][0][i] + maxscores[idx - 1][0];
          for (int j = 1; j < _labelSize; ++j) {
            double curscore = _tagBigram[j][i] + output[idx][0][i] + maxscores[idx - 1][j];
            if (curscore > maxscore) {
              maxlastlabel = j;
              maxscore = curscore;
            }
          }
          maxscores[idx][i] = maxscore;
          maxlastlabels[idx][i] = maxlastlabel;
        }
      }
    }

    results.resize(seq_size);
    double maxFinalScore = maxscores[seq_size - 1][0];
    results[seq_size - 1] = 0;
    for (int i = 1; i < _labelSize; ++i) {
      if (maxscores[seq_size - 1][i] > maxFinalScore) {
        maxFinalScore = maxscores[seq_size - 1][i];
        results[seq_size - 1] = i;
      }
    }

    for (int idx = seq_size - 2; idx >= 0; idx--) {
      results[idx] = maxlastlabels[idx + 1][results[idx + 1]];
    }

    //release
    for (int idx = 0; idx < seq_size; idx++) {
      for (int idy = 0; idy < _wordcontext; idy++) {
        FreeSpace(&(inputcontext[idx][idy]));
      }
      FreeSpace(&(input[idx]));
      FreeSpace(&(project[idx]));
      FreeSpace(&(sparse[idx]));
      FreeSpace(&(sparseout[idx]));
      FreeSpace(&(denseout[idx]));
      FreeSpace(&(output[idx]));
    }
  }

  double computeScore(const Example& example) {
    int seq_size = example.m_features.size();
    int offset = 0;

    Tensor<xpu, 2, double> input[seq_size];
    Tensor<xpu, 2, double> project[seq_size];
    Tensor<xpu, 2, double> sparse[seq_size];
    Tensor<xpu, 2, double> sparseout[seq_size];
    Tensor<xpu, 2, double> denseout[seq_size];
    Tensor<xpu, 2, double> output[seq_size];

    Tensor<xpu, 2, double> inputcontext[seq_size][_wordcontext];

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      for (int idy = 0; idy < _wordcontext; idy++) {
        inputcontext[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
      }

      input[idx] = NewTensor<xpu>(Shape2(1, _inputsize), 0.0);
      project[idx] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
      sparse[idx] = NewTensor<xpu>(Shape2(1, _linearHiddenSize), 0.0);
      sparseout[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      denseout[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      output[idx] = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
    }

    //forward propagation
    //input setting, and linear setting
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = example.m_features[idx];
      //linear features should not be dropped out

      const vector<int>& ngram_words = feature.ngram_words;
      assert(ngram_words.size() == _wordcontext);

      for (int i = 0; i < _wordcontext; i++) {
        offset = ngram_words[i];
        inputcontext[idx][i][0] = _wordEmb[offset] / _ft_wordEmb[offset];
      }

      offset = 0;
      for (int i = 0; i < _wordcontext; i++) {
        for (int j = 0; j < _wordDim; j++) {
          input[idx][0][offset] = inputcontext[idx][i][0][j];
          offset++;
        }
      }

    }

    for (int idx = 0; idx < seq_size; idx++) {
      _tanh_project.ComputeForwardScore(input[idx], project[idx]);
      _olayer_linear.ComputeForwardScore(project[idx], denseout[idx]);
      _sparselayer_linear.ComputeForwardScore(example.m_features[idx].linear_features, sparse[idx]);
      _olayer_sparselinear.ComputeForwardScore(sparse[idx], sparseout[idx]);
      output[idx] = denseout[idx] + sparseout[idx];
    }

    // get delta for each output
    // viterbi algorithm
    NRVec<int> goldlabels(seq_size);
    goldlabels = -1;
    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < _labelSize; ++i) {
        if (example.m_labels[idx][i] == 1) {
          goldlabels[idx] = i;
        }
      }
    }

    NRMat<double> maxscores(seq_size, _labelSize);
    NRMat<int> maxlastlabels(seq_size, _labelSize);
    double goldScore = 0.0;
    for (int idx = 0; idx < seq_size; idx++) {
      if (idx == 0)
        goldScore = output[idx][0][goldlabels[idx]];
      else
        goldScore += output[idx][0][goldlabels[idx]] + _tagBigram[goldlabels[idx - 1]][goldlabels[idx]];
      double delta = 1.0;
      for (int i = 0; i < _labelSize; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          maxscores[idx][i] = output[idx][0][i];
          if (goldlabels[idx] != i)
            maxscores[idx][i] = maxscores[idx][i] + delta;
          maxlastlabels[idx][i] = -1;
        } else {
          int maxlastlabel = 0;
          double maxscore = _tagBigram[0][i] + output[idx][0][i] + maxscores[idx - 1][0];
          for (int j = 1; j < _labelSize; ++j) {
            double curscore = _tagBigram[j][i] + output[idx][0][i] + maxscores[idx - 1][j];
            if (curscore > maxscore) {
              maxlastlabel = j;
              maxscore = curscore;
            }
          }
          maxscores[idx][i] = maxscore;
          if (goldlabels[idx] != i)
            maxscores[idx][i] = maxscores[idx][i] + delta;
          maxlastlabels[idx][i] = maxlastlabel;

        }
      }
    }

    NRVec<int> optLabels(seq_size);
    optLabels = 0;
    double maxScore = maxscores[seq_size - 1][0];
    for (int i = 1; i < _labelSize; ++i) {
      if (maxscores[seq_size - 1][i] > maxScore) {
        maxScore = maxscores[seq_size - 1][i];
        optLabels[seq_size - 1] = i;
      }
    }

    for (int idx = seq_size - 2; idx >= 0; idx--) {
      optLabels[idx] = maxlastlabels[idx + 1][optLabels[idx + 1]];
    }

    bool bcorrect = true;
    for (int idx = 0; idx < seq_size; idx++) {
      if (goldlabels[idx] == -1)
        continue;
      if (optLabels[idx] == goldlabels[idx]) {
      } else {
        bcorrect = false;
      }
    }

    double cost = bcorrect ? 0.0 : 1.0;
    //release
    for (int idx = 0; idx < seq_size; idx++) {
      for (int idy = 0; idy < _wordcontext; idy++) {
        FreeSpace(&(inputcontext[idx][idy]));
      }
      FreeSpace(&(input[idx]));
      FreeSpace(&(project[idx]));
      FreeSpace(&(sparse[idx]));
      FreeSpace(&(sparseout[idx]));
      FreeSpace(&(denseout[idx]));
      FreeSpace(&(output[idx]));
    }
    return cost;
  }

  void updateParams(double nnRegular, double adaAlpha, double adaEps) {
    _tanh_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _sparselayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_sparselinear.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _grad_tagBigram = _grad_tagBigram + _tagBigram * nnRegular;
    _eg2_tagBigram = _eg2_tagBigram + _grad_tagBigram * _grad_tagBigram;
    _tagBigram = _tagBigram - _grad_tagBigram * adaAlpha / F<nl_sqrt>(_eg2_tagBigram + adaEps);
    _grad_tagBigram = 0.0;

    if (_b_wordEmb_finetune) {
      static hash_set<int>::iterator it;
      Tensor<xpu, 1, double> _grad_wordEmb_ij = NewTensor<xpu>(Shape1(_wordDim), 0.0);
      Tensor<xpu, 1, double> tmp_normaize_alpha = NewTensor<xpu>(Shape1(_wordDim), 0.0);
      Tensor<xpu, 1, double> tmp_alpha = NewTensor<xpu>(Shape1(_wordDim), 0.0);
      Tensor<xpu, 1, double> _ft_wordEmb_ij = NewTensor<xpu>(Shape1(_wordDim), 0.0);

      for (it = _indexers.begin(); it != _indexers.end(); ++it) {
        int index = *it;
        _grad_wordEmb_ij = _grad_wordEmb[index] + nnRegular * _wordEmb[index] / _ft_wordEmb[index];
        _eg2_wordEmb[index] += _grad_wordEmb_ij * _grad_wordEmb_ij;
        tmp_normaize_alpha = F<nl_sqrt>(_eg2_wordEmb[index] + adaEps);
        tmp_alpha = adaAlpha / tmp_normaize_alpha;
        _ft_wordEmb_ij = _ft_wordEmb[index] * tmp_alpha * nnRegular;
        _ft_wordEmb[index] -= _ft_wordEmb_ij;
        _wordEmb[index] -= tmp_alpha * _grad_wordEmb[index] / _ft_wordEmb[index];
        _grad_wordEmb[index] = 0.0;
      }

      FreeSpace(&_grad_wordEmb_ij);
      FreeSpace(&tmp_normaize_alpha);
      FreeSpace(&tmp_alpha);
      FreeSpace(&_ft_wordEmb_ij);
    }
  }

  void writeModel();

  void loadModel();

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, double>& Wd, const Tensor<xpu, 2, double>& gradWd, const string& mark, int iter) {
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

    double orginValue = Wd[check_i][check_j];

    Wd[check_i][check_j] = orginValue + 0.001;
    double lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossAdd += computeScore(oneExam);
    }

    Wd[check_i][check_j] = orginValue - 0.001;
    double lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossPlus += computeScore(oneExam);
    }

    double mockGrad = (lossAdd - lossPlus) / 0.002;
    mockGrad = mockGrad / examples.size();
    double computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;
  }

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, double>& Wd, const Tensor<xpu, 2, double>& gradWd, const string& mark, int iter,
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

    double orginValue = Wd[check_i][check_j];

    Wd[check_i][check_j] = orginValue + 0.001;
    double lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossAdd += computeScore(oneExam);
    }

    Wd[check_i][check_j] = orginValue - 0.001;
    double lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossPlus += computeScore(oneExam);
    }

    double mockGrad = (lossAdd - lossPlus) / 0.002;
    mockGrad = mockGrad / examples.size();
    double computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;

  }

  void checkgrads(const vector<Example>& examples, int iter) {

    checkgrad(examples, _olayer_linear._W, _olayer_linear._gradW, "_olayer_linear._W", iter);
    checkgrad(examples, _olayer_linear._b, _olayer_linear._gradb, "_olayer_linear._b", iter);

    checkgrad(examples, _sparselayer_linear._W, _sparselayer_linear._gradW, "_sparselayer_linear._W", iter, _sparselayer_linear._indexers, false);
    checkgrad(examples, _sparselayer_linear._b, _sparselayer_linear._gradb, "_sparselayer_linear._b", iter);

    checkgrad(examples, _olayer_sparselinear._W, _olayer_sparselinear._gradW, "_olayer_sparselinear._W", iter);
    checkgrad(examples, _olayer_sparselinear._b, _olayer_sparselinear._gradb, "_olayer_sparselinear._b", iter);

    checkgrad(examples, _tanh_project._W, _tanh_project._gradW, "_tanh_project._W", iter);
    checkgrad(examples, _tanh_project._b, _tanh_project._gradb, "_tanh_project._b", iter);

    checkgrad(examples, _wordEmb, _grad_wordEmb, "_wordEmb", iter, _indexers);

  }

public:
  inline void resetEval() {
    _eval.reset();
  }

  inline void setDropValue(double dropOut) {
    _dropOut = dropOut;
  }

  inline void setWordEmbFinetune(bool b_wordEmb_finetune) {
    _b_wordEmb_finetune = b_wordEmb_finetune;
  }

};

#endif /* SRC_Sparse2TNNCRFMMClassifier_H_ */

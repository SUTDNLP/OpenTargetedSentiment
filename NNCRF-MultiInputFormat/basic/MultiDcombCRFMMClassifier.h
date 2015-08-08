/*
 * MultiDcombCRFMMClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_MultiDcombCRFMMClassifier_H_
#define SRC_MultiDcombCRFMMClassifier_H_

#include <iostream>
#include <armadillo>
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
class MultiDcombCRFMMClassifier {
public:
  MultiDcombCRFMMClassifier() {
    _b_wordEmb_finetune = false;
    _dropOut = 0.5;
  }
  ~MultiDcombCRFMMClassifier() {

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

  Tensor<xpu, 2, double> _tagBigram1;
  Tensor<xpu, 2, double> _grad_tagBigram1;
  Tensor<xpu, 2, double> _eg2_tagBigram1;

  Tensor<xpu, 2, double> _tagBigram2;
  Tensor<xpu, 2, double> _grad_tagBigram2;
  Tensor<xpu, 2, double> _eg2_tagBigram2;

  int _inputSize;
  int _inputcontext;
  UniHidderLayer<xpu> _olayer_sparselinear1;
  UniHidderLayer<xpu> _olayer_sparselinear2;
  SparseUniHidderLayer<xpu> _sparselayer_linear;
  int _linearfeatSize;
  int _linearHiddenSize;
  UniHidderLayer<xpu> _olayer_denselinear1;
  UniHidderLayer<xpu> _olayer_denselinear2;
  UniHidderLayer<xpu> _sharelayer_projected;
  int _hiddenSize;

  //Gated Recursive Unit
  UniHidderLayer<xpu> _atom_reset_input;
  UniHidderLayer<xpu> _atom_gate_input;
  BiHidderLayer<xpu> _atom_hidden_input;

  int _atom_composition_layer_num;

  int _label1Size, _label1_o;
  int _label2Size, _label2_o;

  Metric _eval1, _eval2;

  double _dropOut;

public:

  inline void init(const NRMat<double>& wordEmb, int wordcontext, int label1Size, int label2Size, int hiddenSize, int atom_composition_layer_num,
      int linearHiddenSize, int linearfeatSize) {
    _wordcontext = wordcontext;
    _wordSize = wordEmb.nrows();
    _wordDim = wordEmb.ncols();

    _hiddenSize = hiddenSize;
    _linearfeatSize = linearfeatSize;
    _linearHiddenSize = linearHiddenSize;
    _label1Size = label1Size;
    _label2Size = label2Size;
    _atom_composition_layer_num = atom_composition_layer_num;
    if (atom_composition_layer_num > _wordcontext)
      _atom_composition_layer_num = _wordcontext;

    _wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 0.0);
    _grad_wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 0.0);
    _eg2_wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 0.0);
    _ft_wordEmb = NewTensor<xpu>(Shape2(_wordSize, _wordDim), 1.0);
    assign(_wordEmb, wordEmb);
    for (int idx = 0; idx < _wordSize; idx++) {
      norm2one(_wordEmb, idx);
    }

    Random<xpu, double> rnd(100);
    _tagBigram1 = NewTensor<xpu>(Shape2(_label1Size, _label1Size), 0.0);
    _grad_tagBigram1 = NewTensor<xpu>(Shape2(_label1Size, _label1Size), 0.0);
    _eg2_tagBigram1 = NewTensor<xpu>(Shape2(_label1Size, _label1Size), 0.0);

    rnd.SampleUniform(&_tagBigram1, -0.1, 0.1);

    _tagBigram2 = NewTensor<xpu>(Shape2(_label2Size, _label2Size), 0.0);
    _grad_tagBigram2 = NewTensor<xpu>(Shape2(_label2Size, _label2Size), 0.0);
    _eg2_tagBigram2 = NewTensor<xpu>(Shape2(_label2Size, _label2Size), 0.0);

    rnd.SampleUniform(&_tagBigram2, -0.1, 0.1);

    _atom_reset_input.initial(2 * _wordDim, 2 * _wordDim, false, 101, 1);
    _atom_gate_input.initial(3*_wordDim, 3 * _wordDim, false, 102, 2);
    _atom_hidden_input.initial(_wordDim, _wordDim, _wordDim, false, 106, 0);

    _inputSize = _wordDim * _wordcontext;
    _inputcontext = _wordcontext;

    for (int idx = 1; idx < _atom_composition_layer_num; idx++) {
      _inputSize += _wordDim * (_wordcontext - idx);
      _inputcontext = _inputcontext + (_wordcontext - idx);
    }

    _sharelayer_projected.initial(_hiddenSize, _inputSize, true, 3, 0);
    _sparselayer_linear.initial(_linearHiddenSize, _linearfeatSize, false, 4, 0);
    _olayer_denselinear1.initial(_label1Size, _hiddenSize, false, 5, 2);
    _olayer_sparselinear1.initial(_label1Size, _linearHiddenSize, false, 6, 2);
    _olayer_denselinear2.initial(_label2Size, _hiddenSize, false, 7, 2);
    _olayer_sparselinear2.initial(_label2Size, _linearHiddenSize, false, 8, 2);
  }

  inline void release() {
    FreeSpace(&_wordEmb);
    FreeSpace(&_grad_wordEmb);
    FreeSpace(&_eg2_wordEmb);
    FreeSpace(&_ft_wordEmb);
    _olayer_denselinear1.release();
    _olayer_sparselinear1.release();
    _olayer_denselinear2.release();
    _olayer_sparselinear2.release();
    _sharelayer_projected.release();
    _sparselayer_linear.release();

    FreeSpace(&_tagBigram1);
    FreeSpace(&_grad_tagBigram1);
    FreeSpace(&_eg2_tagBigram1);

    FreeSpace(&_tagBigram2);
    FreeSpace(&_grad_tagBigram2);
    FreeSpace(&_eg2_tagBigram2);

    _atom_reset_input.release();
    _atom_gate_input.release();
    _atom_hidden_input.release();

  }

  inline double process(const vector<Example>& examples, int iter) {
    _eval1.reset(); _eval2.reset();
    _indexers.clear();

    int example_num = examples.size();
    double cost = 0.0;
    int offset = 0;
    int gru_end = _inputcontext;
    int curlayer, curlayerSize, leftchild, rightchild;
    for (int count = 0; count < example_num; count++) {
      const Example& example = examples[count];

      int seq_size = example.m_features.size();

      Tensor<xpu, 2, double> input[seq_size], inputLoss[seq_size];
      Tensor<xpu, 2, double> projected[seq_size], projectedLoss[seq_size];
      Tensor<xpu, 2, double> sparse[seq_size], sparseLoss[seq_size];

      Tensor<xpu, 2, double> denseout1[seq_size], denseout1Loss[seq_size];
      Tensor<xpu, 2, double> sparseout1[seq_size], sparseout1Loss[seq_size];
      Tensor<xpu, 2, double> output1[seq_size], output1Loss[seq_size];

      Tensor<xpu, 2, double> denseout2[seq_size], denseout2Loss[seq_size];
      Tensor<xpu, 2, double> sparseout2[seq_size], sparseout2Loss[seq_size];
      Tensor<xpu, 2, double> output2[seq_size], output2Loss[seq_size];

      Tensor<xpu, 2, double> sparseLossTmp = NewTensor<xpu>(Shape2(1, _linearHiddenSize), 0.0);
      Tensor<xpu, 2, double> denseLossTmp = NewTensor<xpu>(Shape2(1, _hiddenSize), 0.0);

      vector<vector<int> > linear_features(seq_size);

      //GRU
      Tensor<xpu, 2, double> inputcontext[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontextMask[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontextLoss[seq_size][_inputcontext];

      Tensor<xpu, 2, double> inputcontext_reset_left[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_reset_leftLoss[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_reset_right[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_reset_rightLoss[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_reset[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_resetLoss[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_afterreset_left[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_afterreset_leftLoss[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_afterreset_right[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_afterreset_rightLoss[seq_size][_inputcontext];

      Tensor<xpu, 2, double> inputcontext_gate_left[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_gate_leftLoss[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_gate_right[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_gate_rightLoss[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_gate_current[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_gate_currentLoss[seq_size][_inputcontext];

      Tensor<xpu, 2, double> inputcontext_gate_tmp[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_gate_tmpLoss[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_gate[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_gateLoss[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_gate_pool[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_gate_poolLoss[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_gate_norm[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_gate_normLoss[seq_size][_inputcontext];

      Tensor<xpu, 2, double> inputcontext_current[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_currentLoss[seq_size][_inputcontext];

      Tensor<xpu, 2, double> inputcontext_lr[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_lrLoss[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_lrc[seq_size][_inputcontext];
      Tensor<xpu, 2, double> inputcontext_lrcLoss[seq_size][_inputcontext];
      //end gru

      //initialize
      for (int idx = 0; idx < seq_size; idx++) {
        for (int idy = 0; idy < _inputcontext; idy++) {
          inputcontext[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
          inputcontextMask[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 1.0);
          inputcontextLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);

          inputcontext_current[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
          inputcontext_currentLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
          inputcontext_lr[idx][idy] = NewTensor<xpu>(Shape2(1, 2 * _wordDim), 0.0);
          inputcontext_lrLoss[idx][idy] = NewTensor<xpu>(Shape2(1, 2 * _wordDim), 0.0);
          inputcontext_lrc[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _wordDim), 0.0);
          inputcontext_lrcLoss[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _wordDim), 0.0);

          inputcontext_reset[idx][idy] = NewTensor<xpu>(Shape2(1, 2 * _wordDim), 0.0);
          inputcontext_resetLoss[idx][idy] = NewTensor<xpu>(Shape2(1, 2 * _wordDim), 0.0);
          inputcontext_reset_left[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
          inputcontext_reset_leftLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
          inputcontext_reset_right[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
          inputcontext_reset_rightLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);

          inputcontext_afterreset_left[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
          inputcontext_afterreset_leftLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
          inputcontext_afterreset_right[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
          inputcontext_afterreset_rightLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);

          inputcontext_gate_left[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
          inputcontext_gate_leftLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
          inputcontext_gate_right[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
          inputcontext_gate_rightLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
          inputcontext_gate_current[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
          inputcontext_gate_currentLoss[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);

          inputcontext_gate_tmp[idx][idy] = NewTensor<xpu>(Shape2(1, 3*_wordDim), 0.0);
          inputcontext_gate_tmpLoss[idx][idy] = NewTensor<xpu>(Shape2(1, 3*_wordDim), 0.0);
          inputcontext_gate[idx][idy] = NewTensor<xpu>(Shape2(1, 3*_wordDim), 0.0);
          inputcontext_gateLoss[idx][idy] = NewTensor<xpu>(Shape2(1, 3*_wordDim), 0.0);
          inputcontext_gate_pool[idx][idy] = NewTensor<xpu>(Shape2(1, 3*_wordDim), 0.0);
          inputcontext_gate_poolLoss[idx][idy] = NewTensor<xpu>(Shape2(1, 3*_wordDim), 0.0);
          inputcontext_gate_norm[idx][idy] = NewTensor<xpu>(Shape2(1, 3*_wordDim), 0.0);
          inputcontext_gate_normLoss[idx][idy] = NewTensor<xpu>(Shape2(1, 3*_wordDim), 0.0);
        }
        input[idx] = NewTensor<xpu>(Shape2(1, _inputSize), 0.0);
        inputLoss[idx] = NewTensor<xpu>(Shape2(1, _inputSize), 0.0);
        projected[idx] = NewTensor<xpu>(Shape2(1, _hiddenSize), 0.0);
        projectedLoss[idx] = NewTensor<xpu>(Shape2(1, _hiddenSize), 0.0);
        sparse[idx] = NewTensor<xpu>(Shape2(1, _linearHiddenSize), 0.0);
        sparseLoss[idx] = NewTensor<xpu>(Shape2(1, _linearHiddenSize), 0.0);

        sparseout1[idx] = NewTensor<xpu>(Shape2(1, _label1Size), 0.0);
        sparseout1Loss[idx] = NewTensor<xpu>(Shape2(1, _label1Size), 0.0);
        denseout1[idx] = NewTensor<xpu>(Shape2(1, _label1Size), 0.0);
        denseout1Loss[idx] = NewTensor<xpu>(Shape2(1, _label1Size), 0.0);
        output1[idx] = NewTensor<xpu>(Shape2(1, _label1Size), 0.0);
        output1Loss[idx] = NewTensor<xpu>(Shape2(1, _label1Size), 0.0);

        sparseout2[idx] = NewTensor<xpu>(Shape2(1, _label2Size), 0.0);
        sparseout2Loss[idx] = NewTensor<xpu>(Shape2(1, _label2Size), 0.0);
        denseout2[idx] = NewTensor<xpu>(Shape2(1, _label2Size), 0.0);
        denseout2Loss[idx] = NewTensor<xpu>(Shape2(1, _label2Size), 0.0);
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

        //gru
        curlayer = 1;
        curlayerSize = _wordcontext - 1;
        offset = _wordcontext;
        while (curlayer < _atom_composition_layer_num) {
          for (int i = 0; i < curlayerSize; i++) {
            leftchild = offset - curlayerSize - 1;
            rightchild = leftchild + 1;
            //reset
            concat(inputcontext[idx][leftchild], inputcontext[idx][rightchild], inputcontext_lr[idx][offset]);
            _atom_reset_input.ComputeForwardScore(inputcontext_lr[idx][offset], inputcontext_reset[idx][offset]);
            unconcat(inputcontext_reset_left[idx][offset], inputcontext_reset_right[idx][offset], inputcontext_reset[idx][offset]);
            //current input
            inputcontext_afterreset_left[idx][offset] = inputcontext_reset_left[idx][offset] * inputcontext[idx][leftchild];
            inputcontext_afterreset_right[idx][offset] = inputcontext_reset_right[idx][offset] * inputcontext[idx][rightchild];
            _atom_hidden_input.ComputeForwardScore(inputcontext_afterreset_left[idx][offset], inputcontext_afterreset_right[idx][offset],
                inputcontext_current[idx][offset]);

            //gate
            concat(inputcontext[idx][leftchild], inputcontext[idx][rightchild], inputcontext_current[idx][offset], inputcontext_lrc[idx][offset]);
            //gateleft
            _atom_gate_input.ComputeForwardScore(inputcontext_lrc[idx][offset], inputcontext_gate_tmp[idx][offset]);
            inputcontext_gate[idx][offset] = F<nl_exp>(inputcontext_gate_tmp[idx][offset]);
            for(int j = 0; j < _wordDim; j++){
              double sum = inputcontext_gate[idx][offset][0][j] + inputcontext_gate[idx][offset][0][_wordDim+j] + inputcontext_gate[idx][offset][0][2*_wordDim+j];
              sum = 1.0/sum;
              inputcontext_gate_pool[idx][offset][0][j] = sum;
              inputcontext_gate_pool[idx][offset][0][_wordDim+j] = sum;
              inputcontext_gate_pool[idx][offset][0][2*_wordDim+j] = sum;
            }
            inputcontext_gate_norm[idx][offset] =  inputcontext_gate_pool[idx][offset] * inputcontext_gate[idx][offset];
            unconcat(inputcontext_gate_left[idx][offset], inputcontext_gate_right[idx][offset], inputcontext_gate_current[idx][offset], inputcontext_gate_norm[idx][offset]);

            //for(int j = 0; j <_wordDim; j++)
            //{
              //std::cout << inputcontext_gate_left[idx][offset][0][j] << " " << inputcontext_gate_right[idx][offset][0][j] << " " << inputcontext_gate_current[idx][offset][0][j] << std::endl;
            //}
            //std::cout << std::endl;

            //current hidden
            inputcontext[idx][offset] = inputcontext_gate_left[idx][offset] * inputcontext[idx][leftchild]
                + inputcontext_gate_right[idx][offset] * inputcontext[idx][rightchild]
                + inputcontext_gate_current[idx][offset] * inputcontext_current[idx][offset];
            offset++;
          }
          curlayer++;
          curlayerSize--;
        }
        //end gru
        if (offset != gru_end) {
          std::cout << "error forward computation here" << std::endl;
        }

        offset = 0;
        for (int i = 0; i < _inputcontext; i++) {
          for (int j = 0; j < _wordDim; j++) {
            input[idx][0][offset] = inputcontext[idx][i][0][j];
            offset++;
          }
        }

      }

      for (int idx = 0; idx < seq_size; idx++) {
        _sharelayer_projected.ComputeForwardScore(input[idx], projected[idx]);
        _sparselayer_linear.ComputeForwardScore(linear_features[idx], sparse[idx]);

        _olayer_denselinear1.ComputeForwardScore(projected[idx], denseout1[idx]);
        _olayer_sparselinear1.ComputeForwardScore(sparse[idx], sparseout1[idx]);
        output1[idx] = denseout1[idx] + sparseout1[idx];

        _olayer_denselinear2.ComputeForwardScore(projected[idx], denseout2[idx]);
        _olayer_sparselinear2.ComputeForwardScore(sparse[idx], sparseout2[idx]);
        output2[idx] = denseout2[idx] + sparseout2[idx];
      }

      //compute delta
      // viterbi algorithm
      NRVec<int> goldlabel1s(seq_size);
      goldlabel1s = -1;
      for (int idx = 0; idx < seq_size; idx++) {
        for (int i = 0; i < _label1Size; ++i) {
          if (example.m_label1s[idx][i] == 1) {
            goldlabel1s[idx] = i;
          }
        }
      }

      NRMat<double> maxscore1s(seq_size, _label1Size);
      NRMat<int> maxlastlabel1s(seq_size, _label1Size);
      double gold1Score = 0.0;
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx == 0)
          gold1Score = output1[idx][0][goldlabel1s[idx]];
        else
          gold1Score += output1[idx][0][goldlabel1s[idx]] + _tagBigram1[goldlabel1s[idx - 1]][goldlabel1s[idx]];
        double delta = 1.0;
        for (int i = 0; i < _label1Size; ++i) {
          // can be changed with probabilities in future work
          if (idx == 0) {
            maxscore1s[idx][i] = output1[idx][0][i];
            if (goldlabel1s[idx] != i)
              maxscore1s[idx][i] = maxscore1s[idx][i] + delta;
            maxlastlabel1s[idx][i] = -1;
          } else {
            int maxlastlabel = 0;
            double maxscore = _tagBigram1[0][i] + output1[idx][0][i] + maxscore1s[idx - 1][0];
            for (int j = 1; j < _label1Size; ++j) {
              double curscore = _tagBigram1[j][i] + output1[idx][0][i] + maxscore1s[idx - 1][j];
              if (curscore > maxscore) {
                maxlastlabel = j;
                maxscore = curscore;
              }
            }
            maxscore1s[idx][i] = maxscore;
            if (goldlabel1s[idx] != i)
              maxscore1s[idx][i] = maxscore1s[idx][i] + delta;
            maxlastlabel1s[idx][i] = maxlastlabel;

          }
        }
      }

      NRVec<int> optLabel1s(seq_size);
      optLabel1s = 0;
      double max1Score = maxscore1s[seq_size - 1][0];
      for (int i = 1; i < _label1Size; ++i) {
        if (maxscore1s[seq_size - 1][i] > max1Score) {
          max1Score = maxscore1s[seq_size - 1][i];
          optLabel1s[seq_size - 1] = i;
        }
      }

      for (int idx = seq_size - 2; idx >= 0; idx--) {
        optLabel1s[idx] = maxlastlabel1s[idx + 1][optLabel1s[idx + 1]];
      }

      bool bCorrect = true;
      for (int idx = 0; idx < seq_size; idx++) {
        if (goldlabel1s[idx] == -1)
          continue;
        _eval1.overall_label_count++;
        if (optLabel1s[idx] == goldlabel1s[idx]) {
          _eval1.correct_label_count++;
        } else {
          bCorrect = false;
        }
      }

      double cur1cost = bCorrect ? 0.0 : max1Score - gold1Score;
      cur1cost = cur1cost / example_num;

      for (int idx = 0; idx < seq_size; idx++) {
        if (goldlabel1s[idx] == -1)
          continue;
        if (optLabel1s[idx] != goldlabel1s[idx]) {
          output1Loss[idx][0][optLabel1s[idx]] = cur1cost;
          output1Loss[idx][0][goldlabel1s[idx]] = -cur1cost;
          cost += cur1cost;
        }
        if (idx > 0 && goldlabel1s[idx - 1] >= 0) {
          _grad_tagBigram1[optLabel1s[idx - 1]][optLabel1s[idx]] += cur1cost;
          _grad_tagBigram1[goldlabel1s[idx - 1]][goldlabel1s[idx]] -= cur1cost;
        }
      }

      // viterbi algorithm
      NRVec<int> goldlabel2s(seq_size);
      goldlabel2s = -1;
      for (int idx = 0; idx < seq_size; idx++) {
        for (int i = 0; i < _label2Size; ++i) {
          if (example.m_label2s[idx][i] == 1) {
            goldlabel2s[idx] = i;
          }
        }
      }

      NRMat<double> maxscore2s(seq_size, _label2Size);
      NRMat<int> maxlastlabel2s(seq_size, _label2Size);
      double gold2Score = 0.0;
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx == 0)
          gold2Score = output2[idx][0][goldlabel2s[idx]];
        else
          gold2Score += output2[idx][0][goldlabel2s[idx]] + _tagBigram2[goldlabel2s[idx - 1]][goldlabel2s[idx]];
        double delta = 1.0;
        for (int i = 0; i < _label2Size; ++i) {
          // can be changed with probabilities in future work
          if (idx == 0) {
            maxscore2s[idx][i] = output2[idx][0][i];
            if (goldlabel2s[idx] != i)
              maxscore2s[idx][i] = maxscore2s[idx][i] + delta;
            maxlastlabel2s[idx][i] = -1;
          } else {
            int maxlastlabel = 0;
            double maxscore = _tagBigram2[0][i] + output2[idx][0][i] + maxscore2s[idx - 1][0];
            for (int j = 1; j < _label2Size; ++j) {
              double curscore = _tagBigram2[j][i] + output2[idx][0][i] + maxscore2s[idx - 1][j];
              if (curscore > maxscore) {
                maxlastlabel = j;
                maxscore = curscore;
              }
            }
            maxscore2s[idx][i] = maxscore;
            if (goldlabel2s[idx] != i)
              maxscore2s[idx][i] = maxscore2s[idx][i] + delta;
            maxlastlabel2s[idx][i] = maxlastlabel;
          }
          if (optLabel1s[idx] == _label1_o && i != _label2_o)
            maxscore2s[idx][i] = -1e+20;
        }

      }

      NRVec<int> optLabel2s(seq_size);
      optLabel2s = 0;
      double max2Score = maxscore2s[seq_size - 1][0];
      for (int i = 1; i < _label2Size; ++i) {
        if (maxscore2s[seq_size - 1][i] > max2Score) {
          max2Score = maxscore2s[seq_size - 1][i];
          optLabel2s[seq_size - 1] = i;
        }
      }

      for (int idx = seq_size - 2; idx >= 0; idx--) {
        optLabel2s[idx] = maxlastlabel2s[idx + 1][optLabel2s[idx + 1]];
      }

      bCorrect = true;
      for (int idx = 0; idx < seq_size; idx++) {
        if (goldlabel2s[idx] == -1)
          continue;
        _eval2.overall_label_count++;
        if (optLabel2s[idx] == goldlabel2s[idx]) {
          _eval2.correct_label_count++;
        } else {
          bCorrect = false;
        }
      }

      double cur2cost = bCorrect ? 0.0 : max2Score - gold2Score;

      for (int idx = 0; idx < seq_size; idx++) {
        if (goldlabel2s[idx] == -1)
          continue;
        if (optLabel2s[idx] != goldlabel2s[idx]) {
          output2Loss[idx][0][optLabel2s[idx]] = cur2cost;
          output2Loss[idx][0][goldlabel2s[idx]] = -cur2cost;
          cost += cur2cost;
        }
        if (idx > 0 && goldlabel2s[idx - 1] >= 0) {
          _grad_tagBigram2[optLabel2s[idx - 1]][optLabel2s[idx]] += cur2cost;
          _grad_tagBigram2[goldlabel2s[idx - 1]][goldlabel2s[idx]] -= cur2cost;
        }
      }

      // loss backward propagation
      for (int idx = 0; idx < seq_size; idx++) {
        _olayer_sparselinear1.ComputeBackwardLoss(sparse[idx], sparseout1[idx], output1Loss[idx], sparseLoss[idx]);
        _olayer_denselinear1.ComputeBackwardLoss(projected[idx], denseout1[idx], output1Loss[idx], projectedLoss[idx]);

        _olayer_sparselinear2.ComputeBackwardLoss(sparse[idx], sparseout2[idx], output2Loss[idx], sparseLossTmp);
        _olayer_denselinear2.ComputeBackwardLoss(projected[idx], denseout2[idx], output2Loss[idx], denseLossTmp);

        sparseLoss[idx] = sparseLoss[idx] + sparseLossTmp;
        projectedLoss[idx] = projectedLoss[idx] + denseLossTmp;

        _sparselayer_linear.ComputeBackwardLoss(linear_features[idx], sparse[idx], sparseLoss[idx]);
        _sharelayer_projected.ComputeBackwardLoss(input[idx], projected[idx], projectedLoss[idx], inputLoss[idx]);
        offset = 0;
        for (int i = 0; i < _inputcontext; i++) {
          for (int j = 0; j < _wordDim; j++) {
            inputcontextLoss[idx][i][0][j] = inputLoss[idx][0][offset];
            offset++;
          }
        }

        //gru back-propagation
        curlayer = _atom_composition_layer_num - 1;
        curlayerSize = _wordcontext + 1 - _atom_composition_layer_num;
        offset = gru_end - 1;
        while (curlayer > 0) {
          for (int i = curlayerSize - 1; i >= 0; i--) {
            leftchild = offset - curlayerSize - 1;
            rightchild = leftchild + 1;
            //current hidden
            inputcontextLoss[idx][leftchild] += inputcontextLoss[idx][offset] * inputcontext_gate_left[idx][offset];
            inputcontextLoss[idx][rightchild] += inputcontextLoss[idx][offset] * inputcontext_gate_right[idx][offset];
            inputcontext_currentLoss[idx][offset] += inputcontextLoss[idx][offset] * inputcontext_gate_current[idx][offset];

            inputcontext_gate_leftLoss[idx][offset] += inputcontextLoss[idx][offset] * inputcontext[idx][leftchild];
            inputcontext_gate_rightLoss[idx][offset] += inputcontextLoss[idx][offset] * inputcontext[idx][rightchild];
            inputcontext_gate_currentLoss[idx][offset] += inputcontextLoss[idx][offset] * inputcontext_current[idx][offset];

            //gate
            concat(inputcontext_gate_leftLoss[idx][offset], inputcontext_gate_rightLoss[idx][offset], inputcontext_gate_currentLoss[idx][offset], inputcontext_gate_normLoss[idx][offset]);
            inputcontext_gate_poolLoss[idx][offset] += inputcontext_gate_normLoss[idx][offset] * inputcontext_gate[idx][offset];
            inputcontext_gateLoss[idx][offset] += inputcontext_gate_normLoss[idx][offset] * inputcontext_gate_pool[idx][offset];
            for(int j = 0; j < _wordDim; j++){
              double sumLoss = inputcontext_gate_poolLoss[idx][offset][0][j] + inputcontext_gate_poolLoss[idx][offset][0][_wordDim+j] + inputcontext_gate_poolLoss[idx][offset][0][2*_wordDim+j];
              sumLoss = - sumLoss * inputcontext_gate_pool[idx][offset][0][j] * inputcontext_gate_pool[idx][offset][0][j];
              inputcontext_gateLoss[idx][offset][0][j] += sumLoss;
              inputcontext_gateLoss[idx][offset][0][_wordDim+j] += sumLoss;
              inputcontext_gateLoss[idx][offset][0][2*_wordDim+j] += sumLoss;
            }
            inputcontext_gate_tmpLoss[idx][offset] += inputcontext_gateLoss[idx][offset] * inputcontext_gate[idx][offset];
            _atom_gate_input.ComputeBackwardLoss(inputcontext_lrc[idx][offset], inputcontext_gate_tmp[idx][offset], inputcontext_gate_tmpLoss[idx][offset], inputcontext_lrcLoss[idx][offset]);
            unconcat(inputcontextLoss[idx][leftchild], inputcontextLoss[idx][rightchild], inputcontext_currentLoss[idx][offset],
                inputcontext_lrcLoss[idx][offset]);


            //current input
            _atom_hidden_input.ComputeBackwardLoss(inputcontext_afterreset_left[idx][offset], inputcontext_afterreset_right[idx][offset],
                inputcontext_current[idx][offset], inputcontext_currentLoss[idx][offset], inputcontext_afterreset_leftLoss[idx][offset],
                inputcontext_afterreset_rightLoss[idx][offset]);

            inputcontext_reset_rightLoss[idx][offset] += inputcontext_afterreset_rightLoss[idx][offset] * inputcontext[idx][rightchild];
            inputcontextLoss[idx][rightchild] += inputcontext_afterreset_rightLoss[idx][offset] * inputcontext_reset_right[idx][offset];
            inputcontext_reset_leftLoss[idx][offset] += inputcontext_afterreset_leftLoss[idx][offset] * inputcontext[idx][leftchild];
            inputcontextLoss[idx][leftchild] += inputcontext_afterreset_leftLoss[idx][offset] * inputcontext_reset_left[idx][offset];

            //reset
            concat(inputcontext_reset_leftLoss[idx][offset], inputcontext_reset_rightLoss[idx][offset], inputcontext_resetLoss[idx][offset]);
            _atom_reset_input.ComputeBackwardLoss(inputcontext_lr[idx][offset], inputcontext_reset[idx][offset], inputcontext_resetLoss[idx][offset],
                inputcontext_lrLoss[idx][offset]);
            unconcat(inputcontextLoss[idx][leftchild], inputcontextLoss[idx][rightchild], inputcontext_lrLoss[idx][offset]);

            offset--;
          }
          curlayer--;
          curlayerSize++;
        }

        //end gru
        if (offset != _wordcontext - 1) {
          std::cout << "error back-propagation here" << std::endl;
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
        for (int idy = 0; idy < _inputcontext; idy++) {
          FreeSpace(&(inputcontext[idx][idy]));
          FreeSpace(&(inputcontextMask[idx][idy]));
          FreeSpace(&(inputcontextLoss[idx][idy]));

          FreeSpace(&(inputcontext_current[idx][idy]));
          FreeSpace(&(inputcontext_currentLoss[idx][idy]));
          FreeSpace(&(inputcontext_lr[idx][idy]));
          FreeSpace(&(inputcontext_lrLoss[idx][idy]));
          FreeSpace(&(inputcontext_lrc[idx][idy]));
          FreeSpace(&(inputcontext_lrcLoss[idx][idy]));

          FreeSpace(&(inputcontext_reset[idx][idy]));
          FreeSpace(&(inputcontext_resetLoss[idx][idy]));
          FreeSpace(&(inputcontext_reset_left[idx][idy]));
          FreeSpace(&(inputcontext_reset_leftLoss[idx][idy]));
          FreeSpace(&(inputcontext_reset_right[idx][idy]));
          FreeSpace(&(inputcontext_reset_rightLoss[idx][idy]));

          FreeSpace(&(inputcontext_afterreset_left[idx][idy]));
          FreeSpace(&(inputcontext_afterreset_leftLoss[idx][idy]));
          FreeSpace(&(inputcontext_afterreset_right[idx][idy]));
          FreeSpace(&(inputcontext_afterreset_rightLoss[idx][idy]));

          FreeSpace(&(inputcontext_gate_left[idx][idy]));
          FreeSpace(&(inputcontext_gate_leftLoss[idx][idy]));
          FreeSpace(&(inputcontext_gate_right[idx][idy]));
          FreeSpace(&(inputcontext_gate_rightLoss[idx][idy]));
          FreeSpace(&(inputcontext_gate_current[idx][idy]));
          FreeSpace(&(inputcontext_gate_currentLoss[idx][idy]));

          FreeSpace(&(inputcontext_gate_tmp[idx][idy]));
          FreeSpace(&(inputcontext_gate_tmpLoss[idx][idy]));
          FreeSpace(&(inputcontext_gate[idx][idy]));
          FreeSpace(&(inputcontext_gateLoss[idx][idy]));
          FreeSpace(&(inputcontext_gate_pool[idx][idy]));
          FreeSpace(&(inputcontext_gate_poolLoss[idx][idy]));
          FreeSpace(&(inputcontext_gate_norm[idx][idy]));
          FreeSpace(&(inputcontext_gate_normLoss[idx][idy]));
        }

        FreeSpace(&(input[idx]));
        FreeSpace(&(inputLoss[idx]));
        FreeSpace(&(projected[idx]));
        FreeSpace(&(projectedLoss[idx]));
        FreeSpace(&(sparse[idx]));
        FreeSpace(&(sparseLoss[idx]));
        FreeSpace(&(sparseout1[idx]));
        FreeSpace(&(sparseout1Loss[idx]));
        FreeSpace(&(denseout1[idx]));
        FreeSpace(&(denseout1Loss[idx]));
        FreeSpace(&(output1[idx]));
        FreeSpace(&(output1Loss[idx]));
        FreeSpace(&(sparseout2[idx]));
        FreeSpace(&(sparseout2Loss[idx]));
        FreeSpace(&(denseout2[idx]));
        FreeSpace(&(denseout2Loss[idx]));
        FreeSpace(&(output2[idx]));
        FreeSpace(&(output2Loss[idx]));
      }
      FreeSpace(&sparseLossTmp);
      FreeSpace(&denseLossTmp);
    }

    if (_eval1.getAccuracy() < 0 || _eval2.getAccuracy() < 0) {
      std::cout << "strange" << std::endl;
    }

    return cost;
  }

  void predict(const vector<Feature>& features, vector<int>& result1s, vector<int>& result2s) {
    int seq_size = features.size();
    int offset = 0;
    int gru_end = _inputcontext;
    int curlayer, curlayerSize, leftchild, rightchild;


    Tensor<xpu, 2, double> input[seq_size];
    Tensor<xpu, 2, double> projected[seq_size];
    Tensor<xpu, 2, double> sparse[seq_size];
    Tensor<xpu, 2, double> sparseout1[seq_size];
    Tensor<xpu, 2, double> denseout1[seq_size];
    Tensor<xpu, 2, double> output1[seq_size];
    Tensor<xpu, 2, double> sparseout2[seq_size];
    Tensor<xpu, 2, double> denseout2[seq_size];
    Tensor<xpu, 2, double> output2[seq_size];

    //GRU
    Tensor<xpu, 2, double> inputcontext[seq_size][_inputcontext];

    Tensor<xpu, 2, double> inputcontext_reset_left[seq_size][_inputcontext];
    Tensor<xpu, 2, double> inputcontext_reset_right[seq_size][_inputcontext];
    Tensor<xpu, 2, double> inputcontext_reset[seq_size][_inputcontext];
    Tensor<xpu, 2, double> inputcontext_afterreset_left[seq_size][_inputcontext];
    Tensor<xpu, 2, double> inputcontext_afterreset_right[seq_size][_inputcontext];

    Tensor<xpu, 2, double> inputcontext_gate_left[seq_size][_inputcontext];
    Tensor<xpu, 2, double> inputcontext_gate_right[seq_size][_inputcontext];
    Tensor<xpu, 2, double> inputcontext_gate_current[seq_size][_inputcontext];

    Tensor<xpu, 2, double> inputcontext_gate_tmp[seq_size][_inputcontext];
    Tensor<xpu, 2, double> inputcontext_gate[seq_size][_inputcontext];
    Tensor<xpu, 2, double> inputcontext_gate_pool[seq_size][_inputcontext];
    Tensor<xpu, 2, double> inputcontext_gate_norm[seq_size][_inputcontext];

    Tensor<xpu, 2, double> inputcontext_current[seq_size][_inputcontext];

    Tensor<xpu, 2, double> inputcontext_lr[seq_size][_inputcontext];
    Tensor<xpu, 2, double> inputcontext_lrc[seq_size][_inputcontext];
    //end gru

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      for (int idy = 0; idy < _inputcontext; idy++) {
        inputcontext[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);

        inputcontext_current[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
        inputcontext_lr[idx][idy] = NewTensor<xpu>(Shape2(1, 2 * _wordDim), 0.0);
        inputcontext_lrc[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _wordDim), 0.0);

        inputcontext_reset[idx][idy] = NewTensor<xpu>(Shape2(1, 2 * _wordDim), 0.0);
        inputcontext_reset_left[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
        inputcontext_reset_right[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);

        inputcontext_afterreset_left[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
        inputcontext_afterreset_right[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);

        inputcontext_gate_left[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
        inputcontext_gate_right[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
        inputcontext_gate_current[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);

        inputcontext_gate_tmp[idx][idy] = NewTensor<xpu>(Shape2(1, 3*_wordDim), 0.0);
        inputcontext_gate[idx][idy] = NewTensor<xpu>(Shape2(1, 3*_wordDim), 0.0);
        inputcontext_gate_pool[idx][idy] = NewTensor<xpu>(Shape2(1, 3*_wordDim), 0.0);
        inputcontext_gate_norm[idx][idy] = NewTensor<xpu>(Shape2(1, 3*_wordDim), 0.0);
      }

      input[idx] = NewTensor<xpu>(Shape2(1, _inputSize), 0.0);
      projected[idx] = NewTensor<xpu>(Shape2(1, _hiddenSize), 0.0);
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

      const vector<int>& ngram_words = feature.ngram_words;
      assert(ngram_words.size() == _wordcontext);

      for (int i = 0; i < _wordcontext; i++) {
        offset = ngram_words[i];
        inputcontext[idx][i][0] = _wordEmb[offset] / _ft_wordEmb[offset];
      }

      //gru
      curlayer = 1;
      curlayerSize = _wordcontext - 1;
      offset = _wordcontext;
      while (curlayer < _atom_composition_layer_num) {
        for (int i = 0; i < curlayerSize; i++) {
          leftchild = offset - curlayerSize - 1;
          rightchild = leftchild + 1;
          //reset
          concat(inputcontext[idx][leftchild], inputcontext[idx][rightchild], inputcontext_lr[idx][offset]);
          _atom_reset_input.ComputeForwardScore(inputcontext_lr[idx][offset], inputcontext_reset[idx][offset]);
          unconcat(inputcontext_reset_left[idx][offset], inputcontext_reset_right[idx][offset], inputcontext_reset[idx][offset]);
          //current input
          inputcontext_afterreset_left[idx][offset] = inputcontext_reset_left[idx][offset] * inputcontext[idx][leftchild];
          inputcontext_afterreset_right[idx][offset] = inputcontext_reset_right[idx][offset] * inputcontext[idx][rightchild];
          _atom_hidden_input.ComputeForwardScore(inputcontext_afterreset_left[idx][offset], inputcontext_afterreset_right[idx][offset],
              inputcontext_current[idx][offset]);

          //gate
          concat(inputcontext[idx][leftchild], inputcontext[idx][rightchild], inputcontext_current[idx][offset], inputcontext_lrc[idx][offset]);
          //gateleft
          _atom_gate_input.ComputeForwardScore(inputcontext_lrc[idx][offset], inputcontext_gate_tmp[idx][offset]);
          inputcontext_gate[idx][offset] = F<nl_exp>(inputcontext_gate_tmp[idx][offset]);
          for(int j = 0; j < _wordDim; j++){
            double sum = inputcontext_gate[idx][offset][0][j] + inputcontext_gate[idx][offset][0][_wordDim+j] + inputcontext_gate[idx][offset][0][2*_wordDim+j];
            sum = 1.0/sum;
            inputcontext_gate_pool[idx][offset][0][j] = sum;
            inputcontext_gate_pool[idx][offset][0][_wordDim+j] = sum;
            inputcontext_gate_pool[idx][offset][0][2*_wordDim+j] = sum;
          }
          inputcontext_gate_norm[idx][offset] =  inputcontext_gate_pool[idx][offset] * inputcontext_gate[idx][offset];
          unconcat(inputcontext_gate_left[idx][offset], inputcontext_gate_right[idx][offset], inputcontext_gate_current[idx][offset], inputcontext_gate_norm[idx][offset]);

          //current hidden
          inputcontext[idx][offset] = inputcontext_gate_left[idx][offset] * inputcontext[idx][leftchild]
              + inputcontext_gate_right[idx][offset] * inputcontext[idx][rightchild]
              + inputcontext_gate_current[idx][offset] * inputcontext_current[idx][offset];
          offset++;
        }
        curlayer++;
        curlayerSize--;
      }
      //end gru

      offset = 0;
      for (int i = 0; i < _inputcontext; i++) {
        for (int j = 0; j < _wordDim; j++) {
          input[idx][0][offset] = inputcontext[idx][i][0][j];
          offset++;
        }
      }
    }

    for (int idx = 0; idx < seq_size; idx++) {
      _sharelayer_projected.ComputeForwardScore(input[idx], projected[idx]);
      _sparselayer_linear.ComputeForwardScore(features[idx].linear_features, sparse[idx]);
      _olayer_denselinear1.ComputeForwardScore(projected[idx], denseout1[idx]);
      _olayer_sparselinear1.ComputeForwardScore(sparse[idx], sparseout1[idx]);
      output1[idx] = denseout1[idx] + sparseout1[idx];

      _olayer_denselinear2.ComputeForwardScore(projected[idx], denseout2[idx]);
      _olayer_sparselinear2.ComputeForwardScore(sparse[idx], sparseout2[idx]);
      output2[idx] = denseout2[idx] + sparseout2[idx];
    }

    // viterbi algorithm
    NRMat<double> maxscore1s(seq_size, _label1Size);
    NRMat<int> maxlastlabel1s(seq_size, _label1Size);

    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < _label1Size; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          maxscore1s[idx][i] = output1[idx][0][i];
          maxlastlabel1s[idx][i] = -1;
        } else {
          int maxlastlabel = 0;
          double maxscore = _tagBigram1[0][i] + output1[idx][0][i] + maxscore1s[idx - 1][0];
          for (int j = 1; j < _label1Size; ++j) {
            double curscore = _tagBigram1[j][i] + output1[idx][0][i] + maxscore1s[idx - 1][j];
            if (curscore > maxscore) {
              maxlastlabel = j;
              maxscore = curscore;
            }
          }
          maxscore1s[idx][i] = maxscore;
          maxlastlabel1s[idx][i] = maxlastlabel;
        }
      }
    }

    result1s.resize(seq_size);
    double maxFinalScore = maxscore1s[seq_size - 1][0];
    result1s[seq_size - 1] = 0;
    for (int i = 1; i < _label1Size; ++i) {
      if (maxscore1s[seq_size - 1][i] > maxFinalScore) {
        maxFinalScore = maxscore1s[seq_size - 1][i];
        result1s[seq_size - 1] = i;
      }
    }

    for (int idx = seq_size - 2; idx >= 0; idx--) {
      result1s[idx] = maxlastlabel1s[idx + 1][result1s[idx + 1]];
    }

    // viterbi algorithm
    NRMat<double> maxscore2s(seq_size, _label2Size);
    NRMat<int> maxlastlabel2s(seq_size, _label2Size);

    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < _label2Size; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          maxscore2s[idx][i] = output2[idx][0][i];
          maxlastlabel2s[idx][i] = -1;
        } else {
          int maxlastlabel = 0;
          double maxscore = _tagBigram2[0][i] + output2[idx][0][i] + maxscore2s[idx - 1][0];
          for (int j = 1; j < _label2Size; ++j) {
            double curscore = _tagBigram2[j][i] + output2[idx][0][i] + maxscore2s[idx - 1][j];
            if (curscore > maxscore) {
              maxlastlabel = j;
              maxscore = curscore;
            }
          }
          maxscore2s[idx][i] = maxscore;
          maxlastlabel2s[idx][i] = maxlastlabel;
        }
        if (result1s[idx] == _label1_o && i != _label2_o)
          maxscore2s[idx][i] = -1e+20;
      }
    }

    result2s.resize(seq_size);
    maxFinalScore = maxscore2s[seq_size - 1][0];
    result2s[seq_size - 1] = 0;
    for (int i = 1; i < _label2Size; ++i) {
      if (maxscore2s[seq_size - 1][i] > maxFinalScore) {
        maxFinalScore = maxscore2s[seq_size - 1][i];
        result2s[seq_size - 1] = i;
      }
    }

    for (int idx = seq_size - 2; idx >= 0; idx--) {
      result2s[idx] = maxlastlabel2s[idx + 1][result2s[idx + 1]];
    }

    //release
    for (int idx = 0; idx < seq_size; idx++) {
      for (int idy = 0; idy < _inputcontext; idy++) {
        FreeSpace(&(inputcontext[idx][idy]));

        FreeSpace(&(inputcontext_current[idx][idy]));
        FreeSpace(&(inputcontext_lr[idx][idy]));
        FreeSpace(&(inputcontext_lrc[idx][idy]));

        FreeSpace(&(inputcontext_reset[idx][idy]));
        FreeSpace(&(inputcontext_reset_left[idx][idy]));
        FreeSpace(&(inputcontext_reset_right[idx][idy]));

        FreeSpace(&(inputcontext_afterreset_left[idx][idy]));
        FreeSpace(&(inputcontext_afterreset_right[idx][idy]));

        FreeSpace(&(inputcontext_gate_left[idx][idy]));
        FreeSpace(&(inputcontext_gate_right[idx][idy]));
        FreeSpace(&(inputcontext_gate_current[idx][idy]));

        FreeSpace(&(inputcontext_gate_tmp[idx][idy]));
        FreeSpace(&(inputcontext_gate[idx][idy]));
        FreeSpace(&(inputcontext_gate_pool[idx][idy]));
        FreeSpace(&(inputcontext_gate_norm[idx][idy]));
      }
      FreeSpace(&(input[idx]));
      FreeSpace(&(projected[idx]));
      FreeSpace(&(sparse[idx]));
      FreeSpace(&(denseout1[idx]));
      FreeSpace(&(sparseout1[idx]));
      FreeSpace(&(output1[idx]));
      FreeSpace(&(denseout2[idx]));
      FreeSpace(&(sparseout2[idx]));
      FreeSpace(&(output2[idx]));
    }
  }

  double computeScore(const Example& example) {
    int seq_size = example.m_features.size();
    int offset = 0;
    int gru_end = _inputcontext - _wordcontext;
    int curlayer, curlayerSize, leftchild, rightchild;

    Tensor<xpu, 2, double> input[seq_size];
    Tensor<xpu, 2, double> projected[seq_size];
    Tensor<xpu, 2, double> sparse[seq_size];
    Tensor<xpu, 2, double> sparseout1[seq_size];
    Tensor<xpu, 2, double> denseout1[seq_size];
    Tensor<xpu, 2, double> output1[seq_size];
    Tensor<xpu, 2, double> sparseout2[seq_size];
    Tensor<xpu, 2, double> denseout2[seq_size];
    Tensor<xpu, 2, double> output2[seq_size];

    //GRU
    Tensor<xpu, 2, double> inputcontext[seq_size][_inputcontext];

    Tensor<xpu, 2, double> inputcontext_reset_left[seq_size][_inputcontext];
    Tensor<xpu, 2, double> inputcontext_reset_right[seq_size][_inputcontext];
    Tensor<xpu, 2, double> inputcontext_reset[seq_size][_inputcontext];
    Tensor<xpu, 2, double> inputcontext_afterreset_left[seq_size][_inputcontext];
    Tensor<xpu, 2, double> inputcontext_afterreset_right[seq_size][_inputcontext];

    Tensor<xpu, 2, double> inputcontext_gate_left[seq_size][_inputcontext];
    Tensor<xpu, 2, double> inputcontext_gate_right[seq_size][_inputcontext];
    Tensor<xpu, 2, double> inputcontext_gate_current[seq_size][_inputcontext];

    Tensor<xpu, 2, double> inputcontext_gate_tmp[seq_size][_inputcontext];
    Tensor<xpu, 2, double> inputcontext_gate[seq_size][_inputcontext];
    Tensor<xpu, 2, double> inputcontext_gate_pool[seq_size][_inputcontext];
    Tensor<xpu, 2, double> inputcontext_gate_norm[seq_size][_inputcontext];

    Tensor<xpu, 2, double> inputcontext_current[seq_size][_inputcontext];

    Tensor<xpu, 2, double> inputcontext_lr[seq_size][_inputcontext];
    Tensor<xpu, 2, double> inputcontext_lrc[seq_size][_inputcontext];
    //end gru

    //initialize
    for (int idx = 0; idx < seq_size; idx++) {
      for (int idy = 0; idy < _inputcontext; idy++) {
        inputcontext[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);

        inputcontext_current[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
        inputcontext_lr[idx][idy] = NewTensor<xpu>(Shape2(1, 2 * _wordDim), 0.0);
        inputcontext_lrc[idx][idy] = NewTensor<xpu>(Shape2(1, 3 * _wordDim), 0.0);

        inputcontext_reset[idx][idy] = NewTensor<xpu>(Shape2(1, 2 * _wordDim), 0.0);
        inputcontext_reset_left[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
        inputcontext_reset_right[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);

        inputcontext_afterreset_left[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
        inputcontext_afterreset_right[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);

        inputcontext_gate_left[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
        inputcontext_gate_right[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);
        inputcontext_gate_current[idx][idy] = NewTensor<xpu>(Shape2(1, _wordDim), 0.0);

        inputcontext_gate_tmp[idx][idy] = NewTensor<xpu>(Shape2(1, 3*_wordDim), 0.0);
        inputcontext_gate[idx][idy] = NewTensor<xpu>(Shape2(1, 3*_wordDim), 0.0);
        inputcontext_gate_pool[idx][idy] = NewTensor<xpu>(Shape2(1, 3*_wordDim), 0.0);
        inputcontext_gate_norm[idx][idy] = NewTensor<xpu>(Shape2(1, 3*_wordDim), 0.0);
      }

      input[idx] = NewTensor<xpu>(Shape2(1, _inputSize), 0.0);
      projected[idx] = NewTensor<xpu>(Shape2(1, _hiddenSize), 0.0);
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
      const Feature& feature = example.m_features[idx];
      //linear features should not be dropped out

      const vector<int>& ngram_words = feature.ngram_words;
      assert(ngram_words.size() == _wordcontext);

      for (int i = 0; i < _wordcontext; i++) {
        offset = ngram_words[i];
        inputcontext[idx][i][0] = _wordEmb[offset] / _ft_wordEmb[offset];
      }

      //gru
      curlayer = 1;
      curlayerSize = _wordcontext - 1;
      offset = _wordcontext;

      while (curlayer < _atom_composition_layer_num) {
        for (int i = 0; i < curlayerSize; i++) {
          leftchild = offset - curlayerSize - 1;
          rightchild = leftchild + 1;
          //reset
          concat(inputcontext[idx][leftchild], inputcontext[idx][rightchild], inputcontext_lr[idx][offset]);
          _atom_reset_input.ComputeForwardScore(inputcontext_lr[idx][offset], inputcontext_reset[idx][offset]);
          unconcat(inputcontext_reset_left[idx][offset], inputcontext_reset_right[idx][offset], inputcontext_reset[idx][offset]);
          //current input
          inputcontext_afterreset_left[idx][offset] = inputcontext_reset_left[idx][offset] * inputcontext[idx][leftchild];
          inputcontext_afterreset_right[idx][offset] = inputcontext_reset_right[idx][offset] * inputcontext[idx][rightchild];
          _atom_hidden_input.ComputeForwardScore(inputcontext_afterreset_left[idx][offset], inputcontext_afterreset_right[idx][offset],
              inputcontext_current[idx][offset]);

          //gate
          concat(inputcontext[idx][leftchild], inputcontext[idx][rightchild], inputcontext_current[idx][offset], inputcontext_lrc[idx][offset]);
          //gateleft
          _atom_gate_input.ComputeForwardScore(inputcontext_lrc[idx][offset], inputcontext_gate_tmp[idx][offset]);
          inputcontext_gate[idx][offset] = F<nl_exp>(inputcontext_gate_tmp[idx][offset]);
          for(int j = 0; j < _wordDim; j++){
            double sum = inputcontext_gate[idx][offset][0][j] + inputcontext_gate[idx][offset][0][_wordDim+j] + inputcontext_gate[idx][offset][0][2*_wordDim+j];
            sum = 1.0/sum;
            inputcontext_gate_pool[idx][offset][0][j] = sum;
            inputcontext_gate_pool[idx][offset][0][_wordDim+j] = sum;
            inputcontext_gate_pool[idx][offset][0][2*_wordDim+j] = sum;
          }
          inputcontext_gate_norm[idx][offset] =  inputcontext_gate_pool[idx][offset] * inputcontext_gate[idx][offset];
          unconcat(inputcontext_gate_left[idx][offset], inputcontext_gate_right[idx][offset], inputcontext_gate_current[idx][offset], inputcontext_gate_norm[idx][offset]);

          //current hidden
          inputcontext[idx][offset] = inputcontext_gate_left[idx][offset] * inputcontext[idx][leftchild]
              + inputcontext_gate_right[idx][offset] * inputcontext[idx][rightchild]
              + inputcontext_gate_current[idx][offset] * inputcontext_current[idx][offset];
          offset++;
        }
        curlayer++;
        curlayerSize--;
      }
      //end gru

      offset = 0;
      for (int i = 0; i < _inputcontext; i++) {
        for (int j = 0; j < _wordDim; j++) {
          input[idx][0][offset] = inputcontext[idx][i][0][j];
          offset++;
        }
      }

    }

    for (int idx = 0; idx < seq_size; idx++) {
      _sharelayer_projected.ComputeForwardScore(input[idx], projected[idx]);
      _sparselayer_linear.ComputeForwardScore(example.m_features[idx].linear_features, sparse[idx]);
      _olayer_denselinear1.ComputeForwardScore(projected[idx], denseout1[idx]);
      _olayer_sparselinear1.ComputeForwardScore(sparse[idx], sparseout1[idx]);
      output1[idx] = denseout1[idx] + sparseout1[idx];

      _olayer_denselinear2.ComputeForwardScore(projected[idx], denseout2[idx]);
      _olayer_sparselinear2.ComputeForwardScore(sparse[idx], sparseout2[idx]);
      output2[idx] = denseout2[idx] + sparseout2[idx];
    }

    //compute delta
    double cost = 0.0;
    // viterbi algorithm
    NRVec<int> goldlabel1s(seq_size);
    goldlabel1s = -1;
    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < _label1Size; ++i) {
        if (example.m_label1s[idx][i] == 1) {
          goldlabel1s[idx] = i;
        }
      }
    }

    NRMat<double> maxscore1s(seq_size, _label1Size);
    NRMat<int> maxlastlabel1s(seq_size, _label1Size);
    double gold1Score = 0.0;
    for (int idx = 0; idx < seq_size; idx++) {
      if (idx == 0)
        gold1Score = output1[idx][0][goldlabel1s[idx]];
      else
        gold1Score += output1[idx][0][goldlabel1s[idx]] + _tagBigram1[goldlabel1s[idx - 1]][goldlabel1s[idx]];
      double delta = 1.0;
      for (int i = 0; i < _label1Size; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          maxscore1s[idx][i] = output1[idx][0][i];
          if (goldlabel1s[idx] != i)
            maxscore1s[idx][i] = maxscore1s[idx][i] + delta;
          maxlastlabel1s[idx][i] = -1;
        } else {
          int maxlastlabel = 0;
          double maxscore = _tagBigram1[0][i] + output1[idx][0][i] + maxscore1s[idx - 1][0];
          for (int j = 1; j < _label1Size; ++j) {
            double curscore = _tagBigram1[j][i] + output1[idx][0][i] + maxscore1s[idx - 1][j];
            if (curscore > maxscore) {
              maxlastlabel = j;
              maxscore = curscore;
            }
          }
          maxscore1s[idx][i] = maxscore;
          if (goldlabel1s[idx] != i)
            maxscore1s[idx][i] = maxscore1s[idx][i] + delta;
          maxlastlabel1s[idx][i] = maxlastlabel;

        }
      }
    }

    NRVec<int> optLabel1s(seq_size);
    optLabel1s = 0;
    double max1Score = maxscore1s[seq_size - 1][0];
    for (int i = 1; i < _label1Size; ++i) {
      if (maxscore1s[seq_size - 1][i] > max1Score) {
        max1Score = maxscore1s[seq_size - 1][i];
        optLabel1s[seq_size - 1] = i;
      }
    }

    for (int idx = seq_size - 2; idx >= 0; idx--) {
      optLabel1s[idx] = maxlastlabel1s[idx + 1][optLabel1s[idx + 1]];
    }

    bool bCorrect = true;
    for (int idx = 0; idx < seq_size; idx++) {
      if (goldlabel1s[idx] == -1)
        continue;
      _eval1.overall_label_count++;
      if (optLabel1s[idx] == goldlabel1s[idx]) {
        _eval1.correct_label_count++;
      } else {
        bCorrect = false;
      }
    }

    double cur1cost = bCorrect ? 0.0 : max1Score - gold1Score;
    cost += cur1cost;

    // viterbi algorithm
    NRVec<int> goldlabel2s(seq_size);
    goldlabel2s = -1;
    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < _label2Size; ++i) {
        if (example.m_label2s[idx][i] == 1) {
          goldlabel2s[idx] = i;
        }
      }
    }

    NRMat<double> maxscore2s(seq_size, _label2Size);
    NRMat<int> maxlastlabel2s(seq_size, _label2Size);
    double gold2Score = 0.0;
    for (int idx = 0; idx < seq_size; idx++) {
      if (idx == 0)
        gold2Score = output2[idx][0][goldlabel2s[idx]];
      else
        gold2Score += output2[idx][0][goldlabel2s[idx]] + _tagBigram2[goldlabel2s[idx - 1]][goldlabel2s[idx]];
      double delta = 1.0;
      for (int i = 0; i < _label2Size; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          maxscore2s[idx][i] = output2[idx][0][i];
          if (goldlabel2s[idx] != i)
            maxscore2s[idx][i] = maxscore2s[idx][i] + delta;
          maxlastlabel2s[idx][i] = -1;
        } else {
          int maxlastlabel = 0;
          double maxscore = _tagBigram2[0][i] + output2[idx][0][i] + maxscore2s[idx - 1][0];
          for (int j = 1; j < _label2Size; ++j) {
            double curscore = _tagBigram2[j][i] + output2[idx][0][i] + maxscore2s[idx - 1][j];
            if (curscore > maxscore) {
              maxlastlabel = j;
              maxscore = curscore;
            }
          }
          maxscore2s[idx][i] = maxscore;
          if (goldlabel2s[idx] != i)
            maxscore2s[idx][i] = maxscore2s[idx][i] + delta;
          maxlastlabel2s[idx][i] = maxlastlabel;
        }
        if (optLabel1s[idx] == _label1_o && i != _label2_o)
          maxscore2s[idx][i] = -1e+20;
      }

    }

    NRVec<int> optLabel2s(seq_size);
    optLabel2s = 0;
    double max2Score = maxscore2s[seq_size - 1][0];
    for (int i = 1; i < _label2Size; ++i) {
      if (maxscore2s[seq_size - 1][i] > max2Score) {
        max2Score = maxscore2s[seq_size - 1][i];
        optLabel2s[seq_size - 1] = i;
      }
    }

    for (int idx = seq_size - 2; idx >= 0; idx--) {
      optLabel2s[idx] = maxlastlabel2s[idx + 1][optLabel2s[idx + 1]];
    }

    bCorrect = true;
    for (int idx = 0; idx < seq_size; idx++) {
      if (goldlabel2s[idx] == -1)
        continue;
      _eval2.overall_label_count++;
      if (optLabel2s[idx] == goldlabel2s[idx]) {
        _eval2.correct_label_count++;
      } else {
        bCorrect = false;
      }
    }

    double cur2cost = bCorrect ? 0.0 : max2Score - gold2Score;
    cost += cur2cost;

    //release
    for (int idx = 0; idx < seq_size; idx++) {
      for (int idy = 0; idy < _inputcontext; idy++) {
        FreeSpace(&(inputcontext[idx][idy]));

        FreeSpace(&(inputcontext_current[idx][idy]));
        FreeSpace(&(inputcontext_lr[idx][idy]));
        FreeSpace(&(inputcontext_lrc[idx][idy]));

        FreeSpace(&(inputcontext_reset[idx][idy]));
        FreeSpace(&(inputcontext_reset_left[idx][idy]));
        FreeSpace(&(inputcontext_reset_right[idx][idy]));

        FreeSpace(&(inputcontext_afterreset_left[idx][idy]));
        FreeSpace(&(inputcontext_afterreset_right[idx][idy]));

        FreeSpace(&(inputcontext_gate_left[idx][idy]));
        FreeSpace(&(inputcontext_gate_right[idx][idy]));
        FreeSpace(&(inputcontext_gate_current[idx][idy]));

        FreeSpace(&(inputcontext_gate_tmp[idx][idy]));
        FreeSpace(&(inputcontext_gate[idx][idy]));
        FreeSpace(&(inputcontext_gate_pool[idx][idy]));
        FreeSpace(&(inputcontext_gate_norm[idx][idy]));
      }
      FreeSpace(&(input[idx]));
      FreeSpace(&(projected[idx]));
      FreeSpace(&(sparse[idx]));
      FreeSpace(&(denseout1[idx]));
      FreeSpace(&(sparseout1[idx]));
      FreeSpace(&(output1[idx]));
      FreeSpace(&(denseout2[idx]));
      FreeSpace(&(sparseout2[idx]));
      FreeSpace(&(output2[idx]));
    }
    return cost;
  }

  void updateParams(double nnRegular, double adaAlpha, double adaEps) {
    _olayer_denselinear1.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_sparselinear1.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_denselinear2.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_sparselinear2.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _sparselayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _sharelayer_projected.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _atom_reset_input.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _atom_gate_input.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _atom_hidden_input.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _grad_tagBigram1 = _grad_tagBigram1 + _tagBigram1 * nnRegular;
    _eg2_tagBigram1 = _eg2_tagBigram1 + _grad_tagBigram1 * _grad_tagBigram1;
    _tagBigram1 = _tagBigram1 - _grad_tagBigram1 * adaAlpha / F<nl_sqrt>(_eg2_tagBigram1 + adaEps);
    _grad_tagBigram1 = 0.0;

    _grad_tagBigram2 = _grad_tagBigram2 + _tagBigram2 * nnRegular;
    _eg2_tagBigram2 = _eg2_tagBigram2 + _grad_tagBigram2 * _grad_tagBigram2;
    _tagBigram2 = _tagBigram2 - _grad_tagBigram2 * adaAlpha / F<nl_sqrt>(_eg2_tagBigram2 + adaEps);
    _grad_tagBigram2 = 0.0;

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
      const hash_set<int>& indexes) {
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols;
    idRows.clear();
    idCols.clear();
    static hash_set<int>::iterator it;
    for (it = indexes.begin(); it != indexes.end(); ++it)
      idRows.push_back(*it);
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

  void checkgrads(const vector<Example>& examples, int iter) {
    //checkgrad(examples, _olayer_linear2._W, _olayer_linear2._gradW, "_olayer_linear2._W", iter);
    //checkgrad(examples, _olayer_linear2._b, _olayer_linear2._gradb, "_olayer_linear2._b", iter);

    checkgrad(examples, _atom_reset_input._W, _atom_reset_input._gradW, "_atom_reset_input._W", iter);
    checkgrad(examples, _atom_reset_input._b, _atom_reset_input._gradb, "_atom_reset_input._b", iter);

    checkgrad(examples, _atom_gate_input._W, _atom_gate_input._gradW, "_atom_gate_input._W", iter);
    checkgrad(examples, _atom_gate_input._b, _atom_gate_input._gradb, "_atom_gate_input._b", iter);

    checkgrad(examples, _atom_hidden_input._WL, _atom_hidden_input._gradWL, "_atom_hidden_input._WL", iter);
    checkgrad(examples, _atom_hidden_input._WR, _atom_hidden_input._gradWR, "_atom_hidden_input._WR", iter);
    checkgrad(examples, _atom_hidden_input._b, _atom_hidden_input._gradb, "_atom_hidden_input._b", iter);

    checkgrad(examples, _wordEmb, _grad_wordEmb, "_wordEmb", iter, _indexers);
  }

public:
  inline void resetEval() {
    _eval2.reset();
  }

  inline void setDropValue(double dropOut) {
    _dropOut = dropOut;
  }

  inline void setWordEmbFinetune(bool b_wordEmb_finetune) {
    _b_wordEmb_finetune = b_wordEmb_finetune;
  }

  inline void setLabelO(int label1_o, int label2_o)
  {
    _label1_o = label1_o; _label2_o = label2_o;
  }

};

#endif /* SRC_MultiDcombCRFMMClassifier_H_ */

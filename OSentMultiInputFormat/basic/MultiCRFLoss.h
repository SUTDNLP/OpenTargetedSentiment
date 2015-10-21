/*
 * MultiCRFLoss.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_MultiCRFLoss_H_
#define SRC_MultiCRFLoss_H_
#include "tensor.h"
#include "MyLib.h"
#include "Utiltensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class MultiCRFLoss {

public:

  Tensor<xpu, 2, dtype> _tagBigram1;
  Tensor<xpu, 2, dtype> _grad_tagBigram1;
  Tensor<xpu, 2, dtype> _eg2_tagBigram1;

  Tensor<xpu, 2, dtype> _tagBigram2;
  Tensor<xpu, 2, dtype> _grad_tagBigram2;
  Tensor<xpu, 2, dtype> _eg2_tagBigram2;

  dtype _delta;

  int _label1_o, _label2_o;

public:
  MultiCRFLoss() {
  }

  inline void initial(int label1Size, int label2Size, int seed = 0) {
    //dtype bound = sqrt(6.0 / (nLabelSize + nLabelSize + 1));
    dtype bound = 0.1;

    _tagBigram1 = NewTensor<xpu>(Shape2(label1Size, label1Size), d_zero);
    _grad_tagBigram1 = NewTensor<xpu>(Shape2(label1Size, label1Size), d_zero);
    _eg2_tagBigram1 = NewTensor<xpu>(Shape2(label1Size, label1Size), d_zero);

    random(_tagBigram1, -bound, bound, seed);
    
    _tagBigram2 = NewTensor<xpu>(Shape2(label2Size, label2Size), d_zero);
    _grad_tagBigram2 = NewTensor<xpu>(Shape2(label2Size, label2Size), d_zero);
    _eg2_tagBigram2 = NewTensor<xpu>(Shape2(label2Size, label2Size), d_zero);

    random(_tagBigram2, -bound, bound, seed + 1);    

    _delta = 0.2;
    
    _label1_o = -1;
    _label2_o = -1;
  }


  inline void initial(int label1Size, int label2Size, dtype delta, int seed = 0) {
    //dtype bound = sqrt(6.0 / (nLabelSize + nLabelSize + 1));
    dtype bound = 0.1;

    _tagBigram1 = NewTensor<xpu>(Shape2(label1Size, label1Size), d_zero);
    _grad_tagBigram1 = NewTensor<xpu>(Shape2(label1Size, label1Size), d_zero);
    _eg2_tagBigram1 = NewTensor<xpu>(Shape2(label1Size, label1Size), d_zero);

    random(_tagBigram1, -bound, bound, seed);
    
    _tagBigram2 = NewTensor<xpu>(Shape2(label2Size, label2Size), d_zero);
    _grad_tagBigram2 = NewTensor<xpu>(Shape2(label2Size, label2Size), d_zero);
    _eg2_tagBigram2 = NewTensor<xpu>(Shape2(label2Size, label2Size), d_zero);

    random(_tagBigram2, -bound, bound, seed + 1);    

    _delta = delta;
    
    _label1_o = -1;
    _label2_o = -1;    
  }

  inline void initial(Tensor<xpu, 2, dtype> W1, Tensor<xpu, 2, dtype> W2, dtype delta = 1.0) {
    static int label1Size, label2Size;
    label1Size = W1.size(0);
    label2Size = W2.size(0);

    _tagBigram1 = NewTensor<xpu>(Shape2(label1Size, label1Size), d_zero);
    _grad_tagBigram1 = NewTensor<xpu>(Shape2(label1Size, label1Size), d_zero);
    _eg2_tagBigram1 = NewTensor<xpu>(Shape2(label1Size, label1Size), d_zero);
    Copy(_tagBigram1, W1);

    _tagBigram2 = NewTensor<xpu>(Shape2(label2Size, label2Size), d_zero);
    _grad_tagBigram2 = NewTensor<xpu>(Shape2(label2Size, label2Size), d_zero);
    _eg2_tagBigram2 = NewTensor<xpu>(Shape2(label2Size, label2Size), d_zero);
    Copy(_tagBigram2, W2);
    
    _delta = delta;

    _label1_o = -1;
    _label2_o = -1;  
  }
  
  inline void setLabelO(int label1_o, int label2_o){
    _label1_o = label1_o;
    _label2_o = label2_o;  	
  }

  inline void release() {
  	FreeSpace(&_tagBigram1);
    FreeSpace(&_grad_tagBigram1);
    FreeSpace(&_eg2_tagBigram1);
    
    FreeSpace(&_tagBigram2);
    FreeSpace(&_grad_tagBigram2);
    FreeSpace(&_eg2_tagBigram2);
  }

  virtual ~MultiCRFLoss() {
    // TODO Auto-generated destructor stub
  }

  inline dtype squarenormAll() {
    dtype result = squarenorm(_grad_tagBigram1);
    result += squarenorm(_grad_tagBigram2);

    return result;
  }

  inline void scaleGrad(dtype scale) {
    _grad_tagBigram1 = _grad_tagBigram1 * scale;
    _grad_tagBigram2 = _grad_tagBigram2 * scale;
  }

public:

  inline dtype loss(const vector<Tensor<xpu, 2, dtype> > &output1, const vector<Tensor<xpu, 2, dtype> > &output2, 
      const vector<vector<int> > &answer1s, const vector<vector<int> > &answer2s, 
      vector<Tensor<xpu, 2, dtype> > &loutput1, vector<Tensor<xpu, 2, dtype> > &loutput2, 
      Metric & eval1, Metric & eval2, int batchsize = 1) {
    int seq_size = output1.size();
    if (answer1s.size() != seq_size || answer2s.size() != seq_size || output2.size() != seq_size || seq_size == 0) {
      std::cerr << "multi crf_loss error: vector size or context size invalid" << std::endl;
    }

    int label1size = answer1s[0].size();
    int label2size = answer2s[0].size();
    {
	    int dim1 = output1[0].size(0), dim2 = output1[0].size(1);
	    int odim1 = loutput1[0].size(0), odim2 = loutput1[0].size(1);
	    if (label1size != odim2 || dim2 != odim2 || dim1 != 1 || odim1 != 1) {
	      std::cerr << "multi crf_loss error: dim size invalid" << std::endl;
	    }
	  }

    {
	    int dim1 = output2[0].size(0), dim2 = output2[0].size(1);
	    int odim1 = loutput2[0].size(0), odim2 = loutput2[0].size(1);
	    if (label2size != odim2 || dim2 != odim2 || dim1 != 1 || odim1 != 1) {
	      std::cerr << "multi crf_loss error: dim size invalid" << std::endl;
	    }
	  }
	  
    dtype cost = 0.0;
      // viterbi algorithm
      NRVec<int> goldlabel1s(seq_size);
      goldlabel1s = -1;
      for (int idx = 0; idx < seq_size; idx++) {
        for (int i = 0; i < label1size; ++i) {
          if (answer1s[idx][i] == 1) {
            goldlabel1s[idx] = i;
          }
        }
      }

      NRMat<dtype> maxscore1s(seq_size, label1size);
      NRMat<int> maxlastlabel1s(seq_size, label1size);
      dtype gold1Score = 0.0;
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx == 0)
          gold1Score = output1[idx][0][goldlabel1s[idx]];
        else
          gold1Score += output1[idx][0][goldlabel1s[idx]] + _tagBigram1[goldlabel1s[idx - 1]][goldlabel1s[idx]];
        for (int i = 0; i < label1size; ++i) {
          // can be changed with probabilities in future work
          if (idx == 0) {
            maxscore1s[idx][i] = output1[idx][0][i];
            if (goldlabel1s[idx] != i)
              maxscore1s[idx][i] = maxscore1s[idx][i] + _delta;
            maxlastlabel1s[idx][i] = -1;
          } else {
            int maxlastlabel = 0;
            dtype maxscore = _tagBigram1[0][i] + output1[idx][0][i] + maxscore1s[idx - 1][0];
            for (int j = 1; j < label1size; ++j) {
              dtype curscore = _tagBigram1[j][i] + output1[idx][0][i] + maxscore1s[idx - 1][j];
              if (curscore > maxscore) {
                maxlastlabel = j;
                maxscore = curscore;
              }
            }
            maxscore1s[idx][i] = maxscore;
            if (goldlabel1s[idx] != i)
              maxscore1s[idx][i] = maxscore1s[idx][i] + _delta;
            maxlastlabel1s[idx][i] = maxlastlabel;

          }
        }
      }

      NRVec<int> optLabel1s(seq_size);
      optLabel1s = 0;
      dtype max1Score = maxscore1s[seq_size - 1][0];
      for (int i = 1; i < label1size; ++i) {
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
        eval1.overall_label_count++;
        if (optLabel1s[idx] == goldlabel1s[idx]) {
          eval1.correct_label_count++;
        } else {
          bCorrect = false;
        }
      }

      //dtype cur1cost = bCorrect ? 0.0 : max1Score - gold1Score;
      dtype cur1cost = bCorrect ? 0.0 : 1.0;
      cur1cost = cur1cost / batchsize;

      for (int idx = 0; idx < seq_size; idx++) {
        if (goldlabel1s[idx] == -1)
          continue;
        if (optLabel1s[idx] != goldlabel1s[idx]) {
          loutput1[idx][0][optLabel1s[idx]] = cur1cost;
          loutput1[idx][0][goldlabel1s[idx]] = -cur1cost;
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
        for (int i = 0; i < label2size; ++i) {
          if (answer2s[idx][i] == 1) {
            goldlabel2s[idx] = i;
          }
        }
      }

      NRMat<dtype> maxscore2s(seq_size, label2size);
      NRMat<int> maxlastlabel2s(seq_size, label2size);
      dtype gold2Score = 0.0;
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx == 0)
          gold2Score = output2[idx][0][goldlabel2s[idx]];
        else
          gold2Score += output2[idx][0][goldlabel2s[idx]] + _tagBigram2[goldlabel2s[idx - 1]][goldlabel2s[idx]];
        for (int i = 0; i < label2size; ++i) {
          // can be changed with probabilities in future work
          if (idx == 0) {
            maxscore2s[idx][i] = output2[idx][0][i];
            if (goldlabel2s[idx] != i)
              maxscore2s[idx][i] = maxscore2s[idx][i] + _delta;
            maxlastlabel2s[idx][i] = -1;
          } else {
            int maxlastlabel = 0;
            dtype maxscore = _tagBigram2[0][i] + output2[idx][0][i] + maxscore2s[idx - 1][0];
            for (int j = 1; j < label2size; ++j) {
              dtype curscore = _tagBigram2[j][i] + output2[idx][0][i] + maxscore2s[idx - 1][j];
              if (curscore > maxscore) {
                maxlastlabel = j;
                maxscore = curscore;
              }
            }
            maxscore2s[idx][i] = maxscore;
            if (goldlabel2s[idx] != i)
              maxscore2s[idx][i] = maxscore2s[idx][i] + _delta;
            maxlastlabel2s[idx][i] = maxlastlabel;
          }
          if (optLabel1s[idx] == _label1_o && i != _label2_o)
            maxscore2s[idx][i] = -1e+20;
        }

      }

      NRVec<int> optLabel2s(seq_size);
      optLabel2s = 0;
      dtype max2Score = maxscore2s[seq_size - 1][0];
      for (int i = 1; i < label2size; ++i) {
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
        eval2.overall_label_count++;
        if (optLabel2s[idx] == goldlabel2s[idx]) {
          eval2.correct_label_count++;
        } else {
          bCorrect = false;
        }
      }

      //dtype cur2cost = bCorrect ? 0.0 : max2Score - gold2Score;
      dtype cur2cost = bCorrect ? 0.0 : 1.0;

      for (int idx = 0; idx < seq_size; idx++) {
        if (goldlabel2s[idx] == -1)
          continue;
        if (optLabel2s[idx] != goldlabel2s[idx]) {
          loutput2[idx][0][optLabel2s[idx]] = cur2cost;
          loutput2[idx][0][goldlabel2s[idx]] = -cur2cost;
          cost += cur2cost;
        }
        if (idx > 0 && goldlabel2s[idx - 1] >= 0) {
          _grad_tagBigram2[optLabel2s[idx - 1]][optLabel2s[idx]] += cur2cost;
          _grad_tagBigram2[goldlabel2s[idx - 1]][goldlabel2s[idx]] -= cur2cost;
        }
      }

    return cost;

  }


  inline dtype cost(const vector<Tensor<xpu, 2, dtype> > &output1, const vector<Tensor<xpu, 2, dtype> > &output2, 
  const vector<vector<int> > &answer1s, const vector<vector<int> > &answer2s) {
    int seq_size = output1.size();
    if (answer1s.size() != seq_size || answer2s.size() != seq_size || output2.size() != seq_size || seq_size == 0) {
      std::cerr << "multi crf_cost error: vector size or context size invalid" << std::endl;
    }

    int label1size = answer1s[0].size();
    int label2size = answer2s[0].size();
    {
	    int dim1 = output1[0].size(0), dim2 = output1[0].size(1);
	    if (label1size != dim2 || dim1 != 1) {
	      std::cerr << "multi crf_cost error: dim size invalid" << std::endl;
	    }
	  }
    {
	    int dim1 = output2[0].size(0), dim2 = output2[0].size(1);
	    if (label2size != dim2 || dim1 != 1) {
	      std::cerr << "multi crf_cost error: dim size invalid" << std::endl;
	    }
	  }
	  
    dtype cost = 0.0;	  
    // get delta for each output
    // viterbi algorithm
    NRVec<int> goldlabel1s(seq_size);
    goldlabel1s = -1;
    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < label1size; ++i) {
        if (answer1s[idx][i] == 1) {
          goldlabel1s[idx] = i;
        }
      }
    }

    NRMat<dtype> maxscore1s(seq_size, label1size);
    NRMat<int> maxlastlabel1s(seq_size, label1size);
    dtype gold1Score = 0.0;
    for (int idx = 0; idx < seq_size; idx++) {
      if (idx == 0)
        gold1Score = output1[idx][0][goldlabel1s[idx]];
      else
        gold1Score += output1[idx][0][goldlabel1s[idx]] + _tagBigram1[goldlabel1s[idx - 1]][goldlabel1s[idx]];
      for (int i = 0; i < label1size; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          maxscore1s[idx][i] = output1[idx][0][i];
          if (goldlabel1s[idx] != i)
            maxscore1s[idx][i] = maxscore1s[idx][i] + _delta;
          maxlastlabel1s[idx][i] = -1;
        } else {
          int maxlastlabel = 0;
          dtype maxscore = _tagBigram1[0][i] + output1[idx][0][i] + maxscore1s[idx - 1][0];
          for (int j = 1; j < label1size; ++j) {
            dtype curscore = _tagBigram1[j][i] + output1[idx][0][i] + maxscore1s[idx - 1][j];
            if (curscore > maxscore) {
              maxlastlabel = j;
              maxscore = curscore;
            }
          }
          maxscore1s[idx][i] = maxscore;
          if (goldlabel1s[idx] != i)
            maxscore1s[idx][i] = maxscore1s[idx][i] + _delta;
          maxlastlabel1s[idx][i] = maxlastlabel;

        }
      }
    }

    NRVec<int> optLabel1s(seq_size);
    optLabel1s = 0;
    dtype max1Score = maxscore1s[seq_size - 1][0];
    for (int i = 1; i < label1size; ++i) {
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
      if (optLabel1s[idx] == goldlabel1s[idx]) {
      } else {
        bCorrect = false;
      }
    }

    //dtype cur1cost = bCorrect ? 0.0 : max1Score - gold1Score;
    dtype cur1cost = bCorrect ? 0.0 : 1.0;
    cost += cur1cost;

    // viterbi algorithm
    NRVec<int> goldlabel2s(seq_size);
    goldlabel2s = -1;
    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < label2size; ++i) {
        if (answer2s[idx][i] == 1) {
          goldlabel2s[idx] = i;
        }
      }
    }

    NRMat<dtype> maxscore2s(seq_size, label2size);
    NRMat<int> maxlastlabel2s(seq_size, label2size);
    dtype gold2Score = 0.0;
    for (int idx = 0; idx < seq_size; idx++) {
      if (idx == 0)
        gold2Score = output2[idx][0][goldlabel2s[idx]];
      else
        gold2Score += output2[idx][0][goldlabel2s[idx]] + _tagBigram2[goldlabel2s[idx - 1]][goldlabel2s[idx]];
      for (int i = 0; i < label2size; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          maxscore2s[idx][i] = output2[idx][0][i];
          if (goldlabel2s[idx] != i)
            maxscore2s[idx][i] = maxscore2s[idx][i] + _delta;
          maxlastlabel2s[idx][i] = -1;
        } else {
          int maxlastlabel = 0;
          dtype maxscore = _tagBigram2[0][i] + output2[idx][0][i] + maxscore2s[idx - 1][0];
          for (int j = 1; j < label2size; ++j) {
            dtype curscore = _tagBigram2[j][i] + output2[idx][0][i] + maxscore2s[idx - 1][j];
            if (curscore > maxscore) {
              maxlastlabel = j;
              maxscore = curscore;
            }
          }
          maxscore2s[idx][i] = maxscore;
          if (goldlabel2s[idx] != i)
            maxscore2s[idx][i] = maxscore2s[idx][i] + _delta;
          maxlastlabel2s[idx][i] = maxlastlabel;
        }
        if (optLabel1s[idx] == _label1_o && i != _label2_o)
          maxscore2s[idx][i] = -1e+20;
      }

    }

    NRVec<int> optLabel2s(seq_size);
    optLabel2s = 0;
    dtype max2Score = maxscore2s[seq_size - 1][0];
    for (int i = 1; i < label2size; ++i) {
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
      if (optLabel2s[idx] == goldlabel2s[idx]) {
      } else {
        bCorrect = false;
      }
    }

    //dtype cur2cost = bCorrect ? 0.0 : max2Score - gold2Score;
    dtype cur2cost = bCorrect ? 0.0 : 1.0;
    cost += cur2cost;

    return cost;
  }


  inline void predict(const vector<Tensor<xpu, 2, dtype> > &output1, const vector<Tensor<xpu, 2, dtype> > &output2, 
  vector<int>& result1s, vector<int>& result2s) {
    int seq_size = output1.size();
    if (output2.size() != seq_size || seq_size == 0) {
      std::cerr << "multi crf_predict error: vector size or context size invalid" << std::endl;
    }

    int label1size = _tagBigram1.size(0);
    int label2size = _tagBigram2.size(0);
    {
	    int dim1 = output1[0].size(0), dim2 = output1[0].size(1);
	    if (label1size != dim2 || dim1 != 1) {
	      std::cerr << "multi crf_predict error: dim size invalid" << std::endl;
	    }
	  }
    {
	    int dim1 = output2[0].size(0), dim2 = output2[0].size(1);
	    if (label2size != dim2 || dim1 != 1) {
	      std::cerr << "multi crf_predict error: dim size invalid" << std::endl;
	    }
	  }

    // viterbi algorithm
    NRMat<dtype> maxscore1s(seq_size, label1size);
    NRMat<int> maxlastlabel1s(seq_size, label1size);

    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < label1size; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          maxscore1s[idx][i] = output1[idx][0][i];
          maxlastlabel1s[idx][i] = -1;
        } else {
          int maxlastlabel = 0;
          dtype maxscore = _tagBigram1[0][i] + output1[idx][0][i] + maxscore1s[idx - 1][0];
          for (int j = 1; j < label1size; ++j) {
            dtype curscore = _tagBigram1[j][i] + output1[idx][0][i] + maxscore1s[idx - 1][j];
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
    dtype maxFinalScore = maxscore1s[seq_size - 1][0];
    result1s[seq_size - 1] = 0;
    for (int i = 1; i < label1size; ++i) {
      if (maxscore1s[seq_size - 1][i] > maxFinalScore) {
        maxFinalScore = maxscore1s[seq_size - 1][i];
        result1s[seq_size - 1] = i;
      }
    }

    for (int idx = seq_size - 2; idx >= 0; idx--) {
      result1s[idx] = maxlastlabel1s[idx + 1][result1s[idx + 1]];
    }

    // viterbi algorithm
    NRMat<dtype> maxscore2s(seq_size, label2size);
    NRMat<int> maxlastlabel2s(seq_size, label2size);

    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < label2size; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          maxscore2s[idx][i] = output2[idx][0][i];
          maxlastlabel2s[idx][i] = -1;
        } else {
          int maxlastlabel = 0;
          dtype maxscore = _tagBigram2[0][i] + output2[idx][0][i] + maxscore2s[idx - 1][0];
          for (int j = 1; j < label2size; ++j) {
            dtype curscore = _tagBigram2[j][i] + output2[idx][0][i] + maxscore2s[idx - 1][j];
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
    for (int i = 1; i < label2size; ++i) {
      if (maxscore2s[seq_size - 1][i] > maxFinalScore) {
        maxFinalScore = maxscore2s[seq_size - 1][i];
        result2s[seq_size - 1] = i;
      }
    }

    for (int idx = seq_size - 2; idx >= 0; idx--) {
      result2s[idx] = maxlastlabel2s[idx + 1][result2s[idx + 1]];
    }

  }



  inline void randomprint(int num) {
    static int nOSize, nISize;
    int count = 0;
    while (count < num) {
      nOSize = _tagBigram1.size(0);
      nISize = _tagBigram1.size(1);
      int idx = rand() % nOSize;
      int idy = rand() % nISize;

      std::cout << "_tagBigram1[" << idx << "," << idy << "]=" << _tagBigram1[idx][idy] << " ";
      	
      nOSize = _tagBigram2.size(0);
      nISize = _tagBigram2.size(1);
      idx = rand() % nOSize;
      idy = rand() % nISize;

      std::cout << "_tagBigram2[" << idx << "," << idy << "]=" << _tagBigram2[idx][idy] << " ";      	

      count++;
    }

    std::cout << std::endl;
  }

  inline void updateAdaGrad(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
    _grad_tagBigram1 = _grad_tagBigram1 + _tagBigram1 * nnRegular;
    _eg2_tagBigram1 = _eg2_tagBigram1 + _grad_tagBigram1 * _grad_tagBigram1;
    _tagBigram1 = _tagBigram1 - _grad_tagBigram1 * adaAlpha / F<nl_sqrt>(_eg2_tagBigram1 + adaEps);

    _grad_tagBigram2 = _grad_tagBigram2 + _tagBigram2 * nnRegular;
    _eg2_tagBigram2 = _eg2_tagBigram2 + _grad_tagBigram2 * _grad_tagBigram2;
    _tagBigram2 = _tagBigram2 - _grad_tagBigram2 * adaAlpha / F<nl_sqrt>(_eg2_tagBigram2 + adaEps);
    
    clearGrad();
  }

  inline void clearGrad() {
    _grad_tagBigram1 = 0.0;
    _grad_tagBigram2 = 0.0;
  }
};

#endif /* SRC_MultiCRFLoss_H_ */

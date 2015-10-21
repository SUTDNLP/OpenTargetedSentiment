/*
 * SingleCRFLoss.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_SingleCRFLoss_H_
#define SRC_SingleCRFLoss_H_
#include "tensor.h"
#include "MyLib.h"
#include "Utiltensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class SingleCRFLoss {

public:

  Tensor<xpu, 2, dtype> _tagBigram;
  Tensor<xpu, 2, dtype> _grad_tagBigram;
  Tensor<xpu, 2, dtype> _eg2_tagBigram;
  
  int _label_o;

  dtype _delta;


public:
  SingleCRFLoss() {
  }

  inline void initial(int nLabelSize, int seed = 0) {
    //dtype bound = sqrt(6.0 / (nLabelSize + nLabelSize + 1));
    dtype bound = 0.1;

    _tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);
    _grad_tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);
    _eg2_tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);

    random(_tagBigram, -bound, bound, seed);

    _delta = 0.2;
    
    _label_o = -1;
  }


  inline void initial(int nLabelSize, dtype delta, int seed = 0) {
    //dtype bound = sqrt(6.0 / (nLabelSize + nLabelSize + 1));
    dtype bound = 0.1;

    _tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);
    _grad_tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);
    _eg2_tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);

    random(_tagBigram, -bound, bound, seed);

    _delta = delta;
    
    _label_o = -1;
  }

  inline void initial(Tensor<xpu, 2, dtype> W, dtype delta = 0.2) {
    static int nLabelSize;
    nLabelSize = W.size(0);

    _tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);
    _grad_tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);
    _eg2_tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);
    Copy(_tagBigram, W);

    _delta = delta;

    _label_o = -1;
  }
  
  inline void setLabelO(int label_o){
    _label_o = label_o;	
  }  

  inline void release() {
    FreeSpace(&_tagBigram);
    FreeSpace(&_grad_tagBigram);
    FreeSpace(&_eg2_tagBigram);
  }

  virtual ~SingleCRFLoss() {
    // TODO Auto-generated destructor stub
  }

  inline dtype squarenormAll() {
    dtype result = squarenorm(_grad_tagBigram);

    return result;
  }

  inline void scaleGrad(dtype scale) {
    _grad_tagBigram = _grad_tagBigram * scale;
  }

public:

  inline dtype loss(const vector<Tensor<xpu, 2, dtype> > &output, const vector<vector<int> > &answers, vector<Tensor<xpu, 2, dtype> > &loutput,
      Metric & eval, int batchsize = 1) {
    int seq_size = output.size();
    if (answers.size() != seq_size || seq_size == 0) {
      std::cerr << "single crf_loss error: vector size or context size invalid" << std::endl;
    }

    int dim1 = output[0].size(0), dim2 = output[0].size(1);
    int odim1 = loutput[0].size(0), odim2 = loutput[0].size(1);
    int labelsize = answers[0].size();
    if (labelsize != odim2 || dim2 != odim2 || dim1 != 1 || odim1 != 1) {
      std::cerr << "single crf_loss error: dim size invalid" << std::endl;
    }

    dtype cost = 0.0;
      //compute delta
      NRVec<int> goldlabels(seq_size);
      goldlabels = -1;
      for (int idx = 0; idx < seq_size; idx++) {
        for (int i = 0; i < labelsize; ++i) {
          if (answers[idx][i] == 1) {
            goldlabels[idx] = i;
          }
        }
      }

      NRMat<dtype> maxscores(seq_size, labelsize);
      NRMat<int> maxlastlabels(seq_size, labelsize);
      dtype goldScore = 0.0;
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx == 0)
          goldScore = output[idx][0][goldlabels[idx]];
        else
          goldScore += output[idx][0][goldlabels[idx]] + _tagBigram[goldlabels[idx - 1]][goldlabels[idx]];
        for (int i = 0; i < labelsize; ++i) {
          // can be changed with probabilities in future work
          if (idx == 0) {
            maxscores[idx][i] = output[idx][0][i];
            if (goldlabels[idx] != i)
              maxscores[idx][i] = maxscores[idx][i] + _delta;
            maxlastlabels[idx][i] = -1;
          } else {
            int maxlastlabel = 0;
            dtype maxscore = _tagBigram[0][i] + output[idx][0][i] + maxscores[idx - 1][0];
            for (int j = 1; j < labelsize; ++j) {
              dtype curscore = _tagBigram[j][i] + output[idx][0][i] + maxscores[idx - 1][j];
              if (curscore > maxscore) {
                maxlastlabel = j;
                maxscore = curscore;
              }
            }
            maxscores[idx][i] = maxscore;
            if (goldlabels[idx] != i)
              maxscores[idx][i] = maxscores[idx][i] + _delta;
            maxlastlabels[idx][i] = maxlastlabel;
          }
          if (goldlabels[idx] == _label_o && i != _label_o)
            maxscores[idx][i] = -1e+20;
        }

      }

      NRVec<int> optLabels(seq_size);
      optLabels = 0;
      dtype maxScore = maxscores[seq_size - 1][0];
      for (int i = 1; i < labelsize; ++i) {
        if (maxscores[seq_size - 1][i] > maxScore) {
          maxScore = maxscores[seq_size - 1][i];
          optLabels[seq_size - 1] = i;
        }
      }

      for (int idx = seq_size - 2; idx >= 0; idx--) {
        optLabels[idx] = maxlastlabels[idx + 1][optLabels[idx + 1]];
      }

      bool bCorrect = true;
      for (int idx = 0; idx < seq_size; idx++) {
        if (goldlabels[idx] == -1 || goldlabels[idx] == _label_o)
          continue;
        eval.overall_label_count++;
        if (optLabels[idx] == goldlabels[idx]) {
          eval.correct_label_count++;
        } else {
          bCorrect = false;
        }
      }

      //dtype curcost = bCorrect ? 0.0 : maxScore - goldScore;
      dtype curcost = bCorrect ? 0.0 : 1.0;
      curcost = curcost / batchsize;

      for (int idx = 0; idx < seq_size; idx++) {
        if (goldlabels[idx] == -1 || goldlabels[idx] == _label_o)
          continue;
        if (optLabels[idx] != goldlabels[idx]) {
          loutput[idx][0][optLabels[idx]] = curcost;
          loutput[idx][0][goldlabels[idx]] = -curcost;
          cost += curcost;
        }
        if (idx > 0 && goldlabels[idx - 1] >= 0) {
          _grad_tagBigram[optLabels[idx - 1]][optLabels[idx]] += curcost;
          _grad_tagBigram[goldlabels[idx - 1]][goldlabels[idx]] -= curcost;
        }
      }
    return cost;

  }


  inline dtype cost(const vector<Tensor<xpu, 2, dtype> > &output, const vector<vector<int> > &answers) {
    int seq_size = output.size();
    if (answers.size() != seq_size || seq_size == 0) {
      std::cerr << "single crf_cost error: vector size or context size invalid" << std::endl;
    }

    int dim1 = output[0].size(0), dim2 = output[0].size(1);
    int labelsize = answers[0].size();

    if (labelsize != dim2 || dim1 != 1) {
      std::cerr << "single crf_cost error: dim size invalid" << std::endl;
    }

    //compute delta
    NRVec<int> goldlabels(seq_size);
    goldlabels = -1;
    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < labelsize; ++i) {
        if (answers[idx][i] == 1) {
          goldlabels[idx] = i;
        }
      }
    }

    NRMat<dtype> maxscores(seq_size, labelsize);
    NRMat<int> maxlastlabels(seq_size, labelsize);
    dtype goldScore = 0.0;
    for (int idx = 0; idx < seq_size; idx++) {
      if (idx == 0)
        goldScore = output[idx][0][goldlabels[idx]];
      else
        goldScore += output[idx][0][goldlabels[idx]] + _tagBigram[goldlabels[idx - 1]][goldlabels[idx]];
      for (int i = 0; i < labelsize; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          maxscores[idx][i] = output[idx][0][i];
          if (goldlabels[idx] != i)
            maxscores[idx][i] = maxscores[idx][i] + _delta;
          maxlastlabels[idx][i] = -1;
        } else {
          int maxlastlabel = 0;
          dtype maxscore = _tagBigram[0][i] + output[idx][0][i] + maxscores[idx - 1][0];
          for (int j = 1; j < labelsize; ++j) {
            dtype curscore = _tagBigram[j][i] + output[idx][0][i] + maxscores[idx - 1][j];
            if (curscore > maxscore) {
              maxlastlabel = j;
              maxscore = curscore;
            }
          }
          maxscores[idx][i] = maxscore;
          if (goldlabels[idx] != i)
            maxscores[idx][i] = maxscores[idx][i] + _delta;
          maxlastlabels[idx][i] = maxlastlabel;
        }
        if (goldlabels[idx] == _label_o && i != _label_o)
          maxscores[idx][i] = -1e+20;
      }

    }

    NRVec<int> optLabels(seq_size);
    optLabels = 0;
    dtype maxScore = maxscores[seq_size - 1][0];
    for (int i = 1; i < labelsize; ++i) {
      if (maxscores[seq_size - 1][i] > maxScore) {
        maxScore = maxscores[seq_size - 1][i];
        optLabels[seq_size - 1] = i;
      }
    }

    for (int idx = seq_size - 2; idx >= 0; idx--) {
      optLabels[idx] = maxlastlabels[idx + 1][optLabels[idx + 1]];
    }

    bool bCorrect = true;
    for (int idx = 0; idx < seq_size; idx++) {
      if (goldlabels[idx] == -1 || goldlabels[idx] == _label_o)
        continue;
      if (optLabels[idx] == goldlabels[idx]) {
      } else {
        bCorrect = false;
      }
    }

    dtype cost = bCorrect ? 0.0 : 1.0;

    return cost;
  }


  inline void predict(const vector<Tensor<xpu, 2, dtype> > &output, const vector<string>& refer_results, vector<int>& results) {
    int seq_size = output.size();
    if (seq_size == 0 || refer_results.size() != seq_size) {
      std::cerr << "single crf_predict error: vector size or context size invalid" << std::endl;
    }

    int dim1 = output[0].size(0), dim2 = output[0].size(1);
    if (dim1 != 1) {
      std::cerr << "single crf_predict error: dim size invalid" << std::endl;
    }

    int labelsize = _tagBigram.size(0);
    // viterbi algorithm
    NRMat<dtype> maxscores(seq_size,labelsize);
    NRMat<int> maxlastlabels(seq_size,labelsize);

    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < labelsize; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          maxscores[idx][i] = output[idx][0][i];
          maxlastlabels[idx][i] = -1;
        } else {
          int maxlastlabel = 0;
          dtype maxscore = _tagBigram[0][i] + output[idx][0][i] + maxscores[idx - 1][0];
          for (int j = 1; j < labelsize; ++j) {
            dtype curscore = _tagBigram[j][i] + output[idx][0][i] + maxscores[idx - 1][j];
            if (curscore > maxscore) {
              maxlastlabel = j;
              maxscore = curscore;
            }
          }
          maxscores[idx][i] = maxscore;
          maxlastlabels[idx][i] = maxlastlabel;
        }
        if(refer_results[idx] == "o" && i != _label_o) maxscores[idx][i] = -1e+20;
      }
    }

    results.resize(seq_size);
    dtype maxFinalScore = maxscores[seq_size - 1][0];
    results[seq_size - 1] = 0;
    for (int i = 1; i < labelsize; ++i) {
      if (maxscores[seq_size - 1][i] > maxFinalScore) {
        maxFinalScore = maxscores[seq_size - 1][i];
        results[seq_size - 1] = i;
      }
    }

    for (int idx = seq_size - 2; idx >= 0; idx--) {
      results[idx] = maxlastlabels[idx + 1][results[idx + 1]];
    }
  }

  inline void randomprint(int num) {
    static int nOSize, nISize;
    nOSize = _tagBigram.size(0);
    nISize = _tagBigram.size(1);
    int count = 0;
    while (count < num) {
      int idx = rand() % nOSize;
      int idy = rand() % nISize;

      std::cout << "_tagBigram[" << idx << "," << idy << "]=" << _tagBigram[idx][idy] << " ";

      count++;
    }

    std::cout << std::endl;
  }

  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _grad_tagBigram = _grad_tagBigram + _tagBigram * regularizationWeight;
    _eg2_tagBigram = _eg2_tagBigram + _grad_tagBigram * _grad_tagBigram;
    _tagBigram = _tagBigram - _grad_tagBigram * adaAlpha / F<nl_sqrt>(_eg2_tagBigram + adaEps);


    clearGrad();
  }

  inline void clearGrad() {
    _grad_tagBigram = 0.0;
  }
};

#endif /* SRC_SingleCRFLoss_H_ */

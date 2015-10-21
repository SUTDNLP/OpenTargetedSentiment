#ifndef _JST_INSTANCE_
#define _JST_INSTANCE_

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include "MyLib.h"
#include "Metric.h"

using namespace std;

class Instance {
public:
  Instance() {
  }
  ~Instance() {
  }

  int size() const {
    return words.size();
  }

  void clear() {
    label1s.clear();
    label2s.clear();
    words.clear();
    for (int i = 0; i < size(); i++) {
      sparsefeatures[i].clear();
    }
    sparsefeatures.clear();
  }

  void allocate(int length) {
    clear();
    label1s.resize(length);
    label2s.resize(length);
    words.resize(length);
    sparsefeatures.resize(length);
  }

  void copyValuesFrom(const Instance& anInstance) {
    allocate(anInstance.size());
    for (int i = 0; i < anInstance.size(); i++) {
      label1s[i] = anInstance.label1s[i];
      label2s[i] = anInstance.label2s[i];
      words[i] = anInstance.words[i];
      for (int j = 0; j < anInstance.sparsefeatures[i].size(); j++) {
        sparsefeatures[i].push_back(anInstance.sparsefeatures[i][j]);
      }
    }

  }

  void assignLabel(const vector<string>& resulted_label1s, const vector<string>& resulted_label2s) {
    assert(resulted_label1s.size() == words.size());
    assert(resulted_label2s.size() == words.size());
    label1s.clear();
    label2s.clear();

    for (int idx = 0; idx < resulted_label1s.size(); idx++) {
      label1s.push_back(resulted_label1s[idx]);
      label2s.push_back(resulted_label2s[idx]);
    }
  }

  void Evaluate(const vector<string>& resulted_label1s, const vector<string>& resulted_label2s, Metric& eval1, Metric& eval2) const {
    for (int idx = 0; idx < label1s.size(); idx++) {
      if (!validlabels(label1s[idx]))
        continue;
      if (resulted_label1s[idx].compare(label1s[idx]) == 0)
        eval1.correct_label_count++;
      eval1.overall_label_count++;
    }
    for (int idx = 0; idx < label2s.size(); idx++) {
      if (!validlabels(label2s[idx]))
        continue;
      if (resulted_label2s[idx].compare(label2s[idx]) == 0)
        eval2.correct_label_count++;
      eval2.overall_label_count++;
    }
  }

  void SegEvaluate(const vector<string>& resulted_label1s, const vector<string>& resulted_label2s, Metric& eval1, Metric& eval2) const {
    static int idx, idy, endpos;
    hash_set<string> gold1s, gold2s;
    // segmentation should be agree in both layers, usually, the first layer defines segmentation
    idx = 0;
    while (idx < label1s.size()) {
      if (is_start_label(label1s[idx])) {
        idy = idx;
        endpos = -1;
        while (idy < label1s.size()) {
          if (!is_continue_label(label1s[idy], label1s[idx], idy - idx)) {
            endpos = idy - 1;
            break;
          }
          endpos = idy;
          idy++;
        }
        stringstream ss;
        ss << "[" << idx << "," << endpos << "]";
        gold1s.insert(cleanLabel(label1s[idx]) + ss.str());
        idx = endpos;
      }
      idx++;
    }

    idx = 0;
    while (idx < label2s.size()) {
      if (is_start_label(label2s[idx])) {
        idy = idx;
        endpos = -1;
        while (idy < label2s.size()) {
          if (!is_continue_label(label2s[idy], label2s[idx], idy - idx)) {
            endpos = idy - 1;
            break;
          }
          endpos = idy;
          idy++;
        }
        stringstream ss;
        ss << "[" << idx << "," << endpos << "]";
        gold2s.insert(cleanLabel(label2s[idx]) + ss.str());
        idx = endpos;
      }
      idx++;
    }

    hash_set<string> pred1s, pred2s;
    idx = 0;
    while (idx < resulted_label1s.size()) {
      if (is_start_label(resulted_label1s[idx])) {
        stringstream ss;
        idy = idx;
        endpos = -1;
        while (idy < resulted_label1s.size()) {
          if (!is_continue_label(resulted_label1s[idy], resulted_label1s[idx], idy - idx)) {
            endpos = idy - 1;
            break;
          }
          endpos = idy;
          idy++;
        }
        ss << "[" << idx << "," << endpos << "]";
        pred1s.insert(cleanLabel(resulted_label1s[idx]) + ss.str());
        idx = endpos;
      }
      idx++;
    }

    idx = 0;
    while (idx < resulted_label2s.size()) {
      if (is_start_label(resulted_label2s[idx])) {
        stringstream ss;
        idy = idx;
        endpos = -1;
        while (idy < resulted_label2s.size()) {
          if (!is_continue_label(resulted_label2s[idy], resulted_label2s[idx], idy - idx)) {
            endpos = idy - 1;
            break;
          }
          endpos = idy;
          idy++;
        }
        ss << "[" << idx << "," << endpos << "]";
        pred2s.insert(cleanLabel(resulted_label2s[idx]) + ss.str());
        idx = endpos;
      }
      idx++;
    }

    hash_set<string>::iterator iter;

    eval1.overall_label_count += gold1s.size();
    eval1.predicated_label_count += pred1s.size();
    for (iter = pred1s.begin(); iter != pred1s.end(); iter++) {
      if (gold1s.find(*iter) != gold1s.end()) {
        eval1.correct_label_count++;
      }
    }

    eval2.overall_label_count += gold2s.size();
    eval2.predicated_label_count += pred2s.size();
    for (iter = pred2s.begin(); iter != pred2s.end(); iter++) {
      if (gold2s.find(*iter) != gold2s.end()) {
        eval2.correct_label_count++;
      }
    }
  }

public:
  vector<string> label1s;
  vector<string> label2s;
  vector<string> words;
  vector<vector<string> > sparsefeatures;
};

#endif


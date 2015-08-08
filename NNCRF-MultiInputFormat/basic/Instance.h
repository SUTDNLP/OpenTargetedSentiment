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
      charfeatures[i].clear();
    }
    sparsefeatures.clear();
    charfeatures.clear();
  }

  void allocate(int length) {
    clear();
    label1s.resize(length);
    label2s.resize(length);
    words.resize(length);
    sparsefeatures.resize(length);
    charfeatures.resize(length);
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
      for (int j = 0; j < anInstance.charfeatures[i].size(); j++) {
        charfeatures[i].push_back(anInstance.charfeatures[i][j]);
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
    hash_set<string> golds1, golds2;
    // segmentation should be agree in both layers, usually, the first layer defines segmentation
    int idx = 0;
    while (idx < label1s.size()) {
      if (is_start_label(label1s[idx])) {
        stringstream ss;
        int idy = idx;
        while (idy < label1s.size()) {
          if (is_end_label(label1s[idy], idy == label1s.size() - 1 ? "o" : label1s[idy + 1])) {
            break;
          }
          idy++;
        }
        ss << "[" << idx << "," << idy << "]";
        string cleanl = cleanLabel(label1s[idx]);
        //if(cleanl.compare("neutral") != 0)
        golds1.insert(cleanl + ss.str());
        idx = idy + 1;
      } else {
        idx++;
      }
    }

    idx = 0;
    while (idx < label2s.size()) {
      if (is_start_label(label2s[idx])) {
        stringstream ss;
        int idy = idx;
        while (idy < label2s.size()) {
          if (is_end_label(label2s[idy], idy == label2s.size() - 1 ? "o" : label2s[idy + 1])) {
            break;
          }
          idy++;
        }
        ss << "[" << idx << "," << idy << "]";
        string cleanl = cleanLabel(label2s[idx]);
        //if(cleanl.compare("neutral") != 0)
        golds2.insert(cleanl + ss.str());
        idx = idy + 1;
      } else {
        idx++;
      }
    }

    hash_set<string> preds1, preds2;
    idx = 0;
    while (idx < resulted_label1s.size()) {
      if (is_start_label(resulted_label1s[idx])) {
        stringstream ss;
        int idy = idx;
        while (idy < resulted_label1s.size()) {
          if (is_end_label(resulted_label1s[idy], idy == resulted_label1s.size() - 1 ? "o" : resulted_label1s[idy + 1])) {
            break;
          }
          idy++;
        }
        ss << "[" << idx << "," << idy << "]";
        string cleanl = cleanLabel(resulted_label1s[idx]);
        //if(cleanl.compare("neutral") != 0)
        preds1.insert(cleanl + ss.str());
        idx = idy + 1;
      } else {
        idx++;
      }
    }

    idx = 0;
    while (idx < resulted_label2s.size()) {
      if (is_start_label(resulted_label2s[idx])) {
        stringstream ss;
        int idy = idx;
        while (idy < resulted_label2s.size()) {
          if (is_end_label(resulted_label2s[idy], idy == resulted_label2s.size() - 1 ? "o" : resulted_label2s[idy + 1])) {
            break;
          }
          idy++;
        }
        ss << "[" << idx << "," << idy << "]";
        string cleanl = cleanLabel(resulted_label2s[idx]);
        //if(cleanl.compare("neutral") != 0)
        preds2.insert(cleanl + ss.str());
        idx = idy + 1;
      } else {
        idx++;
      }
    }

    hash_set<string>::iterator iter;

    eval1.overall_label_count += golds1.size();
    eval1.predicated_label_count += preds1.size();
    for (iter = preds1.begin(); iter != preds1.end(); iter++) {
      if (golds1.find(*iter) != golds1.end()) {
        eval1.correct_label_count++;
      }
    }

    eval2.overall_label_count += golds2.size();
    eval2.predicated_label_count += preds2.size();
    for (iter = preds2.begin(); iter != preds2.end(); iter++) {
      if (golds2.find(*iter) != golds2.end()) {
        eval2.correct_label_count++;
      }
    }
  }

public:
  vector<string> label1s;
  vector<string> label2s;
  vector<string> words;
  vector<vector<string> > sparsefeatures;
  vector<vector<string> > charfeatures;

};

#endif


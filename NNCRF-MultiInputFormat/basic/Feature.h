/*
 * Feature.h
 *
 *  Created on: Mar 17, 2015
 *      Author: mszhang
 */

#ifndef SRC_FEATURE_H_
#define SRC_FEATURE_H_

#include <vector>

using namespace std;
class Feature {

public:
  vector<int> ngram_words;
  vector<vector<int> > ngram_chars;
  vector<int> linear_features;

public:
  Feature() {
  }
  virtual ~Feature() {

  }

  void clear() {
    ngram_words.clear();
    ngram_chars.clear();
    linear_features.clear();
  }
};

#endif /* SRC_FEATURE_H_ */

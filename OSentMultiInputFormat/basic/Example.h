/*
 * Example.h
 *
 *  Created on: Mar 17, 2015
 *      Author: mszhang
 */

#ifndef SRC_EXAMPLE_H_
#define SRC_EXAMPLE_H_

#include "Feature.h"

using namespace std;

class Example {

public:
  vector<vector<int> > m_label2s;
  vector<vector<int> > m_label1s;
  vector<Feature> m_features;

public:
  Example() {

  }
  virtual ~Example() {

  }

  void clear() {
    m_label1s.clear();
    m_label2s.clear();
    m_features.clear();
  }

};

#endif /* SRC_EXAMPLE_H_ */

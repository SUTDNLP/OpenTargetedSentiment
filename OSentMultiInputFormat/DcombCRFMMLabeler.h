/*
 * Labeler.h
 *
 *  Created on: Mar 16, 2015
 *      Author: mszhang
 */

#ifndef SRC_DcombCRFMMLabeler_H_
#define SRC_DcombCRFMMLabeler_H_


#include "N3L.h"
#include "basic/DcombCRFMMClassifier.h"
#include "Options.h"
#include "Instance.h"
#include "Example.h"


#include "Pipe.h"
#include "Utf.h"

using namespace nr;
using namespace std;

class Labeler {
public:
  std::string nullkey;
  std::string unknownkey;
  std::string seperateKey;

public:
  Alphabet m_featAlphabet;
  Alphabet m_label2Alphabet;
  Alphabet m_wordAlphabet;


public:
  Options m_options;
  Pipe m_pipe;

#if USE_CUDA==1
  DcombCRFMMClassifier<gpu> m_classifier;
#else
  DcombCRFMMClassifier<cpu> m_classifier;
#endif

public:
  void readWordEmbeddings(const string& inFile, NRMat<dtype>& wordEmb);

public:
  Labeler();
  virtual ~Labeler();

public:

  int createAlphabet(const vector<Instance>& vecInsts);

  int addTestWordAlpha(const vector<Instance>& vecInsts);

  void extractLinearFeatures(vector<string>& features, const Instance* pInstance, int idx);
  void extractFeature(Feature& feat, const Instance* pInstance, int idx);

  void convert2Example(const Instance* pInstance, Example& exam);
  void initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams);

public:
  void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile, const string& wordEmbFile, const string& charEmbFile);
  int predict(const vector<Feature>& features, const vector<string>& output1s, vector<string>& output2s, const vector<string>& words);
  void test(const string& testFile, const string& outputFile, const string& modelFile);

  void writeModelFile(const string& outputModelFile);
  void loadModelFile(const string& inputModelFile);

};

#endif /* SRC_DcombCRFMMLabeler_H_ */

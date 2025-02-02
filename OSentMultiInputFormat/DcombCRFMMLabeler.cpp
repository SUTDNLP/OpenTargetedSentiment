/*
 * Labeler.cpp
 *
 *  Created on: Mar 16, 2015
 *      Author: mszhang
 */

#include "Argument_helper.h"
#include "DcombCRFMMLabeler.h"

Labeler::Labeler() {
  // TODO Auto-generated constructor stub
  nullkey = "-null-";
  unknownkey = "-unknown-";
  seperateKey = "#";

}

Labeler::~Labeler() {
  // TODO Auto-generated destructor stub
}

int Labeler::createAlphabet(const vector<Instance>& vecInsts) {
  cout << "Creating Alphabet..." << endl;

  int numInstance, labelId1, labelId2;
  hash_map<string, int> feature_stat;
  hash_map<string, int> word_stat;
  m_label2Alphabet.clear();

  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];

    const vector<string> &words = pInstance->words;
    const vector<string> &label1s = pInstance->label1s;
    const vector<string> &label2s = pInstance->label2s;
    const vector<vector<string> > &sparsefeatures = pInstance->sparsefeatures;

    vector<string> features;
    int curInstSize = label1s.size();

    for (int i = 0; i < curInstSize; ++i) {
      if (label2s[i].length() > 2) {
        labelId2 = m_label2Alphabet.from_string(label2s[i].substr(2));
      } else {
        labelId2 = m_label2Alphabet.from_string(label2s[i]);
      }

      string curword = normalize_to_lowerwithdigit(words[i]);
      word_stat[curword]++;

      for (int j = 0; j < sparsefeatures[i].size(); j++)
        feature_stat[sparsefeatures[i][j]]++;
    }

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  cout << numInstance << " " << endl;
  cout << "Label2 num: " << m_label2Alphabet.size() << endl;
  cout << "Total word num: " << word_stat.size() << endl;
  cout << "Total feature num: " << feature_stat.size() << endl;

  m_featAlphabet.clear();
  m_wordAlphabet.clear();
  m_wordAlphabet.from_string(nullkey);
  m_wordAlphabet.from_string(unknownkey);

  hash_map<string, int>::iterator feat_iter;
  for (feat_iter = feature_stat.begin(); feat_iter != feature_stat.end(); feat_iter++) {
    if (feat_iter->second > m_options.featCutOff) {
      m_featAlphabet.from_string(feat_iter->first);
    }
  }

  for (feat_iter = word_stat.begin(); feat_iter != word_stat.end(); feat_iter++) {
    if (!m_options.wordEmbFineTune || feat_iter->second > m_options.wordCutOff) {
      m_wordAlphabet.from_string(feat_iter->first);
    }
  }

  cout << "Remain feature num: " << m_featAlphabet.size() << endl;
  cout << "Remain words num: " << m_wordAlphabet.size() << endl;

  m_label2Alphabet.set_fixed_flag(true);
  m_featAlphabet.set_fixed_flag(true);
  m_wordAlphabet.set_fixed_flag(true);

  return 0;
}

int Labeler::addTestWordAlpha(const vector<Instance>& vecInsts) {
  cout << "Adding word Alphabet..." << endl;

  int numInstance;
  hash_map<string, int> word_stat;
  m_wordAlphabet.set_fixed_flag(false);

  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];

    const vector<string> &words = pInstance->words;

    int curInstSize = words.size();
    for (int i = 0; i < curInstSize; ++i) {
      string curword = normalize_to_lowerwithdigit(words[i]);
      word_stat[curword]++;
    }

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  hash_map<string, int>::iterator feat_iter;
  for (feat_iter = word_stat.begin(); feat_iter != word_stat.end(); feat_iter++) {
    if (!m_options.wordEmbFineTune || feat_iter->second > m_options.wordCutOff) {
      m_wordAlphabet.from_string(feat_iter->first);
    }
  }

  m_wordAlphabet.set_fixed_flag(true);

  return 0;
}


void Labeler::extractFeature(Feature& feat, const Instance* pInstance, int idx) {
  feat.clear();

  const vector<string>& words = pInstance->words;
  int sentsize = words.size();

  string curWord = idx >= 0 && idx < sentsize ? normalize_to_lowerwithdigit(words[idx]) : nullkey;

  // words
  int unknownId = m_wordAlphabet.from_string(unknownkey);

  int curWordId = m_wordAlphabet.from_string(curWord);
  if (curWordId >= 0)
    feat.words.push_back(curWordId);
  else
    feat.words.push_back(unknownId);

  const vector<string>& linear_features = pInstance->sparsefeatures[idx];
  for (int i = 0; i < linear_features.size(); i++) {
    int curFeatId = m_featAlphabet.from_string(linear_features[i]);
    if (curFeatId >= 0)
      feat.linear_features.push_back(curFeatId);
  }

}

void Labeler::convert2Example(const Instance* pInstance, Example& exam) {
  exam.clear();
  const vector<string> &label1s = pInstance->label1s;
  const vector<string> &label2s = pInstance->label2s;
  int curInstSize = label1s.size();
  for (int i = 0; i < curInstSize; ++i) {
    string orcale1 = label1s[i];
    string orcale2 = label2s[i];
    if (orcale2.length() > 2)
      orcale2 = orcale2.substr(2);

    int numLabel2s = m_label2Alphabet.size();
    vector<int> curlabel2s;
    for (int j = 0; j < numLabel2s; ++j) {
      string str = m_label2Alphabet.from_id(j);
      if (str.compare(orcale2) == 0)
        curlabel2s.push_back(1);
      else
        curlabel2s.push_back(0);
    }

    exam.m_label2s.push_back(curlabel2s);

    Feature feat;
    extractFeature(feat, pInstance, i);
    exam.m_features.push_back(feat);
  }

}

void Labeler::initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams) {
  int numInstance;
  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];
    Example curExam;
    convert2Example(pInstance, curExam);
    vecExams.push_back(curExam);

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  cout << numInstance << " " << endl;
}

void Labeler::train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile,
    const string& wordEmbFile, const string& charEmbFile) {
  if (optionFile != "")
    m_options.load(optionFile);

  m_options.showOptions();

  vector<Instance> trainInsts, devInsts, testInsts;
  static vector<Instance> decodeInstResults;
  static Instance curDecodeInst;
  bool bCurIterBetter = false;

  m_pipe.readInstances(trainFile, trainInsts, m_options.maxInstance);
  if (devFile != "")
    m_pipe.readInstances(devFile, devInsts, m_options.maxInstance);
  if (testFile != "")
    m_pipe.readInstances(testFile, testInsts, m_options.maxInstance);

  //Ensure that each file in m_options.testFiles exists!
  vector<vector<Instance> > otherInsts(m_options.testFiles.size());
  for (int idx = 0; idx < m_options.testFiles.size(); idx++) {
    m_pipe.readInstances(m_options.testFiles[idx], otherInsts[idx], m_options.maxInstance);
  }

  //std::cout << "Training example number: " << trainInsts.size() << std::endl;
  //std::cout << "Dev example number: " << trainInsts.size() << std::endl;
  //std::cout << "Test example number: " << trainInsts.size() << std::endl;

  createAlphabet(trainInsts);

  if (!m_options.wordEmbFineTune) {
    addTestWordAlpha(devInsts);
    addTestWordAlpha(testInsts);
    for (int idx = 0; idx < otherInsts.size(); idx++) {
      addTestWordAlpha(otherInsts[idx]);
    }
    cout << "Remain words num: " << m_wordAlphabet.size() << endl;
  }


  NRMat<dtype> wordEmb;
  if (wordEmbFile != "") {
    readWordEmbeddings(wordEmbFile, wordEmb);
  } else {
    wordEmb.resize(m_wordAlphabet.size(), m_options.wordEmbSize);
    wordEmb.randu(1000);
  }


  m_classifier.setWordEmbFineTune(m_options.wordEmbFineTune);
  //m_classifier.setCharEmbFinetune(m_options.charEmbFineTune);
  m_classifier.init(wordEmb, m_options.wordcontext, m_label2Alphabet.size(), m_options.hiddenSize, m_options.linearHiddenSize, m_featAlphabet.size());

  m_classifier.setDropValue(m_options.dropProb);
  m_classifier.setLabel2O(m_label2Alphabet.from_string("o"));

  vector<Example> trainExamples, devExamples, testExamples;
  initialExamples(trainInsts, trainExamples);
  initialExamples(devInsts, devExamples);
  initialExamples(testInsts, testExamples);

  vector<int> otherInstNums(otherInsts.size());
  vector<vector<Example> > otherExamples(otherInsts.size());
  for (int idx = 0; idx < otherInsts.size(); idx++) {
    initialExamples(otherInsts[idx], otherExamples[idx]);
    otherInstNums[idx] = otherExamples[idx].size();
  }

  dtype bestDIS = 0;

  int inputSize = trainExamples.size();

  int batchBlock = inputSize / m_options.batchSize;
  if (inputSize % m_options.batchSize != 0)
    batchBlock++;

  srand(0);
  std::vector<int> indexes;
  for (int i = 0; i < inputSize; ++i)
    indexes.push_back(i);

  static Metric eval1, metric1_dev, metric1_test, eval2, metric2_dev, metric2_test;
  static vector<Example> subExamples;
  int devNum = devExamples.size(), testNum = testExamples.size();

  int maxIter = m_options.maxIter;
  if (m_options.batchSize > 1)
    maxIter = m_options.maxIter * (inputSize / m_options.batchSize + 1);

  for (int iter = 0; iter < m_options.maxIter; ++iter) {
    std::cout << "##### Iteration " << iter << std::endl;

    random_shuffle(indexes.begin(), indexes.end());
    eval1.reset(); eval2.reset();
    if (m_options.batchSize == 1) {
      random_shuffle(indexes.begin(), indexes.end());
      for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
        subExamples.clear();
        int start_pos = updateIter;
        int end_pos = (updateIter + 1);
        if (end_pos > inputSize)
          end_pos = inputSize;

        for (int idy = start_pos; idy < end_pos; idy++) {
          subExamples.push_back(trainExamples[indexes[idy]]);
        }

        int curUpdateIter = iter * batchBlock + updateIter;
        int cost = m_classifier.process(subExamples, curUpdateIter);


        eval2.overall_label_count += m_classifier._eval2.overall_label_count;
        eval2.correct_label_count += m_classifier._eval2.correct_label_count;

        if ((curUpdateIter + 1) % m_options.verboseIter == 0) {
          //m_classifier.checkgrads(subExamples, curUpdateIter+1);
          std::cout << "current: " << updateIter + 1 << ", total block: " << batchBlock << std::endl;
          std::cout << "Cost = " << cost <<  ", Sent Correct(%) = " << eval2.getAccuracy() << std::endl;
        }
        m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);
      }
    } else {
      for (int updateIter = 0; updateIter < m_options.verboseIter; updateIter++) {
        random_shuffle(indexes.begin(), indexes.end());
        subExamples.clear();
        for (int idy = 0; idy < m_options.batchSize; idy++) {
          subExamples.push_back(trainExamples[indexes[idy]]);
        }
        int curUpdateIter = iter * batchBlock + updateIter;
        int cost = m_classifier.process(subExamples, curUpdateIter);

        eval2.overall_label_count += m_classifier._eval2.overall_label_count;
        eval2.correct_label_count += m_classifier._eval2.correct_label_count;

        m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);
      }
      std::cout << "current: " << iter + 1 << ", total block: " << batchBlock << std::endl;
      std::cout << "Sent Correct(%) = " << eval2.getAccuracy() << std::endl;

    }

    if (devNum > 0) {
      bCurIterBetter = false;
      if (!m_options.outBest.empty())
        decodeInstResults.clear();
      metric1_dev.reset();  metric2_dev.reset();
      for (int idx = 0; idx < devExamples.size(); idx++) {
        vector<string> result_label2s;
        int ret = predict(devExamples[idx].m_features, devInsts[idx].label1s, result_label2s, devInsts[idx].words);
        if (m_options.seg)
          devInsts[idx].SegEvaluate(devInsts[idx].label1s, result_label2s, metric1_dev, metric2_dev);
        else
          devInsts[idx].Evaluate(devInsts[idx].label1s, result_label2s, metric1_dev, metric2_dev);

        if (!m_options.outBest.empty()) {
          curDecodeInst.copyValuesFrom(devInsts[idx]);
          curDecodeInst.assignLabel(devInsts[idx].label1s, result_label2s);
          decodeInstResults.push_back(curDecodeInst);
        }
      }
      std::cout << "dev:" << std::endl;
      metric1_dev.print(); metric2_dev.print();

      if (!m_options.outBest.empty() && metric2_dev.getAccuracy() > bestDIS) {
        m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
        bCurIterBetter = true;
      }

      if (testNum > 0) {
        if (!m_options.outBest.empty())
          decodeInstResults.clear();
        metric1_test.reset(); metric2_test.reset();
        for (int idx = 0; idx < testExamples.size(); idx++) {
          vector<string> result_label2s;
          predict(testExamples[idx].m_features, testInsts[idx].label1s, result_label2s, testInsts[idx].words);

          if (m_options.seg)
            testInsts[idx].SegEvaluate(testInsts[idx].label1s, result_label2s, metric1_test, metric2_test);
          else
            testInsts[idx].Evaluate(testInsts[idx].label1s, result_label2s, metric1_test, metric2_test);

          if (bCurIterBetter && !m_options.outBest.empty()) {
            curDecodeInst.copyValuesFrom(testInsts[idx]);
            curDecodeInst.assignLabel(testInsts[idx].label1s, result_label2s);
            decodeInstResults.push_back(curDecodeInst);
          }
        }
        std::cout << "test:" << std::endl;
        metric1_test.print(); metric2_test.print();

        if (!m_options.outBest.empty() && bCurIterBetter) {
          m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
        }
      }

      for (int idx = 0; idx < otherExamples.size(); idx++) {
        std::cout << "processing " << m_options.testFiles[idx] << std::endl;
        if (!m_options.outBest.empty())
          decodeInstResults.clear();
        metric1_test.reset();
        metric2_test.reset();
        for (int idy = 0; idy < otherExamples[idx].size(); idy++) {
          vector<string> result_label2s;
          predict(otherExamples[idx][idy].m_features, otherInsts[idx][idy].label1s, result_label2s, otherInsts[idx][idy].words);

          if (m_options.seg)
            otherInsts[idx][idy].SegEvaluate(otherInsts[idx][idy].label1s, result_label2s, metric1_test, metric2_test);
          else
            otherInsts[idx][idy].Evaluate(otherInsts[idx][idy].label1s, result_label2s, metric1_test, metric2_test);

          if (bCurIterBetter && !m_options.outBest.empty()) {
            curDecodeInst.copyValuesFrom(otherInsts[idx][idy]);
            curDecodeInst.assignLabel(otherInsts[idx][idy].label1s, result_label2s);
            decodeInstResults.push_back(curDecodeInst);
          }
        }
        std::cout << "test:" << std::endl;
        metric1_test.print(); metric2_test.print();

        if (!m_options.outBest.empty() && bCurIterBetter) {
          m_pipe.outputAllInstances(m_options.testFiles[idx] + m_options.outBest, decodeInstResults);
        }
      }

      if (m_options.saveIntermediate && metric2_dev.getAccuracy() > bestDIS) {
        if (metric2_dev.getAccuracy() > bestDIS) {
          std::cout << "Exceeds best previous performance of " << bestDIS << ". Saving model file.." << std::endl;
          bestDIS = metric2_dev.getAccuracy();
        }
        writeModelFile(modelFile);
      }

    }
    // Clear gradients
  }
}

int Labeler::predict(const vector<Feature>& features, const vector<string>& output1s, vector<string>& output2s, const vector<string>& words) {
  assert(features.size() == words.size());
  vector<int> label2Idx;
  m_classifier.predict(features, output1s, label2Idx);
  output2s.clear();

  for (int idx = 0; idx < words.size(); idx++) {
    output2s.push_back("o");
  }

  bool bModified = false;
  int id_start = 0;
  int optlabel = -1;
  int numLabel2s = m_label2Alphabet.size();
  int neutralId = m_label2Alphabet.from_string("neutral");
  NRVec<int> label2count(numLabel2s);
  while (id_start < output1s.size()) {
    if (is_start_label(output1s[id_start])) {
      //stringstream ss;
      int id_end = -1;
      int id_current = id_start;
      while (id_current < output1s.size()) {
        if (!is_continue_label(output1s[id_current], output1s[id_start], id_current - id_start)) {
        	id_end = id_current - 1;
        	break;
        }
        id_end = id_current;
        id_current++;
      }
      //ss << "[" << id_start << "," << id_end << "]";
      label2count = 0;
      for(int idx = id_start; idx <= id_end; idx++)
      {
        label2count[label2Idx[idx]]++;
      }

      //int firstlabelId = label2Idx[id_start];
      //int lastlabelId = label2Idx[id_end];
      //optlabel = label2count[firstlabelId] >= label2count[lastlabelId] ? firstlabelId : lastlabelId;
      optlabel = -1;
      for(int idx = 0; idx < numLabel2s; idx ++)
      {
        if(idx == neutralId)continue;
        if(optlabel == -1 || label2count[idx] > label2count[optlabel])
        {
          optlabel = idx;
        }
      }

      if(label2count[optlabel] == 0)
      {
        optlabel = neutralId;
      }

      output2s[id_start] = "b-" + m_label2Alphabet.from_id(optlabel);
      //std::cout << output2s[id_start] << std::endl;
      for(int idx = id_start + 1; idx <= id_end; idx++)
      {
        output2s[idx] = "i-" + m_label2Alphabet.from_id(optlabel);
      }

      id_start = id_end + 1;
    } else {
      id_start++;
    }
  }

  if (bModified)
    return 1;

  return 0;
}

void Labeler::test(const string& testFile, const string& outputFile, const string& modelFile) {
  loadModelFile(modelFile);
  vector<Instance> testInsts;
  m_pipe.readInstances(testFile, testInsts);

  vector<Example> testExamples;
  initialExamples(testInsts, testExamples);

  int testNum = testExamples.size();
  vector<Instance> testInstResults;
  Metric metric1_test, metric2_test;
  metric1_test.reset();
  metric2_test.reset();
  for (int idx = 0; idx < testExamples.size(); idx++) {
    vector<string> result_label1s, result_label2s;
    predict(testExamples[idx].m_features, result_label1s, result_label2s, testInsts[idx].words);
    testInsts[idx].SegEvaluate(result_label1s, result_label2s, metric1_test, metric2_test);
    Instance curResultInst;
    curResultInst.copyValuesFrom(testInsts[idx]);
    testInstResults.push_back(curResultInst);
  }
  std::cout << "test:" << std::endl;
  metric1_test.print();
  metric2_test.print();

  m_pipe.outputAllInstances(outputFile, testInstResults);

}

void Labeler::readWordEmbeddings(const string& inFile, NRMat<dtype>& wordEmb) {
  static ifstream inf;
  if (inf.is_open()) {
    inf.close();
    inf.clear();
  }
  inf.open(inFile.c_str());

  static string strLine, curWord;
  static int wordId;

  //find the first line, decide the wordDim;
  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (!strLine.empty())
      break;
  }

  int unknownId = m_wordAlphabet.from_string(unknownkey);

  static vector<string> vecInfo;
  split_bychar(strLine, vecInfo, ' ');
  int wordDim = vecInfo.size() - 1;

  std::cout << "word embedding dim is " << wordDim << std::endl;
  m_options.wordEmbSize = wordDim;

  wordEmb.resize(m_wordAlphabet.size(), wordDim);
  wordEmb = 0.0;
  curWord = normalize_to_lowerwithdigit(vecInfo[0]);
  wordId = m_wordAlphabet.from_string(curWord);
  hash_set<int> indexers;
  dtype sum[wordDim];
  int count = 0;
  bool bHasUnknown = false;
  if (wordId >= 0) {
    count++;
    if (unknownId == wordId)
      bHasUnknown = true;
    indexers.insert(wordId);
    for (int idx = 0; idx < wordDim; idx++) {
      dtype curValue = atof(vecInfo[idx + 1].c_str());
      sum[idx] = curValue;
      wordEmb[wordId][idx] = curValue;
    }

  } else {
    for (int idx = 0; idx < wordDim; idx++) {
      sum[idx] = 0.0;
    }
  }

  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (strLine.empty())
      continue;
    split_bychar(strLine, vecInfo, ' ');
    if (vecInfo.size() != wordDim + 1) {
      std::cout << "error embedding file" << std::endl;
    }
    curWord = normalize_to_lowerwithdigit(vecInfo[0]);
    wordId = m_wordAlphabet.from_string(curWord);
    if (wordId >= 0) {
      count++;
      if (unknownId == wordId)
        bHasUnknown = true;
      indexers.insert(wordId);

      for (int idx = 0; idx < wordDim; idx++) {
        dtype curValue = atof(vecInfo[idx + 1].c_str());
        sum[idx] = curValue;
        wordEmb[wordId][idx] += curValue;
      }
    }

  }

  if (!bHasUnknown) {
    for (int idx = 0; idx < wordDim; idx++) {
      wordEmb[unknownId][idx] = sum[idx] / count;
    }
    count++;
    std::cout << unknownkey << " not found, using averaged value to initialize." << std::endl;
  }

  int oovWords = 0;
  int totalWords = 0;
  for (int id = 0; id < m_wordAlphabet.size(); id++) {
    if (indexers.find(id) == indexers.end()) {
      oovWords++;
      for (int idx = 0; idx < wordDim; idx++) {
        wordEmb[id][idx] = wordEmb[unknownId][idx];
      }
    }
    totalWords++;
  }

  std::cout << "OOV num is " << oovWords << ", total num is " << m_wordAlphabet.size() << ", embedding oov ratio is " << oovWords * 1.0 / m_wordAlphabet.size()
      << std::endl;

}

void Labeler::loadModelFile(const string& inputModelFile) {

}

void Labeler::writeModelFile(const string& outputModelFile) {

}

int main(int argc, char* argv[]) {
#if USE_CUDA==1
  InitTensorEngine();
#else
  InitTensorEngine<cpu>();
#endif


  std::string trainFile = "", devFile = "", testFile = "", modelFile="";
  std::string wordEmbFile = "", charEmbFile = "", optionFile = "";
  std::string outputFile = "";
  bool bTrain = false;
  dsr::Argument_helper ah;

  ah.new_flag("l", "learn", "train or test", bTrain);
  ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
  ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
  ah.new_named_string("test", "testCorpus", "named_string", "testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
  ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
  ah.new_named_string("word", "wordEmbFile", "named_string", "pretrained word embedding file to train a model, optional when training", wordEmbFile);
  ah.new_named_string("char", "charEmbFile", "named_string", "pretrained char embedding file to train a model, optional when training", charEmbFile);
  ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
  ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);

  ah.process(argc, argv);

  Labeler tagger;
  if(bTrain)
  {
      tagger.train(trainFile, devFile, testFile,  modelFile, optionFile, wordEmbFile, charEmbFile);
  }
  else
  {
    tagger.test(testFile, outputFile, modelFile);
  }


#if USE_CUDA==1
  ShutdownTensorEngine();
#else
  ShutdownTensorEngine<cpu>();
#endif
}

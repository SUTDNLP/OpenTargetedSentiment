#ifndef _PARSER_OPTIONS_
#define _PARSER_OPTIONS_

#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include "N3L.h"

using namespace std;

class Options {
public:

  int wordCutOff;
  int featCutOff;
  int charCutOff;
  dtype initRange;
  int maxIter;
  int batchSize;
  dtype adaEps;
  dtype adaAlpha;
  dtype regParameter;
  dtype dropProb;

  int linearHiddenSize;
  int hiddenSize;
  int wordEmbSize;
  int wordcontext;
  bool wordEmbFineTune;
  int charEmbSize;
  int charcontext;
  bool charEmbFineTune;

  int verboseIter;
  bool saveIntermediate;
  bool train;
  int maxInstance;
  vector<string> testFiles;
  string outBest;
  bool seg;
  int relu;
  int atomLayers;

  Options() {
    wordCutOff = 0;
    featCutOff = 0;
    charCutOff = 0;
    initRange = 0.01;
    maxIter = 1000;
    batchSize = 1;
    adaEps = 1e-6;
    adaAlpha = 0.01;
    regParameter = 1e-8;
    dropProb = 0.0;

    linearHiddenSize = 30;
    hiddenSize = 200;
    wordEmbSize = 50;
    wordcontext = 2;
    wordEmbFineTune = true;
    charEmbSize = 50;
    charcontext = 2;
    charEmbFineTune = true;

    verboseIter = 100;
    saveIntermediate = true;
    train = false;
    maxInstance = -1;
    testFiles.clear();
    outBest = "";
    relu = 0;
    seg = false;
    atomLayers = 1;

  }

  virtual ~Options() {

  }

  void setOptions(const vector<string> &vecOption) {
    int i = 0;
    for (; i < vecOption.size(); ++i) {
      pair<string, string> pr;
      string2pair(vecOption[i], pr, '=');
      if (pr.first == "wordCutOff")
        wordCutOff = atoi(pr.second.c_str());
      if (pr.first == "featCutOff")
        featCutOff = atoi(pr.second.c_str());
      if (pr.first == "charCutOff")
        charCutOff = atoi(pr.second.c_str());
      if (pr.first == "initRange")
        initRange = atof(pr.second.c_str());
      if (pr.first == "maxIter")
        maxIter = atoi(pr.second.c_str());
      if (pr.first == "batchSize")
        batchSize = atoi(pr.second.c_str());
      if (pr.first == "adaEps")
        adaEps = atof(pr.second.c_str());
      if (pr.first == "adaAlpha")
        adaAlpha = atof(pr.second.c_str());
      if (pr.first == "regParameter")
        regParameter = atof(pr.second.c_str());
      if (pr.first == "dropProb")
        dropProb = atof(pr.second.c_str());

      if (pr.first == "linearHiddenSize")
        linearHiddenSize = atoi(pr.second.c_str());
      if (pr.first == "hiddenSize")
        hiddenSize = atoi(pr.second.c_str());
      if (pr.first == "wordcontext")
        wordcontext = atoi(pr.second.c_str());
      if (pr.first == "wordEmbSize")
        wordEmbSize = atoi(pr.second.c_str());
      if (pr.first == "wordEmbFineTune")
        wordEmbFineTune = (pr.second == "true") ? true : false;
      if (pr.first == "charcontext")
        charcontext = atoi(pr.second.c_str());
      if (pr.first == "charEmbSize")
        charEmbSize = atoi(pr.second.c_str());
      if (pr.first == "charEmbFineTune")
        charEmbFineTune = (pr.second == "true") ? true : false;

      if (pr.first == "verboseIter")
        verboseIter = atoi(pr.second.c_str());
      if (pr.first == "train")
        train = (pr.second == "true") ? true : false;
      if (pr.first == "saveIntermediate")
        saveIntermediate = (pr.second == "true") ? true : false;
      if (pr.first == "maxInstance")
        maxInstance = atoi(pr.second.c_str());
      if (pr.first == "testFile")
        testFiles.push_back(pr.second);
      if (pr.first == "outBest")
        outBest = pr.second;
      if (pr.first == "relu")
        relu = atoi(pr.second.c_str());
      if (pr.first == "seg")
        seg = (pr.second == "true") ? true : false;
      if (pr.first == "atomLayers")
        atomLayers = atoi(pr.second.c_str());
    }
  }

  void showOptions() {
    std::cout << "wordCutOff = " << wordCutOff << std::endl;
    std::cout << "featCutOff = " << featCutOff << std::endl;
    std::cout << "charCutOff = " << charCutOff << std::endl;
    std::cout << "initRange = " << initRange << std::endl;
    std::cout << "maxIter = " << maxIter << std::endl;
    std::cout << "batchSize = " << batchSize << std::endl;
    std::cout << "adaEps = " << adaEps << std::endl;
    std::cout << "adaAlpha = " << adaAlpha << std::endl;
    std::cout << "regParameter = " << regParameter << std::endl;
    std::cout << "dropProb = " << dropProb << std::endl;

    std::cout << "linearHiddenSize = " << linearHiddenSize << std::endl;
    std::cout << "hiddenSize = " << hiddenSize << std::endl;
    std::cout << "wordEmbSize = " << wordEmbSize << std::endl;
    std::cout << "wordcontext = " << wordcontext << std::endl;
    std::cout << "wordEmbFineTune = " << wordEmbFineTune << std::endl;
    std::cout << "charEmbSize = " << charEmbSize << std::endl;
    std::cout << "charcontext = " << charcontext << std::endl;
    std::cout << "charEmbFineTune = " << charEmbFineTune << std::endl;

    std::cout << "verboseIter = " << verboseIter << std::endl;
    std::cout << "saveItermediate = " << saveIntermediate << std::endl;
    std::cout << "train = " << train << std::endl;
    std::cout << "maxInstance = " << maxInstance << std::endl;
    for (int idx = 0; idx < testFiles.size(); idx++) {
      std::cout << "testFile = " << testFiles[idx] << std::endl;
    }
    std::cout << "outBest = " << outBest << std::endl;
    std::cout << "relu = " << relu << std::endl;
    std::cout << "seg = " << seg << std::endl;
    std::cout << "atomLayers = " << atomLayers << std::endl;
  }

  void load(const std::string& infile) {
    ifstream inf;
    inf.open(infile.c_str());
    vector<string> vecLine;
    while (1) {
      string strLine;
      if (!my_getline(inf, strLine)) {
        break;
      }
      if (strLine.empty())
        continue;
      vecLine.push_back(strLine);
    }
    inf.close();
    setOptions(vecLine);
  }
};

#endif


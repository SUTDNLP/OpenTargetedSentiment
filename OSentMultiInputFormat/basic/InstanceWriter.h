#ifndef _CONLL_WRITER_
#define _CONLL_WRITER_

#include "Writer.h"
#include <sstream>

using namespace std;
/*
 this class writes conll-format result (no srl-info).
 */
class InstanceWriter: public Writer {
public:
  InstanceWriter() {
  }
  ~InstanceWriter() {
  }
  int write(const Instance *pInstance) {
    if (!m_outf.is_open())
      return -1;

    const vector<string> &label1s = pInstance->label1s;
    const vector<string> &label2s = pInstance->label2s;

    for (int i = 0; i < label1s.size(); ++i) {
      m_outf << pInstance->words[i] << " ";
      m_outf << label1s[i] << " " << label2s[i] << endl;
    }
    m_outf << endl;
    return 0;
  }
};

#endif


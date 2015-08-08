#ifndef UTILTENSOR
#define UTILTENSOR

#include "tensor.h"
#include "NRMat.h"
#include <string>
#include <cstdio>
#include <cstdlib>
#include <fstream>

using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;
using namespace nr;

// define tanh operation
struct nl_tanh{
    MSHADOW_XINLINE static double Map(double a) {
//    	return a>0?a:0;
        return  tanh(a);
    }
};
struct nl_dtanh{
    MSHADOW_XINLINE static double Map(double a) {
//    	return a>0?1:0;
        return  (1.0-a)*(1.0+a);
    }
};
struct nl_sigmoid{
    MSHADOW_XINLINE static double Map(double a) {
//    	return a>0?a:0;
        return 1.0/(1.0+exp(-a));
    }
};
struct nl_dsigmoid{
    MSHADOW_XINLINE static double Map(double a) {
//    	return a>0?1:0;
        return  (1.0-a)*a;
    }
};
struct nl_relu{
    MSHADOW_XINLINE static double Map(double a) {
    	return a>0?a:0;
    }
};
struct nl_drelu{
    MSHADOW_XINLINE static double Map(double a) {
    	return a>0?1:0;
    }
};
struct nl_exp{
    MSHADOW_XINLINE static double Map(double a) {
//    	return a>0?a:0;
        return  exp(a);
    }
};
struct xe_dx{
	MSHADOW_XINLINE static double Map(double a, double b){
		return (b-a)/(a*(1.0-a)+1e-6);
	}
};
struct xe_ll{
	MSHADOW_XINLINE static double Map(double a, double b){
		return b>0.5f?log(a+1e-10):log(1.0-a+1e-10);
	}
};
struct square{
    MSHADOW_XINLINE static double Map(double a) {
        return  a*a;

    }
};
struct clip{
    MSHADOW_XINLINE static double Map(double a) {
        return  a>10.0?10.0:(a<-10.0?-10.0:a);

    }
};
struct inv_sqrt{
    MSHADOW_XINLINE static double Map(double a, double b) {
        return a/(sqrt(b)+0.0001);
    }
};

struct nl_sqrt{
    MSHADOW_XINLINE static double Map(double a) {
        return sqrt(a);
    }
};

struct dropout{
	// p: prob to dropout
    MSHADOW_XINLINE static double Map(double p, double r) {
        if(p>r) return 0.0;
        else return 1.0/(1.0-p);
    }
};

template<typename xpu, typename DType>
inline DType squarenorm(const Tensor<xpu, 1, DType>& w)
{
	DType result = 0;
	for(int idx = 0; idx < w.size(0); idx++)
	{
		result += w[idx] * w[idx];
	}
	return result;
}

template<typename xpu, typename DType>
inline DType squarenorm(const Tensor<xpu, 2, DType>& w)
{
	DType result = 0;
	for(int idx = 0; idx < w.size(0); idx++)
	{
		for(int idy = 0; idy < w.size(1); idy++)
		{
			result += w[idx][idy] * w[idx][idy];
		}
	}
	return result;
}

template<typename xpu, typename DType>
inline DType squarenorm(const Tensor<xpu, 3, DType>& w)
{
	DType result = 0;
	for(int idx = 0; idx < w.size(0); idx++)
	{
		for(int idy = 0; idy < w.size(1); idy++)
		{
			for(int idz = 0; idz < w.size(2); idz++)
			{
				result += w[idx][idy][idz] * w[idx][idy][idz];
			}
		}
	}
	return result;
}

template<typename xpu, typename DType>
inline void assign(Tensor<xpu, 1, DType>& w, const NRVec<DType>& wnr)
{
	int dim = wnr.size();
	for(int idx = 0; idx < dim; idx++)
	{
		w[idx]  =  wnr[idx];
	}
}

template<typename xpu, typename DType>
inline void assign(Tensor<xpu, 2, DType> w, const NRMat<DType>& wnr)
{
	int dim1 = wnr.nrows();
	int dim2 = wnr.ncols();
	for(int idx = 0; idx < dim1; idx++)
	{
		for(int idy = 0; idy < dim2; idy++)
		{
			w[idx][idy] = wnr[idx][idy];
		}
	}
}

template<typename xpu, typename DType>
inline void assign(Tensor<xpu, 3, DType> w, const NRMat3d<DType>& wnr)
{
	int dim1 = wnr.dim1();
	int dim2 = wnr.dim2();
	int dim3 = wnr.dim3();
	for(int idx = 0; idx < dim1; idx++)
	{
		for(int idy = 0; idy < dim2; idy++)
		{
			for(int idz = 0; idz < dim3; idz++)
			{
				w[idx][idy][idz] = wnr[idx][idy][idz];
			}
		}
	}
}

template<typename xpu, typename DType>
inline void norm2one(Tensor<xpu, 2, DType>& w, int idx) {
  DType sum = 0.000001;
  for(int idy = 0; idy < w.size(1); idy++)
  {
    sum += w[idx][idy] * w[idx][idy];
  }
  DType scale = sqrt(sum);
  for(int idy = 0; idy < w.size(1); idy++)
    w[idx][idy] = w[idx][idy] / scale;
}

//only applicable on Shape2(1,x), notice that we add the value to the target
template<typename xpu, typename DType>
inline void concat(const Tensor<xpu, 2, DType>& w1, const Tensor<xpu, 2, DType>& w2, Tensor<xpu, 2, DType>& w)
{
  if(w1.size(0) != 1 || w2.size(0) != 1 || w.size(0) != 1)
  {
    std::cerr << "concat error, only support Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  int col2 = w2.size(1);
  if(col1 + col2 != col)
  {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  for(int idx = 0; idx < row; idx++)
  {
    offset = 0;
    for(int idy = 0; idy < col1; idy++)
    {
      w[idx][offset] += w1[idx][idy];
      offset++;
    }
    for(int idy = 0; idy < col2; idy++)
    {
      w[idx][offset] += w2[idx][idy];
      offset++;
    }
  }
  return;
}

template<typename xpu, typename DType>
inline void concat(const Tensor<xpu, 2, DType>& w1, const Tensor<xpu, 2, DType>& w2, const Tensor<xpu, 2, DType>& w3, Tensor<xpu, 2, DType>& w)
{
  if(w1.size(0) != 1 || w2.size(0) != 1 || w3.size(0) != 1 || w.size(0) != 1)
  {
    std::cerr << "concat error, only support Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  int col2 = w2.size(1);
  int col3 = w3.size(1);
  if(col1 + col2 + col3 != col)
  {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  for(int idx = 0; idx < row; idx++)
  {
    offset = 0;
    for(int idy = 0; idy < col1; idy++)
    {
      w[idx][offset] += w1[idx][idy];
      offset++;
    }
    for(int idy = 0; idy < col2; idy++)
    {
      w[idx][offset] += w2[idx][idy];
      offset++;
    }
    for(int idy = 0; idy < col3; idy++)
    {
      w[idx][offset] += w3[idx][idy];
      offset++;
    }
  }
  return;
}

template<typename xpu, typename DType>
inline void concat(const Tensor<xpu, 2, DType>& w1, const Tensor<xpu, 2, DType>& w2, const Tensor<xpu, 2, DType>& w3,
    const Tensor<xpu, 2, DType>& w4, Tensor<xpu, 2, DType>& w)
{
  if(w1.size(0) != 1 || w2.size(0) != 1 || w3.size(0) != 1 || w4.size(0) != 1 || w.size(0) != 1)
  {
    std::cerr << "concat error, only support Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  int col2 = w2.size(1);
  int col3 = w3.size(1);
  int col4 = w4.size(1);
  if(col1 + col2 + col3 + col4  != col)
  {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  for(int idx = 0; idx < row; idx++)
  {
    offset = 0;
    for(int idy = 0; idy < col1; idy++)
    {
      w[idx][offset] += w1[idx][idy];
      offset++;
    }
    for(int idy = 0; idy < col2; idy++)
    {
      w[idx][offset] += w2[idx][idy];
      offset++;
    }
    for(int idy = 0; idy < col3; idy++)
    {
      w[idx][offset] += w3[idx][idy];
      offset++;
    }
    for(int idy = 0; idy < col4; idy++)
    {
      w[idx][offset] += w4[idx][idy];
      offset++;
    }
  }
  return;
}

template<typename xpu, typename DType>
inline void concat(const Tensor<xpu, 2, DType>& w1, const Tensor<xpu, 2, DType>& w2, const Tensor<xpu, 2, DType>& w3,
    const Tensor<xpu, 2, DType>& w4, const Tensor<xpu, 2, DType>& w5, Tensor<xpu, 2, DType>& w)
{
  if(w1.size(0) != 1 || w2.size(0) != 1 || w3.size(0) != 1 || w4.size(0) != 1 || w5.size(0) != 1 || w.size(0) != 1)
  {
    std::cerr << "concat error, only support Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  int col2 = w2.size(1);
  int col3 = w3.size(1);
  int col4 = w4.size(1);
  int col5 = w5.size(1);
  if(col1 + col2 + col3 + col4 + col5 != col)
  {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  for(int idx = 0; idx < row; idx++)
  {
    offset = 0;
    for(int idy = 0; idy < col1; idy++)
    {
      w[idx][offset] += w1[idx][idy];
      offset++;
    }
    for(int idy = 0; idy < col2; idy++)
    {
      w[idx][offset] += w2[idx][idy];
      offset++;
    }
    for(int idy = 0; idy < col3; idy++)
    {
      w[idx][offset] += w3[idx][idy];
      offset++;
    }
    for(int idy = 0; idy < col4; idy++)
    {
      w[idx][offset] += w4[idx][idy];
      offset++;
    }
    for(int idy = 0; idy < col5; idy++)
    {
      w[idx][offset] += w5[idx][idy];
      offset++;
    }
  }
  return;
}

//only applicable on Shape2(1,x), notice that we add the value to the target
template<typename xpu, typename DType>
inline void unconcat(Tensor<xpu, 2, DType>& w1, Tensor<xpu, 2, DType>& w2, const Tensor<xpu, 2, DType>& w)
{
  if(w1.size(0) != 1 || w2.size(0) != 1 || w.size(0) != 1)
  {
    std::cerr << "unconcat error, only spport Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  int col2 = w2.size(1);
  if(col1 + col2 != col)
  {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  for(int idx = 0; idx < row; idx++)
  {
    offset = 0;
    for(int idy = 0; idy < col1; idy++)
    {
      w1[idx][idy] += w[idx][offset];
      offset++;
    }
    for(int idy = 0; idy < col2; idy++)
    {
      w2[idx][idy] += w[idx][offset];
      offset++;
    }
  }
  return;
}

template<typename xpu, typename DType>
inline void unconcat(Tensor<xpu, 2, DType>& w1, Tensor<xpu, 2, DType>& w2, Tensor<xpu, 2, DType>& w3, const Tensor<xpu, 2, DType>& w)
{
  if(w1.size(0) != 1 || w2.size(0) != 1 || w3.size(0) != 1 || w.size(0) != 1)
  {
    std::cerr << "unconcat error, only spport Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  int col2 = w2.size(1);
  int col3 = w3.size(1);
  if(col1 + col2 + col3 != col)
  {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  for(int idx = 0; idx < row; idx++)
  {
    offset = 0;
    for(int idy = 0; idy < col1; idy++)
    {
      w1[idx][idy] += w[idx][offset];
      offset++;
    }
    for(int idy = 0; idy < col2; idy++)
    {
      w2[idx][idy] += w[idx][offset];
      offset++;
    }
    for(int idy = 0; idy < col3; idy++)
    {
      w3[idx][idy] += w[idx][offset];
      offset++;
    }
  }
  return;
}

template<typename xpu, typename DType>
inline void unconcat(Tensor<xpu, 2, DType>& w1, Tensor<xpu, 2, DType>& w2, Tensor<xpu, 2, DType>& w3,
    Tensor<xpu, 2, DType>& w4, const Tensor<xpu, 2, DType>& w)
{
  if(w1.size(0) != 1 || w2.size(0) != 1 || w3.size(0) != 1 || w4.size(0) != 1 || w.size(0) != 1)
  {
    std::cerr << "unconcat error, only spport Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  int col2 = w2.size(1);
  int col3 = w3.size(1);
  int col4 = w4.size(1);
  if(col1 + col2 + col3 + col4  != col)
  {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  for(int idx = 0; idx < row; idx++)
  {
    offset = 0;
    for(int idy = 0; idy < col1; idy++)
    {
      w1[idx][idy] += w[idx][offset];
      offset++;
    }
    for(int idy = 0; idy < col2; idy++)
    {
      w2[idx][idy] += w[idx][offset];
      offset++;
    }
    for(int idy = 0; idy < col3; idy++)
    {
      w3[idx][idy] += w[idx][offset];
      offset++;
    }
    for(int idy = 0; idy < col4; idy++)
    {
      w4[idx][idy] += w[idx][offset];
      offset++;
    }
  }
  return;
}

template<typename xpu, typename DType>
inline void unconcat(Tensor<xpu, 2, DType>& w1, Tensor<xpu, 2, DType>& w2, Tensor<xpu, 2, DType>& w3,
    Tensor<xpu, 2, DType>& w4, Tensor<xpu, 2, DType>& w5, const Tensor<xpu, 2, DType>& w)
{
  if(w1.size(0) != 1 || w2.size(0) != 1 || w3.size(0) != 1 || w4.size(0) != 1 || w5.size(0) != 1 || w.size(0) != 1)
  {
    std::cerr << "unconcat error, only spport Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  int col2 = w2.size(1);
  int col3 = w3.size(1);
  int col4 = w4.size(1);
  int col5 = w5.size(1);
  if(col1 + col2 + col3 + col4 + col5 != col)
  {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  for(int idx = 0; idx < row; idx++)
  {
    offset = 0;
    for(int idy = 0; idy < col1; idy++)
    {
      w1[idx][idy] += w[idx][offset];
      offset++;
    }
    for(int idy = 0; idy < col2; idy++)
    {
      w2[idx][idy] += w[idx][offset];
      offset++;
    }
    for(int idy = 0; idy < col3; idy++)
    {
      w3[idx][idy] += w[idx][offset];
      offset++;
    }
    for(int idy = 0; idy < col4; idy++)
    {
      w4[idx][idy] += w[idx][offset];
      offset++;
    }
    for(int idy = 0; idy < col5; idy++)
    {
      w5[idx][idy] += w[idx][offset];
      offset++;
    }
  }
  return;
}



#endif

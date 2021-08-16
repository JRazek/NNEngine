//
// Created by user on 15.08.2021.
//

#include "TMatrix.h"

TMatrix::TMatrix(const TMatrix::func &_a, const TMatrix::func &_b, const TMatrix::func &_c, const TMatrix::func &_d)
    :
    a(_a),
    b(_b),
    c(_c),
    d(_d){}

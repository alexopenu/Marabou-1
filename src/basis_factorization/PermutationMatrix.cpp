/********************                                                        */
/*! \file PermutationMatrix.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz
 ** This file is part of the Marabou project.
 ** Copyright (c) 2016-2017 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **/

#include "BasisFactorizationError.h"
#include "PermutationMatrix.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>

PermutationMatrix::PermutationMatrix( unsigned m )
    : _ordering( NULL )
    , _m( m )
{
    _ordering = new unsigned[m];
    if ( !_ordering )
        throw BasisFactorizationError( BasisFactorizationError::ALLOCATION_FAILED,
                                       "PermutationMatrix::ordering" );

    resetToIdentity();
}

PermutationMatrix::~PermutationMatrix()
{
    if ( _ordering )
    {
        delete[] _ordering;
        _ordering = NULL;
    }
}

void PermutationMatrix::resetToIdentity()
{
    for ( unsigned i = 0; i < _m; ++i )
        _ordering[i] = i;
}

PermutationMatrix &PermutationMatrix::operator=( const PermutationMatrix &other )
{
    if ( _m != other._m )
    {
        _m = other._m;

        if ( _ordering )
            delete[] _ordering;

        _ordering = new unsigned[_m];
    }

    memcpy( _ordering, other._ordering, sizeof(unsigned) * _m );

    return *this;
}

PermutationMatrix *PermutationMatrix::invert() const
{
    PermutationMatrix *inverse = new PermutationMatrix( _m );

    for ( unsigned i = 0; i < _m; ++i )
        inverse->_ordering[_ordering[i]] = i;

    return inverse;
}

unsigned PermutationMatrix::getM() const
{
    return _m;
}

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//

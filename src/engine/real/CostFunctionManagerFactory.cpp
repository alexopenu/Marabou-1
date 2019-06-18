/*********************                                                        */
/*! \file CostFunctionManagerFactory.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

**/

#include <T/CostFunctionManagerFactory.h>

#include "CostFunctionManager.h"

class ITableau;

namespace T
{
	ICostFunctionManager *createCostFunctionManager( ITableau *tableau )
	{
		return new CostFunctionManager( tableau );
	}

    ICostFunctionManager *createCostFunctionManager(ITableau *tableau, double *costFunction) {
        return new CostFunctionManager( tableau , costFunction);
    }

	void discardCostFunctionManager( ICostFunctionManager *costFunctionManager )
	{
		delete costFunctionManager;
	}


}

//
// Local Variables:
// compile-command: "make -C ../../.. "
// tags-file-name: "../../../TAGS"
// c-basic-offset: 4
// End:
//

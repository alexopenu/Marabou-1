#!/usr/bin/csh

set GUROBI_HOME "/cs/labs/guykatz/alexus/gurobi903/linux64/"
set PATH "${PATH}:${GUROBI_HOME}/bin"
set LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
set GRB_LICENSE_FILE "/cs/share/etc/license/gurobi/gurobi.lic"


#!/usr/bin/csh

setenv GUROBI_HOME "/cs/labs/guykatz/alexus/gurobi903/linux64/"
setenv PATH "${PATH}:${GUROBI_HOME}/bin"
setenv LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
setenv GRB_LICENSE_FILE "/cs/share/etc/license/gurobi/gurobi.lic"


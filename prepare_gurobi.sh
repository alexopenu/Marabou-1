#!/usr/bin/bash

export GUROBI_HOME="PATH_TO_YOUR_GUROBI"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
export GRB_LICENSE_FILE="/cs/share/etc/license/gurobi/gurobi.lic"

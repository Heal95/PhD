#!/bin/csh

set VELMODEL = DF_1D.fk1d
set OUTFILE = stimes.dat
set SHALL_DEP = 0.0001
set TMPVELFILE = temp_vel-stimes.mod

gawk 'BEGIN {
        i = 0;
    } 
    {
        if (substr($0, 1, 1) != "#" && NF == 6) { 
            i++;
            thk[i] = $1; vp[i] = $2; vs[i] = $3; rho[i] = $4; qp[i] = $5; qs[i] = $6;
        }
    }
    END {
        nl = i;
        printf "%d\n", nl;
        for (i = 1; i <= nl; i++) {
            printf "%11.4f%11.4f%11.4f%11.4f%11.2f%11.2f\n", thk[i], vp[i], vs[i], rho[i], qp[i], qs[i];
        }
    }' $VELMODEL > $TMPVELFILE

./ray_stimes shallowest_depth=$SHALL_DEP velfile=$TMPVELFILE locfile=locations.dat > $OUTFILE

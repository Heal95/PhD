#!/bin/csh

set OUTFILE = locations.dat

set DSTR = ( 0.1   16.0   36.0 )
set DEND = ( 15.81 35.10  100.0 )
set DDEL = ( 0.1   0.2    1.0 )

set RSTR = ( 0.1   21.0   46.0  100.0 )
set REND = ( 20.91 45.81  99.1  160.1 )
set RDEL = ( 0.1   0.2    1.0    2.0 )

# Process RSTR, REND, RDEL
echo $RSTR $REND $RDEL | gawk -v nr=$#RSTR '
{
    for (i = 1; i <= nr; i++) {
        rs[i] = $i;
        re[i] = $(i + nr);
        dr[i] = $(i + 2 * nr);
    }
}
END {
    nn = 0;
    for (i = 1; i <= nr; i++) {
        j = 0;
        rr = rs[i];
        while (rr <= re[i]) {
            nn++;
            xr[nn] = rr;
            j++;
            rr = rs[i] + j * dr[i];
        }
    }
    printf "%d\n", nn;
    for (i = 1; i <= nn; i++) {
        printf "%10.4f", xr[i];
        if ((i % 6) == 0) printf "\n";
    }
    if ((nn % 6) != 0) printf "\n";
}' > $OUTFILE

# Process DSTR, DEND, DDEL
echo $DSTR $DEND $DDEL | gawk -v nr=$#DSTR '
{
    for (i = 1; i <= nr; i++) {
        rs[i] = $i;
        re[i] = $(i + nr);
        dr[i] = $(i + 2 * nr);
    }
}
END {
    nn = 0;
    for (i = 1; i <= nr; i++) {
        j = 0;
        rr = rs[i];
        while (rr <= re[i]) {
            nn++;
            xr[nn] = rr;
            j++;
            rr = rs[i] + j * dr[i];
        }
    }
    printf "%d\n", nn;
    for (i = 1; i <= nn; i++) {
        printf "%10.4f", xr[i];
        if ((i % 6) == 0) printf "\n";
    }
    if ((nn % 6) != 0) printf "\n";
}' >> $OUTFILE

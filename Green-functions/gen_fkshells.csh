#!/bin/csh

# Set variables
set VELMODEL      = DF_1D.fk1d
set RUN_NAME      = df1d
set FKDIR         = FkRuns
set RANGE_BLOCKS  = ( 1 2 3 4 )
set NT            = (  256  512 1024 2048 )
set RANGE_MAXS    = ( 20.1 40.1 100.1 140.1 )
set DSTR          = ( 0.1   16.0   36.0 )
set DEND          = ( 15.81 35.10  100.0 )
set DDEL          = ( 0.1   0.2    1.0 )
set RSTR          = ( 0.1   21.0   46.0  100.0 )
set REND          = ( 20.91 45.81  99.1  160.1 )
set RDEL          = ( 0.1   0.2    1.0    2.0 )
set BGFOUT        = 1

set HERE          = `pwd`
set GFDIR         = ${HERE}/BaileyGF

mkdir -p $GFDIR
mkdir -p $FKDIR

# Set constants
set SRC_TYPE      = 2   # Double-couple for 3 fundamental faults
set REC_LAYER     = 1   # Receivers on top of layer 1 (ground surface)
set UP_DOWN       = 0   # Both up and down going waves

set MAXP          = 2.0
set DT            = 0.1
set NBEFORE       = 50
set RESAMP        = 1
set WC1           = 1
set WC2           = 1
set REDV          = 9.0

set ORDER         = 4
set FLO           = 8.0
set FHI           = 0.0

set MIN_SLOW      = 0.0
set SIGMA         = 2.0
set TAPLEN        = 0.0
set DK            = 0.1
set MAXK          = 20

set BINDIR        = /home/hlena/Fk3.2
set FKPROG        = fk2bgf
set VFILE         = ${VELMODEL}

set TMPDEPFILE    = temp_dep.txt
set TMPRNGFILE1   = temp_range1.txt
set TMPRNGFILE2   = temp_range2.txt
set TMPVELFILE    = temp_vel.txt

# Process depth data
echo $DSTR $DEND $DDEL | gawk -v nr=$#DSTR '{
  for(i=1;i<=nr;i++){
    rs[i]=$i;
    re[i]=$(i+nr);
    dr[i]=$(i+2*nr);
  }
} END {
  for(i=1;i<=nr;i++){
    j=0; rr=rs[i];
    while(rr <= re[i]){
      printf "%8.3f\n",rr;
      j++;
      rr=rs[i]+j*dr[i];
    }
  }
}' > $TMPDEPFILE

set DEPS = `\cat $TMPDEPFILE`

# Process range data
echo $RSTR $REND $RDEL | gawk -v nr=$#RSTR '{
  for(i=1;i<=nr;i++){
    rs[i]=$i;
    re[i]=$(i+nr);
    dr[i]=$(i+2*nr);
  }
} END {
  for(i=1;i<=nr;i++){
    j=0; rr=rs[i];
    while(rr <= re[i]){
      printf "%8.3f\n",rr;
      j++;
      rr=rs[i]+j*dr[i];
    }
  }
}' > $TMPRNGFILE1

set dtot = 0
foreach dep ( $DEPS )
  @ dtot++
  echo -n "dep=$dep "

# Set up velocity model for this source depth
gawk -v ds=$dep -v src_type=$SRC_TYPE -v rec_layer=$REC_LAYER -v updwn=$UP_DOWN '
BEGIN { i = 0; }
{
    if (substr($0, 1, 1) != "#" && NF == 6) {
        i++;
        thk[i] = $1; vp[i] = $2; vs[i] = $3; rho[i] = $4; qp[i] = $5; qs[i] = $6;
    }
}
END {
    nl = i; dep1 = 0.0; j = 0;
    for (i = 1; i <= nl; i++) {
        j++; dep = dep1 + thk[i];
        if (dep > ds && dep1 < ds) {
            t[j] = ds - dep1; p[j] = vp[i]; s[j] = vs[i]; r[j] = rho[i]; ap[j] = qp[i]; as[j] = qs[i];
            j++; ns = j;
            t[j] = dep - ds; p[j] = vp[i]; s[j] = vs[i]; r[j] = rho[i]; ap[j] = qp[i]; as[j] = qs[i];
        } else if (dep1 == ds) {
            dsx = ds + 0.005;
            t[j] = dsx - dep1; p[j] = vp[i]; s[j] = vs[i]; r[j] = rho[i]; ap[j] = qp[i]; as[j] = qs[i];
            j++; ns = j;
            t[j] = dep - dsx; p[j] = vp[i]; s[j] = vs[i]; r[j] = rho[i]; ap[j] = qp[i]; as[j] = qs[i];
        } else {
            t[j] = thk[i]; p[j] = vp[i]; s[j] = vs[i]; r[j] = rho[i]; ap[j] = qp[i]; as[j] = qs[i];
        }
        dep1 = dep;
    }
    nl++;
    printf "%d %d %s %s %s\n", nl, ns, src_type, rec_layer, updwn;
    for (i = 1; i <= nl; i++) {
        printf "%11.4f%11.4f%11.4f%11.4f%11.2f%11.2f\n", t[i], p[i], s[i], r[i], ap[i], as[i];
    }
}' $VFILE > $TMPVELFILE

# Set up distance range file for this source depth
set rmin = 0.0
set rtot = 0
set a = 0

foreach rb ($RANGE_BLOCKS)
    @ a++
    set RNGS = `gawk -v rmin=$rmin -v rmax=$RANGE_MAXS[$a] '{if($1>rmin && $1<=rmax)print $1;}' $TMPRNGFILE1`

    echo $#RNGS $BGFOUT $ORDER $FHI $FLO | \
    gawk '{printf "%6d %d %d %13.5e %13.5e\n", $1, $2, $3, $4, $5;}' > $TMPRNGFILE2

    foreach rng ($RNGS)
        @ rtot++
        echo $rng $REDV $GFDIR $RUN_NAME $dtot $rtot | \
        gawk '{printf "%8.2f%8.2f \"%s/%s%.4d%.4d\"\n", $1, $1/$2, $3, $4, $5, $6;}' >> $TMPRNGFILE2
    end

    echo -n "nr=$#RNGS "

    set FKCSH = $FKDIR/fk_d${dep}-r${rb}.csh

    echo "#!/bin/csh"                  > $FKCSH
    echo ""                            >> $FKCSH
    echo "mkdir -p $GFDIR"             >> $FKCSH
    echo ""                            >> $FKCSH
    echo "$BINDIR/${FKPROG} << END >& d${dep}-r${rb}.out" >> $FKCSH
    \cat $TMPVELFILE                   >> $FKCSH
    echo "$SIGMA ${NT[$a]} $DT $TAPLEN $NBEFORE $RESAMP $WC1 $WC2" >> $FKCSH
    echo "$MIN_SLOW $MAXP $DK $MAXK"   >> $FKCSH
    \cat $TMPRNGFILE2                  >> $FKCSH
    echo "END"                         >> $FKCSH

    \chmod 0744 $FKCSH

    set rmin = $RNGS[$#RNGS]
end

echo DONE
end

\rm $TMPDEPFILE $TMPVELFILE $TMPRNGFILE1 $TMPRNGFILE2

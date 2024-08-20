Scripts for DF_1D model Green functions - https://urn.nsk.hr/urn:nbn:hr:217:065258
----------------------------------------------------------------------------------
To calculate Green's functions, the Fk3.2 code was used (https://www.eas.slu.edu/People/LZhu/home.html; accessed in June 2021). Running the code requires three input files:
● DF_1D.fk1d: 1D seismic model.
● locations.dat: Distance matrix.
● stimes.dat: Arrival time matrix for P- and S-waves for the locations of interest.

The distance matrix was defined to cover the domain of interest using gen_locs.csh.
The arrival time matrix for P- and S-waves was obtained using gen_stimes.csh and the ray_stimes.c routine, which is implemented on the SCEC platform (https://doi.org/10.1785/0220140125; accessed in June 2021).
Script gen_fkshells.csh is used to define individual scripts to run Fk3.2 code for each matrix component.
Script run.sh is used to run all fkshells.

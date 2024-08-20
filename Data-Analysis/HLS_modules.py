import csv
import pyproj
import numpy as np

def transform_attenuation(qp, qs, vp, vs):
    
    """
    Transforms qp to qkappa and qs to qmu.
    """
    n = 1 / qp - (4 / 3) * vs ** 2 / vp ** 2 * 1 / qs
    d = 1 - (4 / 3) * vs ** 2 / vp ** 2
    qkappa = d/n
    qmu = qs

    return (qkappa, qmu)


def Olsen_attenuation(vs):
    
    """
    Calculates qs and qp from vs (m/s) using Olsen's ratio
    """
    qs = 0.02*vs
    qp = 1.5*qs
    
    return (qs, qp)


def make_netcdf_from_tomo(infile, outfile, attenuation):

    """
    Create netcdf model from tomography file.

    :param infile: tomography model file
    :param outfile: name of the output netcdf file
    :param attenuation: True/False 

    """

    pom = False
    with open(infile) as ifile:
        for k in range(0,13):
            if ifile.readline().startswith('#nx '):
                pom = True
            if pom:
                nx, ny, nz = ifile.readline().split(' ')
                pom = False

    data = np.loadtxt(infile, skiprows=12, dtype=np.float32)
    x, y, z, vp, vs, rho = data.T

    shape = (int(nz), int(ny), int(nx))

    x = x.reshape(shape)
    y = y.reshape(shape)
    z = z.reshape(shape)
    vp = vp.reshape(shape)
    vs = vs.reshape(shape)
    rho = rho.reshape(shape)
    if attenuation:
        qs, qp = Olsen_attenuation(vs)
        qkappa, qmu = transform_attenuation(qp, qs, vp, vs)
        qkappa = np.moveaxis(np.moveaxis(qkappa, 0, -1), 0, 1)
        qmu = np.moveaxis(np.moveaxis(qmu, 0, -1), 0, 1)

    # Reorder a bit because x, y, z is a bit easier to reason about.
    x = np.moveaxis(np.moveaxis(x, 0, -1), 0, 1)
    y = np.moveaxis(np.moveaxis(y, 0, -1), 0, 1)
    z = np.moveaxis(np.moveaxis(z, 0, -1), 0, 1)
    vp = np.moveaxis(np.moveaxis(vp, 0, -1), 0, 1)
    vs = np.moveaxis(np.moveaxis(vs, 0, -1), 0, 1)
    rho = np.moveaxis(np.moveaxis(rho, 0, -1), 0, 1)

    unique_x = np.unique(x)
    unique_y = np.unique(y)
    unique_z = np.unique(z)

    if attenuation:
        ds = xr.Dataset(
            data_vars={
                "VP": (["x", "y", "z"], np.require(vp[:, :, ::-1], requirements="C")),
                "VS": (["x", "y", "z"], np.require(vs[:, :, ::-1], requirements="C")),
                "RHO": (["x", "y", "z"], np.require(rho[:, :, ::-1], requirements="C")),
                "QMU": (["x", "y", "z"], np.require(qmu[:, :, ::-1], requirements="C")),
                "QKAPPA": (["x", "y", "z"], np.require(qkappa[:, :, ::-1], requirements="C")),
            },
            coords={
                "x": np.require(unique_x, requirements="C"),
                "y": np.require(unique_y, requirements="C"),
                "z": np.require(unique_z, requirements="C")
            },
        )

    else:
        ds = xr.Dataset(
            data_vars={
                "VP": (["x", "y", "z"], np.require(vp[:, :, ::-1], requirements="C")),
                "VS": (["x", "y", "z"], np.require(vs[:, :, ::-1], requirements="C")),
                "RHO": (["x", "y", "z"], np.require(rho[:, :, ::-1], requirements="C")),
            },
            coords={
                "x": np.require(unique_x, requirements="C"),
                "y": np.require(unique_y, requirements="C"),
                "z": np.require(unique_z, requirements="C")
            },
        )

    # Save file to netcdf format for salvus
    ds.to_netcdf(outfile)

def get_domain_from_tomo(infile):

    """
    Read domain of the model from tomography file.

    :param infile: tomography model file
    :return ds*: domain of the model 

    """

    filename = infile
    pom = False
    
    with open(filename) as infile:
        for k in range(0,13):
            if infile.readline().startswith('#orig_x '):
                pom = True
            if pom:
                dsxmin, dsymin, _, dsxmax, dsymax, dszmax = infile.readline().split(' ')
                pom = False

    return (float(dsxmin), float(dsxmax), float(dsymin), float(dsymax), -float(dszmax))

def get_mvalues_from_tomo(infile):

    """
    Read min and max values of the model from tomography file.

    :param infile: tomography model file
    :return vp*, vs*, ro*: min and max values of the seismic model 

    """

    filename = infile
    pom = False
    
    with open(filename) as infile:
        for k in range(0,13):
            if infile.readline().startswith('#vpmin '):
                pom = True
            if pom:
                vpmin, vpmax, vsmin, vsmax, romin, romax = infile.readline().split(' ')
                pom = False

    return (float(vpmin), float(vpmax), float(vsmin), float(vsmax), float(romin), float(romax))

def read_events(infile):

    """
    Read event info from file named infile.

    :param infile: text file 9 columns in CSV format
                 1. event name
                 2. hypocentral time
                 3. latitude
                 4  longitude
                 5. depth (km) 
                 6. moment magnitude Mw
                 7. strike
                 8. dip
                 9. rake

    :return ev_name: event name 
    :return hyp_time: hypocentral time
    :return lat, lon: latitude, longitude 
    :return dep: depth (m) 
    :return Mw: moment magnitude 
    :return S, D, R: strike, dip, rake

    """

    csvData = []
    with open(infile, 'r') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        for csvRow in csvReader:
            csvData.append(csvRow)

    csvData = np.array(csvData)
    x1, x2, x3, x4, x5 = csvData[:,0], csvData[:,1], csvData[:,2], csvData[:,3], csvData[:,4]
    x6, x7, x8, x9 = csvData[:,5], csvData[:,6], csvData[:,7], csvData[:,8]
    
    ev_name = []; hyp_time = []; lat = []; lon = []; dep = []; 
    Mw = []; S = []; D = []; R = []
    
    for k in range(1,len(x1)):
        ev_name.append(str(x1[k]))
        hyp_time.append(str(x2[k]))
        lat.append(float(x3[k]))
        lon.append(float(x4[k]))
        dep.append(-float(x5[k])*1000)
        Mw.append(float(x6[k]))
        S.append(float(x7[k]))
        D.append(float(x8[k]))
        R.append(float(x9[k]))
    
    return (ev_name, hyp_time, lat, lon, dep, Mw, S, D, R)

def read_stations(infile):

    """
    Read stations info from file named infile.

    :param infile: text file 6 columns in CSV format
                 1. station name
                 2. network
                 3. latitude
                 4  longitude
                 5. elevation (m)
                 6. error

    :return sta: station code 
    :return net: network code
    :return lat, lon: latitude, longitude 
    :return elv: elevation (m) 

    """

    csvData = []
    with open(infile, 'r') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        for csvRow in csvReader:
            csvData.append(csvRow)

    csvData = np.array(csvData)
    x1, x2, x3, x4, x5 = csvData[:,0], csvData[:,1], csvData[:,2], csvData[:,3], csvData[:,4]
    
    sta = []; net = []; lat = []; lon = []; elv = []; 
    
    for k in range(0,len(x1)):
        sta.append(str(x1[k]))
        net.append(str(x2[k]))
        lat.append(float(x3[k]))
        lon.append(float(x4[k]))
        elv.append(float(x5[k]))
    
    return (sta, net, lat, lon, elv)

def sdr2cmt(Mw, S, D, R):

    """
    Convert FPS data to 3D moment tensor (Nm) in Cartesian coordinate system.

    :param Mw: moment magnitude of the event
    :param S: strike
    :param D: dip
    :param R: rake

    :return Mxx, Myy, Mzz, Mxy, Mxz, Myz: 3D moment tensor components

    """

    # pi / 180 to convert degrees to radians
    d2r =  np.pi/180

    # convert to radians
    S *= d2r
    D *= d2r
    R *= d2r

    # earthquake magnitude
    M0 = pow(10,1.5*(Mw+10.7))*pow(10,-7); # seismic moment in Nm

    Mxx = -M0 * (np.sin(D) * np.cos(R) * np.sin(2*S) + np.sin(2*D) * np.sin(R) * np.sin(S)*np.sin(S))
    Myy =  M0 * (np.sin(D) * np.cos(R) * np.sin(2*S) - np.sin(2*D) * np.sin(R) * np.cos(S)*np.cos(S))
    Mzz = -1.0 * (Mxx + Myy)
    Mxy =  M0 * (np.sin(D) * np.cos(R) * np.cos(2*S) + 0.5 * np.sin(2*D) * np.sin(R) * np.sin(2*S))
    Mxz = -M0 * (np.cos(D) * np.cos(R) * np.cos(S) + np.cos(2*D) * np.sin(R) * np.sin(S))
    Myz = -M0 * (np.cos(D) * np.cos(R) * np.sin(S) - np.cos(2*D) * np.sin(R) * np.cos(S))
    
    return (Mxx, Myy, Mzz, Mxy, Mxz, Myz)

def convert_to_UTM(lon, lat, zone):

    """
    Convert data to UTM system.

    :param lon: longitude
    :param lat: latitude
    :param zone: UTM

    :return x, y: coordinates in UTM system

    """
    
    proj = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84', preserve_units=False)          
    x, y = proj(lon, lat)
    return (x, y)

def read_finitefaultmodel(infile):

    """
    Read finite fault model info from file named infile.

    :param infile: text file 9 columns in CSV format
                 1. hypocentral time
                 2. latitude
                 3  longitude
                 4. depth (km) 
                 5. mxx
                 6. myy
                 7. mzz
                 8. mxy
                 9. mxz
                 10. myz

    :return ev_name: event name 
    :return hyp_time: hypocentral time
    :return lat, lon: latitude, longitude 
    :return dep: depth (m) 
    :return mxx, myy, mzz, mxy, mxz, myz: components of moment tensor

    """

    csvData = []
    with open(infile, 'r') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        for csvRow in csvReader:
            csvData.append(csvRow)

    csvData = np.array(csvData)
    x1, x2, x3, x4, x5 = csvData[:,0], csvData[:,1], csvData[:,2], csvData[:,3], csvData[:,4]
    x6, x7, x8, x9, x10 = csvData[:,5], csvData[:,6], csvData[:,7], csvData[:,8], csvData[:,9]
    
    hyp_time = []; lat = []; lon = []; dep = []; 
    mxx = []; myy = []; mzz = []; mxy = []; mxz = []; myz = [];
    
    for k in range(1,len(x1)):
        hyp_time.append(str(x1[k]))
        lat.append(float(x2[k]))
        lon.append(float(x3[k]))
        dep.append(float(x4[k]))
        mxx.append(float(x5[k]))
        myy.append(float(x6[k]))
        mzz.append(float(x7[k]))
        mxy.append(float(x8[k]))
        mxz.append(float(x9[k]))
        myz.append(float(x10[k]))
    
    return (hyp_time, lat, lon, dep, mxx, myy, mzz, mxy, mxz, myz)
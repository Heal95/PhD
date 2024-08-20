import os
import time
import pathlib
from pyproj import CRS
import HLS_modules as hls
from salvus import namespace as sn

PROJECT_DIR = "projectDF"

print("Reading and defining project domain from tomography model..")
# Define domain for salvus
try:
    dsxmin, dsxmax, dsymin, dsymax, dszmax = hls.get_domain_from_tomo('./DF_data/tomography_model.xyz')
    crs = CRS(proj='utm', zone=33, ellps='WGS84')
    d = sn.domain.dim3.UtmDomain(
        x0=dsxmin,
        x1=dsxmax,
        y0=dsymin,
        y1=dsymax,
        utm=crs,
    )

    # Define project
    p = sn.Project.from_domain(
        path=PROJECT_DIR,
        domain=d,
        load_if_exists=True,
    )
except Exception as e:
    print("Something went wrong..")

print("Done.")

print("Defining and adding volume model to simulation..")
ats_flag = True # Attenuation flag
# Define and add volume model to the project
try:
    CAS_model = sn.model.volume.cartesian.GenericModel(
        name="DF_model", data="./DF_data/DuFAULT_new.nc", mode="nearest"
    )
    p.add_to_project(CAS_model)

    # Implement model in the simulation configuration
    if ats_flag:
        ats = sn.LinearSolids(reference_frequency=1.0) # Attenuation settings
        mc = sn.model.ModelConfiguration(
            background_model=None, volume_models=["DF_model"], linear_solids=ats
        )
    else:
        mc = sn.model.ModelConfiguration(
            background_model=None, volume_models=["DF_model"]
        )
except Exception as e:
    print("Something went wrong..")

print("Done.")

print("Defining and adding topography and bathymetry to simulation..")
bathy_flag = False # Bathymetry flag
# Define and add topography and bathymetry to the project
try:
    topo_filename = "./DF_data/gmrt_topography.nc"
    check_file = os.path.isfile(topo_filename)
    if not check_file:
        print("Topography file missing - query data from the GMRT web service..")
        d.download_topography_and_bathymetry(
            filename=topo_filename,
            buffer_in_degrees=0.5,
            resolution="max",
        )

    topo = sn.topography.cartesian.SurfaceTopography.from_gmrt_file(
        name="DF_topo",
        data=topo_filename,
        resample_topo_nx=1000,
        decimate_topo_factor=5,
        utm=d.utm,
    )

    p.add_to_project(topo)
    tc = sn.topography.TopographyConfiguration(topography_models="DF_topo")

    if bathy_flag:
        bathy = sn.bathymetry.cartesian.OceanLayer.from_surface_topography(
            name="DF_bathy",
            topography=topo,
        )

        p.add_to_project(bathy)
        bc = sn.bathymetry.BathymetryConfiguration(bathymetry_models=["DF_bathy"])
except Exception as e:
    print("Something went wrong..")

print("Done.")

print("Reading and defining receivers and events for simulation..")
# Define receivers
try:
    sta, net, lat_sta, lon_sta, _ = hls.read_stations(infile="./DF_data/STATIONS")
    x_sta,y_sta = hls.convert_to_UTM(lon=lon_sta, lat=lat_sta, zone=33)
    recs = []

    for k in range(0,len(sta)):
        elv = topo.ds.sel(x=x_sta[k], y=y_sta[k], method='nearest')
        rec = sn.simple_config.receiver.cartesian.SideSetPoint3D(
            point=(x_sta[k], y_sta[k], float(elv.dem.values)),
            direction=(0, 0, 1),
            side_set_name="z1",
            station_code=sta[k],
            network_code=net[k],
            fields=["velocity"],
        )
        recs.append(rec)
except Exception as e:
    print("Something went wrong when defining receivers..")

# Define spatial characteristics of the events and add them and the receivers to the project
try:
    ev_name, hyp_time, lat_ev, lon_ev, dep, Mw, S, D, R = hls.read_events(infile="./DF_data/DF_CMT.txt")
    x_ev,y_ev = hls.convert_to_UTM(lon=lon_ev, lat=lat_ev, zone=33)

    for k in range(0,len(ev_name)):
        Mxx, Myy, Mzz, Mxy, Mxz, Myz = hls.sdr2cmt(Mw=Mw[k], S=S[k], D=D[k], R=R[k])
        elv = topo.ds.sel(x=x_ev[k], y=y_ev[k], method='nearest')

        src = sn.simple_config.source.cartesian.SideSetMomentTensorPoint3D(
            point=(x_ev[k], y_ev[k], dep[k]),
            direction=(0, 0, -1),
            side_set_name="z1",
            offset=-dep[k]+float(elv.dem.values),
            mxx= Mxx,
            myy= Myy,
            mzz= Mzz,
            myz= Myz,
            mxz= Mxz,
            mxy= Mxy,
        )

        p.add_to_project(sn.Event(event_name=str(ev_name[k]), sources=src, receivers=recs))
except Exception as e:
    print("Something went wrong when defining events..")

print("Done.")

print("Last few parametrizations..")
# Define time-frequency axis in the simulation configuration
max_freq = 1.0 # maximum frequency of the simulated waveforms
t_len = 90.0 # length of the simulated waveforms (s)

ec = sn.EventConfiguration(
    wavelet=sn.simple_config.stf.Ricker(center_frequency=max_freq/2),
    waveform_simulation_configuration=sn.WaveformSimulationConfiguration(
        end_time_in_seconds=t_len, attenuation=ats_flag 
    ),
)

# Add absorbing boundary to the project
vpmin, _, _, _, _, _ = hls.get_mvalues_from_tomo('./DF_data/tomography_model.xyz')

abp = sn.AbsorbingBoundaryParameters(
    reference_velocity=vpmin,
    number_of_wavelengths=3.5,
    reference_frequency=max_freq/2,
)

# Wrap everything up in the project
sim_name = "DF_test"
p.add_to_project(
    sn.SimulationConfiguration(
        name=sim_name,
        tensor_order=1,
        elements_per_wavelength=1.0,
        max_frequency_in_hertz=max_freq,
        model_configuration=mc,
        topography_configuration=tc,
        #bathymetry_configuration=bc,
        max_depth_in_meters=dszmax,
        absorbing_boundaries=abp,
        event_configuration=ec,
    ),
    overwrite=True,
)

print("Done with the setup - now launching simulation..")

CPUs = 512;
p.simulations.launch(
    ranks_per_job=CPUs,
    site_name="supek_cpu",
    events=p.events.list()[9], ##change?
    wall_time_in_seconds_per_job=3600*2, #3600 24h
    simulation_configuration=sim_name,

)

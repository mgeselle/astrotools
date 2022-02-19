#!/usr/bin/python3
# Registers and optionally stacks a set of images by first plate-solving them with astrometry.net,
# then computing the transform with reproject.
# When stacking the transformed images are co-added and the exposure time adjusted to the
# total exposure time. DATE-OBS is set to the middle of begin of first exposure and end of last
# exposure. JD-OBS is set to the julian date of the middle of the exposure. The airmass is adjusted
# to reflect the modified time.
# This requires reproject 0.5.1 or higher. Older versions will produce output
# images where all pixels are NAN, see issue #164 on github.
import astropy.units as u
import getopt
import numpy as np
import subprocess
import sys
import tempfile
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.io import fits
from astropy.time import Time, TimeDelta
from pathlib import Path
from reproject import reproject_exact


def usage_exit():
    print('Usage: {} -i input -r reg_dir -l low_size -h high_size [-o output]'.format(sys.argv[0]))
    sys.exit(1)


def plate_solve(p_in_dir, p_reg_dir, p_low_arc_min, p_high_arc_min, p_ra, p_dec):
    cmd_template = 'solve-field -D {} -p -N {} --overwrite -L {} -H {} -u aw -z 4 --crpix-center '
    if p_ra is not None and p_dec is not None:
        cmd_template = cmd_template + '--ra {} --dec {} --radius 2'
    cmd_template = cmd_template + ' {}'
    ret_val = True
    with tempfile.TemporaryDirectory() as tmp_dir:
        in_files = [x for x in p_in_dir.iterdir() if x.is_file() and x.suffix in ('.fit', '.fits')]
        in_files.sort()
        p_reg_dir.mkdir(parents=True, exist_ok=True)
        for in_file in in_files:
            if p_ra is not None and p_dec is not None:
                cmd = cmd_template.format(tmp_dir, p_reg_dir / in_file.name, p_low_arc_min, p_high_arc_min, p_ra, p_dec, in_file)
            else:
                cmd = cmd_template.format(tmp_dir, p_reg_dir / in_file.name, p_low_arc_min, p_high_arc_min, in_file)

            cmd_result = subprocess.run(cmd, shell=True)
            if cmd_result.returncode != 0:
                ret_val = False
        return ret_val


def register(p_reg_dir):
    in_files = [x for x in p_reg_dir.iterdir() if x.is_file()]
    in_files.sort()
    ref_file = in_files[0]
    ref_hdu = fits.open(ref_file)
    out_header = ref_hdu[0].header.copy()
    out_header['HISTORY'] = 'Reprojected to current WCS using reproject_exact'
    for in_file in in_files[1:]:
        in_hdu = fits.open(in_file)
        print('Reprojecting {}'.format(in_file))
        reprojected, _ = reproject_exact(in_hdu, ref_hdu[0].header, hdu_in=0, parallel=True)
        reprojected = np.nan_to_num(reprojected, nan=0)
        out_header['DATE-OBS'] = in_hdu[0].header['DATE-OBS']
        in_hdu.close()
        fits.writeto(in_file, reprojected, out_header, overwrite=True)
    ref_hdu.close()


def co_add(p_reg_dir, p_out_file):
    in_files = [x for x in p_reg_dir.iterdir() if x.is_file()]
    in_files.sort()
    print('Co-adding {:d} images.'.format(len(in_files)))
    ref_hdu = fits.open(in_files[0])
    out_data = ref_hdu[0].data.astype(np.float)
    out_header = ref_hdu[0].header.copy()
    out_header['HISTORY'] = 'Co-added {:d} images:'.format(len(in_files))
    hist_template = '... {} taken at {}'
    out_header['HISTORY'] = hist_template.format(in_files[0].name, out_header['DATE-OBS'])
    min_date_obs = Time(out_header['DATE-OBS'], format='isot', scale='utc')
    max_date_obs = min_date_obs
    for in_file in in_files[1:]:
        in_hdu = fits.open(in_file)
        out_data = out_data + in_hdu[0].data
        date_obs = Time(in_hdu[0].header['DATE-OBS'], format='isot', scale='utc')
        if date_obs < min_date_obs:
            min_date_obs = date_obs
        elif date_obs > max_date_obs:
            max_date_obs = date_obs
        out_header['HISTORY'] = hist_template.format(in_file.name, in_hdu[0].header['DATE-OBS'])
        in_hdu.close()
    ref_hdu.close()
    exposure = out_header['EXPTIME']
    out_header['EXPTIME'] = len(in_files) * exposure
    max_date_obs = max_date_obs + TimeDelta(exposure, format='sec')
    middle_date_obs = min_date_obs + (max_date_obs - min_date_obs) / 2
    print('Middle of observation is {}.'.format(middle_date_obs.value))
    out_header['DATE-OBS'] = middle_date_obs.value
    out_header.comments['DATE-OBS'] = 'UTC time at middle of observation'
    out_header['JD-OBS'] = middle_date_obs.jd
    out_header.comments['JD-OBS'] = 'Julian date at middle of observation'
    if 'SITELAT' in out_header.keys():
        centre_pos = SkyCoord(out_header['CRVAL1'], out_header['CRVAL2'], unit='deg')
        print('Centre position is {}'.format(centre_pos.to_string('hmsdms')))
        obs_loc = EarthLocation(lat=out_header['SITELAT'] * u.deg, lon=out_header['SITELONG'] * u.deg)
        # noinspection PyUnresolvedReferences
        print('Observatory is at {} {}.'.format(obs_loc.lat.to_string(u.degree, sep=':'),
                                                obs_loc.lon.to_string(u.degree, sep=':')))
        alt_az = centre_pos.transform_to(AltAz(obstime=middle_date_obs, location=obs_loc))
        out_header['AIRMASS'] = alt_az.secz.value
        print('Setting airmass to {:f}'.format(alt_az.secz.value))

    fits.writeto(p_out_file, out_data, out_header, overwrite=True)
    print('New image written to {}.'.format(p_out_file))


a_in_dir = None
a_reg_dir = None
a_low_arc_min = None
a_high_arc_min = None
a_out_file = None
a_ra = None
a_dec = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:r:l:h:o:R:D:')
    for opt, arg in opts:
        if opt == '-i':
            a_in_dir = Path(arg)
        elif opt == '-r':
            a_reg_dir = Path(arg)
        elif opt == '-l':
            a_low_arc_min = arg
        elif opt == '-h':
            a_high_arc_min = arg
        elif opt == '-o':
            a_out_file = Path(arg)
        elif opt == '-R':
            a_ra = arg
        elif opt == '-D':
            a_dec = arg
        else:
            print('Unsupported option: {}'.format(opt))
            usage_exit()

    if not a_in_dir:
        print('Input directory not specified.')
        usage_exit()

    if not a_in_dir.exists():
        print('Input directory does not exist.')
        usage_exit()

    if not a_reg_dir:
        print('Directory for registered images not specified.')
        usage_exit()

    if a_low_arc_min is None:
        print('Lower FOV size limit not specified.')
        usage_exit()

    if a_high_arc_min is None:
        print('Upper FOV size limit not specified.')
        usage_exit()

    if not plate_solve(a_in_dir, a_reg_dir, a_low_arc_min, a_high_arc_min, a_ra, a_dec):
        print('Plate solving failed.')
        sys.exit(1)

    register(a_reg_dir)

    if a_out_file is not None:
        co_add(a_reg_dir, a_out_file)


except getopt.GetoptError as err:
    print(err)
    usage_exit()

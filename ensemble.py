#!/usr/bin/python3
import aavso
import getopt
import imgutil
import numpy as np
import photparam
import sys
from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table, Row
from pathlib import Path
from typing import Tuple, Sequence, Dict, Union


class EnsembleException(Exception):
    def __init__(self, message):
        self.message = message


class VStarCatalog:
    def __init__(self, chart_id: str, pgm_table: Table, cmp_table: Table):
        self.chart_id: str = chart_id
        self.pgm_table: Table = pgm_table
        self.cmp_table: Table = cmp_table


def usage_exit():
    print('Usage: {} -c config [-C chart_id] imageA imageB'.format(sys.argv[0]))
    sys.exit(1)


def load_config(config_name: str) -> photparam.PhotParam:
    config_path = Path(config_name)
    if not config_path.exists():
        raise EnsembleException('configuration file not found')
    if not config_path.is_file():
        raise EnsembleException('not a file: {}'.format(config_name))
    try:
        return photparam.PhotParam(config_name)
    except photparam.PhotParamError as phot_err:
        raise EnsembleException(phot_err.message)


def load_and_check_images(files: Sequence[str]) -> Tuple[photparam.ImageData, photparam.ImageData]:
    images = load_images(files)
    sanity_check_images(*images)
    return images


def load_images(files: Sequence[str]) -> Tuple[photparam.ImageData, photparam.ImageData]:
    if len(files) != 2:
        raise EnsembleException('expected 2 input files, got {:d}'.format(len(files)))
    result = []
    for file in files:
        path = Path(file)
        if not path.exists():
            raise EnsembleException('input file {} not found'.format(file))
        if not path.is_file():
            raise EnsembleException('not a file: {}'.format(file))

        try:
            result.append(photparam.ImageData(path))
        except OSError as os_err:
            raise EnsembleException('error reading {}: {}'.format(file, os_err.strerror))
    result.sort(key=lambda img: img.channel.value)

    return result[0], result[1]


def sanity_check_images(image_a: photparam.ImageData, image_b: photparam.ImageData) -> None:
    if image_a.channel == image_b.channel:
        raise EnsembleException('images should be in distinct filters')
    x, y = image_a.wcs.world_to_pixel(image_b.center)
    shape = image_a.data.shape
    if x < 0 or y < 0 or x > shape[1] or y > shape[0]:
        raise EnsembleException('too little overlap between fields')


def query_catalog(image_a: photparam.ImageData, image_b: photparam.ImageData,
                  chart_id: str = None, max_mag_err: float = 0.05) -> VStarCatalog:
    pgm_stars: Table = query_program_stars(image_a, image_b)
    if len(pgm_stars) == 0:
        raise EnsembleException('no program stars found')
    kwargs = {
        'max_mag_err': max_mag_err
    }
    if chart_id is not None:
        kwargs['chart_id'] = chart_id

    q_chart_id, cmp_stars = query_comparison_stars(pgm_stars, image_a, image_b, **kwargs)
    if len(cmp_stars) == 0:
        raise EnsembleException('no comparison stars found')
    return VStarCatalog(q_chart_id, pgm_stars, cmp_stars)


def run_photometry(config: photparam.PhotParam, pgm_stars: Table, cmp_stars: Table,
                   image_a: photparam.ImageData, image_b: photparam.ImageData,
                   sat_limit: float = 0.8, min_sep: float = 12.0, min_snr: float = 40,
                   fwhm_px: Union[None, float] = None) -> aavso.Photometry:
    images = (image_a, image_b)
    pgm_tables, cmp_tables = find_sources_in_tables(pgm_stars, cmp_stars, images,
                                                    sat_limit=sat_limit, min_sep=min_sep, min_snr=min_snr,
                                                    fwhm_px=fwhm_px)
    # noinspection PyTypeChecker
    pgm_tbl = measure_stars(*pgm_tables, image_a, image_b, config, sat_limit=sat_limit)
    print_raw_photometry(pgm_tbl)
    # noinspection PyTypeChecker
    cmp_tbl = measure_stars(*cmp_tables, image_a, image_b, config, sat_limit=sat_limit)
    print_raw_photometry(cmp_tbl)
    discard_saturated(cmp_tbl)
    if len(cmp_tbl) < 2:
        raise EnsembleException('need at least two comparison stars')
    channels = [img.channel for img in images]
    channels.sort(key=lambda c: c.value)
    transformer = config.transformer(*channels)
    # noinspection PyTypeChecker
    phot_tbl = ensemble_photometry(pgm_tbl, cmp_tbl, *channels, transformer)
    transform_info = {}
    channel_names = []
    for channel in channels:
        transform_info[channel.name] = transformer.info(channel)
        channel_names.append(channel.name)
    return aavso.Photometry(channel_names, transform_info, phot_tbl)


def query_program_stars(image_a: photparam.ImageData, image_b: photparam.ImageData) -> Table:
    print('Querying VSX for program stars in a box of {:d} arc minutes centered on {}.'.format(
        image_a.box_size, image_a.center.to_string('hmsdms')))
    result = aavso.query_vsx(image_a.center, boxsize=image_a.box_size, incl_all=False)
    imgutil.get_coords_in_image(result, image_a.wcs, image_a.data.shape)
    imgutil.get_coords_in_image(result, image_b.wcs, image_b.data.shape)
    result['pos'].info.format = lambda a: a.to_string('hmsdms')
    result.remove_column('const')
    result.remove_column('x')
    result.remove_column('y')
    result.remove_column('disc')
    pass_b = []
    for minPass, maxPass in result.iterrows('maxPass', 'minPass'):
        pass_b.append(minPass + '/' + maxPass)
    result.remove_column('maxPass')
    result.remove_column('minPass')
    result.add_column(pass_b, name='pass', index=6)

    print('Query result:')
    result.pprint_all(max_width=132, show_unit=False)

    return result


def query_comparison_stars(pgm_stars: Table, image_a: photparam.ImageData, image_b: photparam.ImageData,
                           chart_id: str = None, max_mag_err: float = 0.05) -> Tuple[str, Table]:
    if chart_id is None:
        center_name, center_sep = get_closest_name(pgm_stars, image_a.center)
        print('Querying VSP with FOV of {:d} arc minutes around {}, {} from image centre.'.format(
            image_a.box_size,
            center_name,
            center_sep.to_string()
        ))
        qry_result = aavso.query_ref_stars(star=center_name, fov=image_a.box_size)
    else:
        print('Retrieving chart {} from VSP.'.format(chart_id))
        qry_result = aavso.query_ref_stars(chart_id=chart_id)

    res_tbl = qry_result['ref_stars']
    imgutil.get_coords_in_image(res_tbl, image_a.wcs, image_a.data.shape)
    imgutil.get_coords_in_image(res_tbl, image_b.wcs, image_b.data.shape)
    res_tbl.remove_column('x')
    res_tbl.remove_column('y')
    res_tbl['pos'].info.format = lambda a: a.to_string('hmsdms')
    channels = [image_a.channel.name, image_b.channel.name]
    for chl_name in [c.name for c in photparam.Channel if c.name not in channels]:
        if chl_name in res_tbl.colnames:
            res_tbl.remove_column(chl_name)
            res_tbl.remove_column(chl_name + '_err')
    for i in range(len(res_tbl) - 1, -1, -1):
        discard_reason = None
        for chl_name in channels:
            row = res_tbl[i]
            if res_tbl[chl_name].mask[i]:
                discard_reason = 'Discarding {}: missing {} magnitude.'
            elif float(row[chl_name + '_err']) == 0.0:
                discard_reason = 'Discarding {}: missing {} error.'
            elif float(row[chl_name + '_err']) > max_mag_err:
                discard_reason = 'Discarding {}: {} error greater than limit.'
            if discard_reason is not None:
                print(discard_reason.format(row['auid'], chl_name))
                res_tbl.remove_row(i)
                break

    print('Retrieved chart {}. Stars:'.format(qry_result['chartid']))
    res_tbl.pprint_all(max_width=132, show_unit=False)

    return qry_result['chartid'], res_tbl


def get_closest_name(candidates: Table, center: SkyCoord) -> Tuple[str, Angle]:
    min_sep = None
    min_name = None
    # noinspection PyTypeChecker
    for name, pos in candidates.iterrows('name', 'pos'):
        sep = center.separation(pos)
        if min_sep is None or sep < min_sep:
            min_sep = sep
            min_name = name
    return min_name, min_sep


def find_sources_in_tables(pgm_tbl: Table, cmp_tbl: Table, images: Tuple[photparam.ImageData, photparam.ImageData],
                           sat_limit: float = 0.8, min_sep: float = 12.0,
                           min_snr: float = 40.0,
                           fwhm_px: Union[None, float] = None) -> Tuple[Tuple[Table, Table], Tuple[Table, Table]]:
    pgm_tables = []
    cmp_tables = []
    for img in images:
        img.fwhm = imgutil.estimate_fwhm(img.data, int(min_sep / img.scale), saturation=sat_limit) \
            if fwhm_px is None else fwhm_px
        print('FWHM in {} is {:3.1f} pixels, {:3.1f} arc seconds.'.format(
            img.channel.name, img.fwhm, img.fwhm * img.scale))
        src_table = imgutil.find_sources(img.data, img.fwhm, min_snr=min_snr)
        pgm_tables.append(match_sources_to_stars(pgm_tbl, src_table, img))
        cmp_tables.append(match_sources_to_stars(cmp_tbl, src_table, img))
    print('Removing {} stars not in {} image'.format(images[0].channel.name, images[1].channel.name))
    remove_not_in_other(*pgm_tables)
    print('Removing {} stars not in {} image'.format(images[1].channel.name, images[0].channel.name))
    remove_not_in_other(pgm_tables[1], pgm_tables[0])
    print('Found {:d} of {:d} program stars in both images.'.format(len(pgm_tables[0]), len(pgm_tbl)))
    remove_not_in_other(*cmp_tables)
    remove_not_in_other(cmp_tables[1], cmp_tables[0])
    print('Found {:d} of {:d} comparison stars in both images.'.format(len(cmp_tables[0]), len(cmp_tbl)))
    return (pgm_tables[0], pgm_tables[1]), (cmp_tables[0], cmp_tables[1])


def match_sources_to_stars(in_table: Table, src_table: Table, img: photparam.ImageData) -> Table:
    star_table = Table(data=in_table, copy=True)
    imgutil.get_coords_in_image(star_table, img.wcs, img.data.shape)
    imgutil.match_sources(src_table, star_table, img.fwhm)
    return star_table


def remove_not_in_other(table_a: Table, table_b: Table):
    key = 'name' if 'name' in table_a.colnames else 'auid'
    in_other = set(table_b[key])
    for i in range(len(table_a) - 1, -1, -1):
        if table_a[i][key] not in in_other:
            print('Removing {}.'.format(table_a[i][key]))
            table_a.remove_rows(i)


def measure_stars(table_a: Table, table_b: Table,
                  image_a: photparam.ImageData, image_b: photparam.ImageData,
                  config: photparam.PhotParam,
                  sat_limit: float = 0.8) -> Table:
    for table, image in zip((table_a, table_b), (image_a, image_b)):
        aperture = imgutil.get_aperture(image.fwhm)
        imgutil.aperture_photometry(image.data, table, image.channel.name.lower(), aperture,
                                    gain=config.gain(),
                                    dark_charge=image.exposure * config.dark_current(),
                                    readout_noise=config.read_noise())
    result = Table(table_a, copy=True)
    result.remove_column('x')
    result.remove_column('y')
    result.remove_column('peak')
    b_channel = image_b.channel.name.lower()
    for col in (b_channel, b_channel + '_err'):
        result[col] = table_b[col]
    sat_flags = np.empty(len(table_a), dtype=np.bool_)
    limit_a = np.max(image_a.data) * sat_limit
    limit_b = np.max(image_b.data) * sat_limit
    # noinspection PyTypeChecker
    for i, peak_a, peak_b in zip(range(0, len(table_a)), table_a.iterrows('peak'), table_b.iterrows('peak')):
        sat_flags[i] = peak_a > limit_a or peak_b > limit_b
    result['sat'] = sat_flags
    return result


def print_raw_photometry(table: Table) -> None:
    col_names = set()
    if 'varType' in table.colnames:
        print('Raw photometry of program star(s)')
        col_names.update(('auid', 'name', 'varType', 'maxMag', 'minMag', 'min_dist_px'))
    else:
        print('Raw photometry of comparison star(s)')
        col_names.update(table.colnames)
        col_names.discard('pos')
        col_names.discard('snr')
    col_names.update(table.colnames[-5:])
    t = Table(table, copy=False)
    for c in table.colnames:
        if c not in col_names:
            t.remove_column(c)
    t.pprint(max_width=132)


def discard_saturated(table: Table) -> None:
    for i in range(len(table) - 1, -1, -1):
        row = table[i]
        if row['sat']:
            print('Discarding {}: star is saturated.'.format(row['auid']))
            table.remove_row(i)


def ensemble_photometry(pgm_table: Table, ref_table: Table,
                        channel_a: photparam.Channel, channel_b: photparam.Channel,
                        transformer: photparam.Transformer) -> Table:
    instr_chl_a = channel_a.name.lower()
    instr_chl_b = channel_b.name.lower()

    cmp_obs_lists = {
        channel_a: [],
        channel_b: []
    }
    for ref_row in ref_table[1:]:
        cmp_obs = row_to_obs(ref_row, instr_chl_a, instr_chl_b)
        cmp_ref = row_to_obs(ref_row, channel_a.name, channel_b.name)
        cmp_obs_lists[channel_a].append(photparam.CmpObs(cmp_obs[instr_chl_a], cmp_ref[channel_a.name]))
        cmp_obs_lists[channel_b].append(photparam.CmpObs(cmp_obs[instr_chl_b], cmp_ref[channel_b.name]))

    rows = []
    for pgm_row in pgm_table:
        pgm_obs = row_to_obs(pgm_row, instr_chl_a, instr_chl_b)
        obs_list_c1 = photparam.ObsList(channel_a, pgm_obs[instr_chl_a], cmp_obs_lists[channel_a])
        obs_list_c2 = photparam.ObsList(channel_b, pgm_obs[instr_chl_b], cmp_obs_lists[channel_b])
        rows.append(single_ensemble(pgm_row, obs_list_c1, obs_list_c2, transformer))
    chk_obs = row_to_obs(ref_table[0], instr_chl_a, instr_chl_b)
    obs_list_c1 = photparam.ObsList(channel_a, chk_obs[instr_chl_a], cmp_obs_lists[channel_a])
    obs_list_c2 = photparam.ObsList(channel_b, chk_obs[instr_chl_b], cmp_obs_lists[channel_b])
    rows.append(single_ensemble(ref_table[0], obs_list_c1, obs_list_c2, transformer))

    names = ['type', 'auid', 'name',
             instr_chl_a, instr_chl_a + '_err',
             instr_chl_b, instr_chl_b + '_err',
             channel_a.name, channel_a.name + '_err',
             channel_b.name, channel_b.name + '_err',
             'sat']
    result = Table(names=names, rows=rows)
    for chl in (channel_a, channel_b):
        for key in (chl.name, chl.name.lower()):
            result[key].info.format = '6.3f'
            result[key + '_err'].info.format = '5.3f'
    return result


def single_ensemble(row_in: Row, obs_list_c1: photparam.ObsList, obs_list_c2: photparam.ObsList,
                    transformer: photparam.Transformer) -> Dict:
    row = {'auid': row_in['auid']}
    if 'name' in row_in.colnames:
        row['name'] = row_in['name']
        row['type'] = 'P'
    else:
        row['type'] = 'C'
    row['sat'] = row_in['sat']
    for obs_list in (obs_list_c1, obs_list_c2):
        ref_mags = np.array(list(o.ref_val.val for o in obs_list.cmp_list))
        cmp_mags = np.array(list(o.obs_val.val for o in obs_list.cmp_list))
        instr_mag = obs_list.pgm.val + np.mean(ref_mags - cmp_mags)
        cmp_errs = np.array(list(o.ref_val.err ** 2 + o.obs_val.err ** 2 for o in obs_list.cmp_list))
        instr_err = np.sqrt(np.sum(cmp_errs) / (len(obs_list.cmp_list) ** 2))
        instr_chl = obs_list.channel.name.lower()
        row[instr_chl] = instr_mag
        row[instr_chl + '_err'] = instr_err
    mag_c1, mag_c2 = transformer.transform_ens(obs_list_c1, obs_list_c2)
    chl = obs_list_c1.channel.name
    row[chl] = mag_c1.val
    row[chl + '_err'] = mag_c1.err
    chl = obs_list_c2.channel.name
    row[chl] = mag_c2.val
    row[chl + '_err'] = mag_c2.err
    return row


def ensemble_photometry_single(pgm_obs: dict, ref_obs: Sequence[Dict], ref_mags: Sequence[Dict],
                               channel_a: photparam.Channel, channel_b: photparam.Channel,
                               transformer) -> Dict:

    result = {}
    for k in ('auid', 'name'):
        if k in pgm_obs:
            result[k] = pgm_obs[k]
    chl_obs = {}
    for chl in (channel_a, channel_b):
        instr_chl = chl.name.lower()
        chl_obs[chl] = [photparam.ChannelObs(chl, pgm_obs[instr_chl],
                                             ro[instr_chl],
                                             rm[chl.name]) for ro, rm in zip(ref_obs, ref_mags)]
        instr_mags = [co.instr_mag() for co in chl_obs[chl]]
        result[instr_chl] = np.mean(np.array([im.val for im in instr_mags]))
        result[instr_chl + '_err'] = np.sqrt(
            np.mean(np.array([im.err for im in instr_mags]) ** 2) / len(chl_obs[chl])
        )
    trans_mags = {
        channel_a: [],
        channel_b: []
    }
    for obs_a, obs_b in zip(chl_obs[channel_a], chl_obs[channel_b]):
        trans_ab = transformer.transform(obs_a, obs_b)
        trans_mags[channel_a].append(trans_ab[0])
        trans_mags[channel_b].append(trans_ab[1])
    for chl in (channel_a, channel_b):
        result[chl.name] = np.mean(np.array([m.val for m in trans_mags[chl]]))
        result[chl.name + '_err'] = np.sqrt(
            np.mean(np.array([m.err for m in trans_mags[chl]]) ** 2) / len(trans_mags[chl])
        )
    return result


def row_to_obs(row: Row, channel_a: str, channel_b: str) -> Dict:
    obs = {
        'auid': row['auid'],
    }
    if 'name' in row.colnames:
        obs['name'] = row['name']
    for channel in (channel_a, channel_b):
        obs[channel] = photparam.ObsValue(row[channel], row[channel + '_err'])
    return obs


# MAIN Main main

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'c:C:f:m:n:s:S:w:')

        config = None
        chart_id = None
        webobs_file = None
        max_mag_error = None
        min_snr = None
        saturation_lim = None
        min_sep = None
        fwhm_px = None
        for opt, arg in opts:
            if opt == '-c':
                config = load_config(arg)
            elif opt == '-C':
                chart_id = arg
            elif opt == '-f':
                webobs_file = arg
            elif opt == '-n':
                min_snr = float(arg)
            elif opt == '-m':
                max_mag_error = float(arg)
            elif opt == '-s':
                saturation_lim = float(arg)
            elif opt == '-S':
                min_sep = float(arg)
            elif opt == '-w':
                fwhm_px = float(arg)

        if config is None:
            print('Configuration file not specified')
            usage_exit()

        images = load_and_check_images(args)
        kwargs = {}
        if chart_id is not None:
            kwargs['chart_id'] = chart_id
        if max_mag_error is not None:
            kwargs['max_mag_err'] = max_mag_error
        catalog = query_catalog(*images, **kwargs)
        kwargs = {}
        if saturation_lim is not None:
            kwargs['sat_limit'] = saturation_lim
        if min_sep is not None:
            kwargs['min_sep'] = min_sep
        if min_snr is not None:
            kwargs['min_snr'] = min_snr
        if fwhm_px is not None:
            kwargs['fwhm_px'] = fwhm_px

        photometry = run_photometry(config, catalog.pgm_table, catalog.cmp_table, *images, **kwargs)

        photometry.phot_tbl.pprint(max_width=132)
        if webobs_file is not None:
            photometry.chart_id = chart_id
            photometry.jd_obs = (images[0].date_obs.jd + images[1].date_obs.jd) / 2
            photometry.airmass = (images[0].airmass + images[1].airmass) / 2
            photometry.obs_code = config.obscode()
            aavso.write_webobs_file(photometry, webobs_file)

    except getopt.GetoptError as err:
        print(err)
        usage_exit()
    except EnsembleException as err:
        print(err.message)
        sys.exit(1)


if __name__ == '__main__':
    main()

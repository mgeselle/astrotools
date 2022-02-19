#!/usr/bin/python3
#
# Utilities to work with the AAVSO DB and AAVSO file format.
#
from typing import Sequence

import requests
import xml.sax
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
from xml.sax.handler import ContentHandler


class AAVSOException(Exception):
    def __init__(self, message):
        self.message = message


class VOTableHandler(ContentHandler):
    """Handles output from a VSX VOTable query"""

    def __init__(self, incl_all=True):
        super().__init__()
        self.__incl_all = incl_all
        self.__rows = []
        self.__field_ids = []
        self.__capture = False
        self.__field_idx = -1
        self.__chars = ""
        self.__row = {}
        self.table = None

    def startElement(self, name, attrs):
        if name == 'FIELD':
            self.__field_ids.append(attrs.getValue('id'))
        elif name == 'TD':
            self.__field_idx += 1
            self.__capture = True

    def characters(self, content):
        if self.__capture:
            self.__chars += content

    def endElement(self, name):
        if name == 'TD':
            field_name = self.__field_ids[self.__field_idx]
            if field_name == 'radec2000':
                ra, dec = self.__chars.split(',')
                # Mapping radec2000 to pos for consistency
                self.__row['pos'] = SkyCoord(ra, dec, unit=u.deg)
            else:
                self.__row[field_name] = self.__chars
            self.__chars = ""
            self.__capture = False
        elif name == 'TR':
            if self.__incl_all or self.__row['auid'] != '':
                self.__rows.append(self.__row)
            self.__row = {}
            self.__field_idx = -1
        elif name == 'TABLEDATA':
            final_names = [n if n != 'radec2000' else 'pos' for n in self.__field_ids]
            self.table = Table(rows=self.__rows, names=final_names)
            self.__rows = []


class Photometry:
    def __init__(self, channels: Sequence[str], transform_info: dict, phot_tbl: Table):
        self.channels = channels
        self.transform_info = transform_info
        self.phot_tbl = phot_tbl
        self.chart_id = None
        self.obs_code = None
        self.jd_obs = None
        self.airmass = None


def query_ref_stars(star=None, pos=None, chart_id=None, fov=60, mag_limit=14.5):
    if star is None and pos is None and chart_id is None:
        raise AAVSOException('neither star nor position nor chart ID specified')

    if star is not None and pos is not None:
        raise AAVSOException('both star and position specified')

    url = 'https://www.aavso.org/apps/vsp/api/chart'
    params = dict()
    if chart_id is not None:
        url += '/' + chart_id + '/'
    else:
        if pos is not None:
            if isinstance(pos, str):
                # noinspection PyUnresolvedReferences
                pos = SkyCoord(pos, unit=(u.hourangle, u.deg))
            if not isinstance(pos, SkyCoord):
                raise AAVSOException('position must be either a string or a SkyCoord')
            params['ra'] = pos.ra.value
            params['dec'] = pos.dec.value
        else:
            params['star'] = star
        params['fov'] = fov
        params['maglimit'] = mag_limit
    params['format'] = 'json'

    rsp = requests.get(url, params=params)
    if rsp.status_code != requests.codes.ok:
        raise AAVSOException('error calling VSP: ' + rsp.json())

    json = rsp.json()
    rows = []
    filters = set()
    for ref_star in json['photometry']:
        # noinspection PyUnresolvedReferences
        row = {'auid': ref_star['auid'], 'pos': SkyCoord(ref_star['ra'], ref_star['dec'], unit=(u.hourangle, u.deg))}
        for band in [x for x in ref_star['bands'] if x['band'] in ('B', 'V', 'Rc', 'Ic')]:
            filter_id = band['band'][0:1]
            filters.add(filter_id)
            row[filter_id] = float(band['mag'])
            if band['error'] is not None:
                row[filter_id + '_err'] = float(band['error'])
        rows.append(row)
    names = ['auid', 'pos']
    for filter_id in ('B', 'V', 'R', 'I'):
        if filter_id in filters:
            names.append(filter_id)
            names.append(filter_id + '_err')
    # noinspection PyUnresolvedReferences
    result = {
        'name': json['star'],
        'auid': json['auid'],
        'chartid': json['chartid'],
        'pos': SkyCoord(json['ra'], json['dec'], unit=(u.hourangle, u.deg)),
        'ref_stars': Table(rows=rows, names=names, masked=True)
    }
    return result


def query_vsx(coords, boxsize=60, vtype=None, incl_all=True):
    if isinstance(coords, str):
        # noinspection PyUnresolvedReferences
        coords = SkyCoord(coords, unit=(u.hourangle, u.deg))
    elif not isinstance(coords, SkyCoord):
        raise AAVSOException('coordinates must be either a string or a SkyCoord instance')

    params = {
        'view': 'query.votable',
        'coords': '{:f} {:f}'.format(coords.ra.value, coords.dec.value),
        'format': 'd',
        'geom': 'b',
        'size': boxsize,
        'unit': '2'
    }
    if vtype is not None:
        params['vtype'] = vtype
    rsp = requests.get('https://www.aavso.org/vsx/index.php', params=params)
    if rsp.status_code != requests.codes.ok:
        raise AAVSOException('error calling VSX: ' + rsp.text)
    handler = VOTableHandler(incl_all)
    xml.sax.parseString(rsp.text, handler)
    return handler.table


def write_webobs_file(photometry: Photometry, file_name: str):
    with open(file_name,'w') as out:
        print('#TYPE=Extended', file=out)
        print('#OBSCODE={}'.format(photometry.obs_code), file=out)
        print('#SOFTWARE=Custom python package using astropy and photutils', file=out)
        delim = ','
        print('#DELIM={}'.format(delim), file=out)
        print('#DATE=JD', file=out)
        print('#OBSTYPE=CCD', file=out)

        check_star = None
        check_mags = {}
        for row in photometry.phot_tbl:
            if row['type'] == 'C':
                check_star = row['auid']
                for channel in photometry.channels:
                    check_mags[channel] = row[channel.lower()]
                break
        for row in photometry.phot_tbl:
            if row['type'] != 'P' or row['sat']:
                continue
            for channel in photometry.channels:
                items = [row['name']]
                jd = row['jd'] if 'jd' in row.colnames else photometry.jd_obs
                items.append('{:.4f}'.format(jd))
                items.append('{:.3f}'.format(row[channel]))
                items.append('{:.3f}'.format(row[channel + '_err']))
                items.append(channel)
                items.append('YES')
                items.append('STD')
                items.append('ENSEMBLE')
                items.append('na')
                items.append(check_star)
                items.append('{:.3f}'.format(check_mags[channel]))
                items.append('{:.3f}'.format(photometry.airmass))
                items.append('1')
                items.append(photometry.chart_id)
                items.append(photometry.transform_info[channel])
                print(delim.join(items), file=out)


#tbl = query_ref_stars('CHI Cyg')['ref_stars']
#tbl = query_vsx('21 55 57.03+48 20 52.5', incl_all=False)
#tbl.show_in_browser(jsviewer=True)



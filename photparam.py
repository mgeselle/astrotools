import astropy.io.fits as fits
import configparser as cfg
import numpy as np
from astropy import wcs
from astropy.time import Time
from math import log10, fabs, floor, sqrt
from os import PathLike
from enum import Enum
from string import Template
from typing import Union, Tuple, Sequence


class Channel(Enum):
    U = 0
    B = 1,
    V = 2,
    R = 3,
    I = 4


class ImageException(Exception):
    def __init__(self, message):
        self.message = message


class ImageData:
    """Holds data plus metadata of a single image"""
    def __init__(self, file: Union[str, PathLike]):
        with fits.open(file) as hdu_list:
            self.data = np.nan_to_num(hdu_list[0].data)
            header = hdu_list[0].header
            if 'FILTER' not in header:
                raise ImageException('FILTER keyword missing in header of {}'.format(file))
            chl_name = header['FILTER'].rstrip()
            try:
                self.channel = Channel[chl_name]
            except KeyError:
                raise ImageException('Unknown filter {} in {}'.format(chl_name, file))
            if 'EXPTIME' not in header:
                raise ImageException('EXPTIME keyword missing in header of {}'.format(file))
            self.exposure = float(header['EXPTIME'])
            if 'AIRMASS' not in header:
                raise ImageException('AIRMASS keyword missing in header of {}'.format(file))
            self.airmass = float(header['AIRMASS'])
            if 'DATE-OBS' not in header:
                raise ImageException('DATE-OBS keyword missing in header of {}'.format(file))
            self.date_obs = Time(header['DATE-OBS'], format='isot', scale='utc')
            try:
                self.wcs = wcs.WCS(header)
            except wcs.NoWcsKeywordsFoundError:
                raise ImageException('{} does not contain a WCS'.format(file))
            self.scale = np.sqrt((self.wcs.pixel_scale_matrix ** 2).sum(axis=0, dtype=float)).mean() * 3600
            shape = self.data.shape
            self.center = self.wcs.pixel_to_world(int(shape[1] / 2), int(shape[0] / 2))
            self.box_size = int(np.max(shape) * self.scale / 60)
            self.fwhm = None
            self.sources = None
            print('Loaded {} image from {}'.format(self.channel.name, file))


class ObsValue:
    """Holds a value and the associated error"""
    def __init__(self, val: float, err: float):
        self.val = val
        self.err = err

    def get_format_spec(self) -> str:
        log_val = 0 if self.val == 0.0 else int(log10(fabs(self.val)))
        log_err = 0 if self.err == 0.0 else log10(self.err)
        err_prec = 0 if log_err >= 0.0 else int(-floor(log_err)) + 1
        err_w = 1 + err_prec + (1 if log_err <= 0 else int(log_err) + 1)
        if err_prec == 0:
            err_w -= 1
        val_w = 1 + err_prec + (1 if log_val <= 0 else log_val + 1)
        tpl = Template('{:${vw}.${ep}f} ({:${ew}.${ep}f})')
        return tpl.substitute(vw=val_w, ep=err_prec, ew=err_w)

    def __str__(self) -> str:
        return self.get_format_spec().format(self.val, self.err)


class CmpObs:
    """A single observation of a comparison star together with the reference magnitude"""
    def __init__(self, obs_val: ObsValue, ref_val: ObsValue):
        self.obs_val = obs_val
        self.ref_val = ref_val


class ObsList:
    """Observation of a program star and its comparison stars in a single channel."""
    def __init__(self, channel: Channel, pgm: ObsValue, cmp_list: Sequence[CmpObs]):
        self.channel = channel
        self.pgm = pgm
        self.cmp_list = cmp_list


class ChannelObs:
    """Holds one observation in a single channel of a program and comparison star
    and the reference magnitude"""
    def __init__(self, channel: Channel, pgm: ObsValue, cmp: ObsValue, ref: ObsValue):
        self.channel = channel
        self.pgm = pgm
        self.cmp = cmp
        self.ref = ref

    def instr_mag(self) -> ObsValue:
        """Computes the instrumental magnitude"""
        m_instr = self.ref.val + self.pgm.val - self.cmp.val
        m_err = sqrt(self.ref.err ** 2 + self.pgm.err ** 2 + self.cmp.err ** 2)
        return ObsValue(m_instr, m_err)


class TransformerException(Exception):
    def __init__(self, message):
        self.message = message


class Transformer:
    def __init__(self, channels: Tuple[Channel, Channel],
                 epsilons: Tuple[ObsValue, ObsValue], mu: ObsValue, field_info: str):
        self._epsilons = {}
        for channel, epsilon in zip(channels, epsilons):
            self._epsilons[channel] = epsilon
        self._mu = mu
        self._info = {
            channels[0]: self._mk_info(channels[0], channels[1], field_info),
            channels[1]: self._mk_info(channels[1], channels[0], field_info)
        }

    def _mk_info(self, channel_a: Channel, channel_b: Channel, field_info: str) -> str:
        info = 'Transform coefficients from ' + field_info
        channels = [channel_a, channel_b]
        channels.sort(key=lambda c: c.value)
        chl_ab = (channels[0].name + channels[1].name).lower()
        chl_a = channel_a.name.lower()
        info += '. T_' + chl_a + chl_ab + ' = ' + str(self._epsilons[channel_a])
        info += ', T_' + chl_ab + ' = ' + str(self._mu)
        return info

    def transform(self, obs_c1: ChannelObs, obs_c2: ChannelObs) -> Tuple[ObsValue, ObsValue]:
        """Transforms a single differential observation to standard magnitudes"""
        if obs_c1.channel == obs_c2.channel:
            raise TransformerException('channels in observations should be distinct')
        for o in (obs_c1, obs_c2):
            if o.channel not in self._epsilons:
                raise TransformerException('Transformer not applicable for target channel ' + o.channel.name)
        mag_c1 = self._transform_channel(obs_c1, obs_c2)
        mag_c2 = self._transform_channel(obs_c2, obs_c1)
        return mag_c1, mag_c2

    def transform_ens(self, obs_list_c1: ObsList, obs_list_c2: ObsList) -> Tuple[ObsValue, ObsValue]:
        """Transforms ensemble differential observations into standard magnitudes.
        The point of this method is that the errors on the resulting magnitudes are not
        simply avg(mag single obs)/N**2: the observed magnitude of the program star
        enters into every differential observation, so by simply using the mean we'd be
        underestimating the contribution of the program star's error."""
        if obs_list_c1.channel == obs_list_c2.channel:
            raise TransformerException('channels in observations should be distinct')
        for o in (obs_list_c1, obs_list_c2):
            if o.channel not in self._epsilons:
                raise TransformerException('Transformer not applicable for target channel ' + o.channel.name)
        mag_c1 = self._transform_ens_chl(obs_list_c1, obs_list_c2)
        msg_c2 = self._transform_ens_chl(obs_list_c2, obs_list_c1)
        return mag_c1, msg_c2

    def _transform_channel(self, obs_c1: ChannelObs, obs_c2: ChannelObs) -> ObsValue:
        eps_v = self._epsilons[obs_c1.channel].val
        eps_err = self._epsilons[obs_c1.channel].err
        mu_v = self._mu.val
        mu_err = self._mu.err
        delta_col = obs_c1.pgm.val - obs_c2.pgm.val - obs_c1.cmp.val + obs_c2.cmp.val
        mag = obs_c1.instr_mag().val + eps_v * mu_v * delta_col
        mag_err = obs_c1.ref.err ** 2
        mag_err += ((1.0 + eps_v * mu_v) ** 2) * (obs_c1.pgm.err ** 2 + obs_c1.cmp.err ** 2)
        mag_err += ((eps_v * mu_v) ** 2) * (obs_c2.pgm.err ** 2 + obs_c2.cmp.err ** 2)
        mag_err += (delta_col ** 2) * ((mu_v * eps_err) ** 2 + (eps_v * mu_err) ** 2)
        return ObsValue(mag, sqrt(mag_err))

    def _transform_ens_chl(self, obs_list_c1: ObsList, obs_list_c2: ObsList) -> ObsValue:
        eps_val = self._epsilons[obs_list_c1.channel].val
        eps_err = self._epsilons[obs_list_c1.channel].err
        mu_val = self._mu.val
        mu_err = self._mu.err

        n_comp = len(obs_list_c1.cmp_list)
        mags = np.empty(n_comp)
        cmp_errs = np.empty(n_comp)
        cmp_col = np.empty(n_comp)
        pgm_col = obs_list_c2.pgm.val - obs_list_c1.pgm.val
        for i in range(0, n_comp):
            cmp_col[i] = obs_list_c2.cmp_list[i].obs_val.val - obs_list_c1.cmp_list[i].obs_val.val
            mags[i] = obs_list_c1.cmp_list[i].ref_val.val + obs_list_c1.pgm.val - obs_list_c1.cmp_list[i].obs_val.val
            mags[i] += eps_val * mu_val * (pgm_col - cmp_col[i])
            cmp_errs[i] = obs_list_c1.cmp_list[i].ref_val.err ** 2
            cmp_errs[i] += ((1.0 - eps_val * mu_val) * obs_list_c1.cmp_list[i].obs_val.err) ** 2
            cmp_errs[i] += (eps_val * mu_val * obs_list_c2.cmp_list[i].obs_val.err) ** 2
        tot_err = ((1.0 - eps_val * mu_val) * obs_list_c1.pgm.err) ** 2
        tot_err += (eps_val * mu_val * obs_list_c2.pgm.err) ** 2
        tot_err += np.mean(cmp_errs) / n_comp
        avg_delta_col = (pgm_col - np.mean(cmp_col)) ** 2
        tot_err += ((eps_val * mu_err) ** 2 + (eps_err * mu_val) ** 2) * avg_delta_col

        return ObsValue(float(np.mean(mags)), sqrt(tot_err))

    def info(self, channel: Channel) -> str:
        return self._info[channel]


class PhotParamError(Exception):
    def __init__(self, message):
        self.message = message


class PhotParam:

    def __init__(self, config_file: Union[str, bytes, PathLike]):
        self._raw_cfg = cfg.ConfigParser()
        try:
            self._raw_cfg.read(config_file)
        except cfg.MissingSectionHeaderError:
            raise PhotParamError('not a configuration file: {}'.format(config_file))
        except cfg.ParsingError:
            raise PhotParamError('error parsing configuration file')

    def obscode(self) -> str:
        return self._raw_cfg['Observatory'].get('OBSCODE')

    def gain(self) -> float:
        return self._raw_cfg['CCD'].getfloat('gain', fallback=1.0)

    def dark_current(self) -> float:
        return self._raw_cfg['CCD'].getfloat('dark_current', fallback=0.0)

    def read_noise(self) -> float:
        return self._raw_cfg['CCD'].getfloat('read_noise', fallback=0.0)

    def transformer(self, colour1: Channel, colour2: Channel) -> Transformer:
        channels = [colour1, colour2]
        channels.sort(key=lambda c: c.value)
        colour_key = channels[0].name + channels[1].name
        if not self._raw_cfg.has_section('Transform'):
            raise TransformerException('no transformation coefficients found')
        sect = self._raw_cfg['Transform']
        epsilons = []
        for chl in channels:
            key = chl.name + colour_key
            if key not in sect:
                raise TransformerException('coefficient ' + key + ' is missing')
            epsilons.append(ObsValue(sect.getfloat(key), sect.getfloat(key + '_err', fallback=0.0)))
        if colour_key not in sect:
            raise TransformerException('colour coefficient ' + colour_key + ' is missing')

        mu = ObsValue(sect.getfloat(colour_key), sect.getfloat(colour_key + '_err', fallback=0.0))
        field_info = sect.get('fields', 'unknown source')
        return Transformer((channels[0], channels[1]), (epsilons[0], epsilons[1]), mu, field_info)


if __name__ == '__main__':
    obs = ObsValue(0.12345678, 0.00227)
    print(obs)
    obs = ObsValue(123.12, 17.0)
    print(obs)
    obs = ObsValue(-10.0123, 0.01234)
    print(obs)
    pt = PhotParam('../../astro/atik-one.ini')
    xf = pt.transformer(Channel.B, Channel.V)
    print('Gain is {:5.3f}'.format(pt.gain()))

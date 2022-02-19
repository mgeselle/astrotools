import gauss
import numpy as np
from astropy import stats, wcs
from astropy.table import Table
from math import ceil, log, sqrt, trunc
from photutils import aperture, find_peaks, DAOStarFinder, CircularAperture, CircularAnnulus
from typing import Tuple, Union


def get_image_scale(p_wcs: wcs.WCS) -> float:
    """Gets the image scale in arc seconds per pixel"""
    return np.sqrt((p_wcs.pixel_scale_matrix ** 2).sum(axis=0, dtype=float)).mean() * 3600


def estimate_fwhm(p_img: np.ndarray, p_min_sep_px: int, saturation: float = 0.8) -> float:
    peaks = find_strongest_peaks(p_img, p_min_sep_px, saturation)
    raw_w = []
    # noinspection PyTypeChecker
    for py, px in peaks.iterrows('x_peak', 'y_peak'):
        pxl = px - p_min_sep_px
        pxh = px + p_min_sep_px
        pyl = py - p_min_sep_px
        pyh = py + p_min_sep_px
        sv = p_img[pxl:pxh, pyl:pyh]
        parms = gauss.fitgaussian(sv)
        raw_w.extend(parms[3:5])
    return np.array(raw_w).mean() * 2 * sqrt(-2 * log(0.5))


def find_sources(p_img: np.ndarray, p_fwhm: float, min_snr: float = 40.0) -> Union[Table, None]:
    """Finds sources in an image. Returns a table with these columns:
    x: x coordinate of centroid
    y: y coordinate of centroid
    round: roundness1 from daofind. Stars should have -1 <= round <= 1
    peak: peak ADU"""
    _, bg_median, stddev = stats.sigma_clipped_stats(p_img)
    in_img = p_img - bg_median
    finder = DAOStarFinder(fwhm=p_fwhm * 1.5, threshold=min_snr * stddev, exclude_border=True)
    raw_result = finder.find_stars(in_img)
    if raw_result is None:
        return None
    _, _, apt_size = get_aperture(p_fwhm)
    y_max, x_max = p_img.shape
    x_max -= apt_size
    y_max -= apt_size
    rows = []
    for x, y, rnd, peak in raw_result.iterrows('xcentroid', 'ycentroid', 'roundness1', 'peak'):
        if apt_size <= x < x_max and apt_size <= y < y_max:
            row = {
                'x': x,
                'y': y,
                'round': rnd,
                'peak': peak
            }
            rows.append(row)
    result = Table(rows=rows, names=['x', 'y', 'round', 'peak'])
    result.sort(['x', 'y'])
    return result


def get_aperture(p_fwhm: float) -> Tuple[int, int, int]:
    src_rad = ceil(1.5 * p_fwhm)
    inner = src_rad + trunc(1.5 * p_fwhm)
    outer = inner + src_rad

    return int(src_rad), int(inner), int(outer)


def find_strongest_peaks(p_img: np.ndarray, p_min_sep_px: int, saturation: float, num_peaks: int = 10) -> Table:
    _, bg_median, _ = stats.sigma_clipped_stats(p_img)
    peaks = find_peaks(p_img, 10 * bg_median, box_size=p_min_sep_px, border_width=p_min_sep_px)
    sat_thresh = saturation * p_img.max()
    peaks.sort(keys='peak_value', reverse=True)
    first_ok = 0
    for pv in peaks.iterrows('peak_value'):
        first_ok += 1
        if pv < sat_thresh:
            break
    peaks.remove_rows(list(range(0, first_ok)))
    if len(peaks) > num_peaks:
        peaks.remove_rows(list(range(num_peaks, len(peaks))))
    return peaks


def get_coords_in_image(p_table: Table, p_wcs: wcs.WCS, p_shape: Tuple[int, int]) -> None:
    """Scans a table of objects with sky coordinates.
    Adds the columns x and y containing the image coordinates to the input table.
    Rows outside the image will be removed."""
    not_in_img = []
    x_vals = []
    y_vals = []
    if 'x' in p_table.colnames:
        p_table.remove_column('x')
    if 'y' in p_table.colnames:
        p_table.remove_column('y')
    y_max, x_max = p_shape
    index = 0
    # noinspection PyTypeChecker
    for pos in p_table.iterrows('pos'):
        x, y = [a for a in p_wcs.world_to_pixel(pos[0])]
        if 0 <= x < x_max and 0 <= y < y_max:
            x_vals.append(x)
            y_vals.append(y)
        else:
            not_in_img.append(index)
        index += 1
    p_table.remove_rows(not_in_img)
    pos_idx = p_table.index_column('pos')
    p_table.add_column(y_vals, name='y', index=pos_idx)
    p_table.add_column(x_vals, name='x', index=pos_idx)


def match_sources(p_in_tbl: Table, p_ref_tbl: Table, p_fwhm: float) -> None:
    p_ref_tbl.sort(['x', 'y'])
    curr_index = 0
    peak_vals = []
    min_dist = []
    in_tbl_start = 0
    max_delta2 = p_fwhm ** 2
    # noinspection PyTypeChecker
    for x_ref, y_ref in p_ref_tbl.iterrows('x', 'y'):
        d2_min = -1
        x_min = -1
        y_min = -1
        peak_min = -1
        for in_idx in range(in_tbl_start, len(p_in_tbl)):
            x_in = p_in_tbl[in_idx]['x']
            dx2 = (x_in - x_ref) ** 2
            y_in = p_in_tbl[in_idx]['y']
            dy2 = (y_in - y_ref) ** 2
            d2 = dx2 + dy2
            if d2_min < 0 or d2 < d2_min:
                d2_min = d2
                x_min = x_in
                y_min = y_in
                peak_min = p_in_tbl[in_idx]['peak']
            if x_in < x_ref and dx2 > max_delta2:
                continue
            elif x_in > x_ref and dx2 > max_delta2:
                break
            if d2 < max_delta2:
                in_tbl_start = in_idx
                break
        p_ref_tbl[curr_index]['x'] = x_min
        p_ref_tbl[curr_index]['y'] = y_min
        peak_vals.append(peak_min)
        min_dist.append(np.sqrt(d2_min))
        curr_index += 1
    p_ref_tbl.add_column(peak_vals, name='peak')
    p_ref_tbl.add_column(min_dist, name='min_dist_px')
    p_ref_tbl['min_dist_px'].info.format = '.0f'

def aperture_photometry(image: np.ndarray, sources: Table, channel_name: str, aperture_params: Tuple[int, int, int],
                        gain: float = 1.0,
                        dark_charge: float = 0.0, readout_noise: float = 0.0) -> None:
    """Performs aperture photometry on a number of sources in an image.
    Parameters:
        image: input image
        sources: table containing sources. The columns x and y should contain the centroid coordinates.
        channel_name: name of the channel for instrumental magnitudes. The sources table will be augmented
                      by the columns <channel_name> and <channel_name>_err containing the instrumental
                      magnitudes and their errors.
        aperture_params: tuple of the radii of the aperture and the inner and outer annulus limits
        gain: CCD gain in e-/ADU
        dark_charge: total dark charge in e-
        readout_noise: read-out noise in e-
    Note: most of this has been shamelessly copied from the example in the photutils documentation"""
    positions = np.transpose((sources['x'], sources['y']))
    apertures = CircularAperture(positions, r=aperture_params[0])
    annuli = CircularAnnulus(positions, r_in=aperture_params[1], r_out=aperture_params[2])
    ann_masks = annuli.to_mask(method='center')

    bkg_median = []
    for mask in ann_masks:
        ann_data = mask.multiply(image)
        ann_data_vec = ann_data[mask.data > 0]
        _, ann_median, _ = stats.sigma_clipped_stats(ann_data_vec)
        bkg_median.append(ann_median * gain)
    bkg_median = np.array(bkg_median)
    phot_tbl = aperture.aperture_photometry(image * gain, apertures)
    background = bkg_median * apertures.area
    signal = phot_tbl['aperture_sum'].data - background
    noise = np.sqrt(signal +
                    apertures.area * (apertures.area/annuli.area + 1) * (bkg_median + dark_charge + readout_noise ** 2))
    sources['snr'] = signal / noise
    sources['snr'].info.format = '.1f'
    sources[channel_name] = np.log10(signal) * (-2.5)
    sources[channel_name].info.format = '7.3f'
    sources[channel_name + '_err'] = (noise / signal) * (2.5 / log(10.0))
    sources[channel_name + '_err'].info.format = '5.3f'

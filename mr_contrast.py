import numpy as np

# Transient steady state magnetisation
def transient_steady_state_mag(M0, Ms, Tr, T1, alpha_degree, npulse):
    E1 = np.exp(-Tr / T1)
    C = np.cos(alpha_degree/180*np.pi)

    print('Calculating transient steady state magnetisation!')

    Mzn_min = np.zeros(npulse)
    for ind in range(npulse):
        # Enter the formula for Mzn below:
        Mzn_min[ind] = 0
    return(Mzn_min)


# Transformation from k-space to image-space
def k2i(kdat):
    if len(kdat.shape) == 1:  # Carry out 1D FFT: k-space -> image space
        im = (np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kdat, (0)),
                                            (kdat.shape[0],), (0,), norm=None), (0,)))
    else:  # Carry  out 2D FFT: k-space -> image space
        im = (np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kdat, (0, 1)),
                                            (kdat.shape[0], kdat.shape[1]), (0, 1), norm=None), (0, 1)))

    return (im)


# Transformation from image-space to k-space
def i2k(im):
    if len(im.shape) == 1:  # Carry out 1D FFT: image space -> k-space
        kdat = (np.fft.fftshift(np.fft.fftn(np.fft.fftshift(im, (0)),
                                            (im.shape[0],), (0,), norm=None), (0,)))
    else:  # Carry out 2D FFT: image space -> k-space
        kdat = (np.fft.fftshift(np.fft.fftn(np.fft.fftshift(im, (0, 1)),
                                            (im.shape[0], im.shape[1]), (0, 1), norm=None), (0, 1)))

    return (kdat)


# Create different sampling patterns
def kspace_sampling(npe, order_strg):

    if order_strg == 'linear':
        k_order = np.linspace(0, npe - 1, npe).astype(np.int)

    elif order_strg == 'low_high':
        k_order = np.zeros(npe, dtype=np.int)
        for ind in range(npe // 2):
            k_order[2 * ind] = np.int(npe // 2 + ind)
            k_order[2 * ind + 1] = np.int(npe // 2 - ind - 1)

    elif order_strg == 'high_low':
        k_order = np.zeros(npe, dtype=np.int)
        for ind in range(npe // 2):
            k_order[2 * ind] = np.int(ind)
            k_order[2 * ind + 1] = np.int(npe - ind - 1)

    else:
        raise NameError('order_strg {:} not recognised'.format(order_strg,))
    return(k_order)


# Create different segmented sampling patterns
def kspace_sampling_segm(npe, order_strg, nsegm):
    if len(npe) % nsegm > 0:
        raise KeyError('Number of k-space points (npe) has to be a multiple of number of segments (nsegm)')

    # Calculate number of k-space samples per segment
    npe_segm = npe // nsegm
    k_order = np.zeros((npe_segm, nsegm), dtype=np.int)
    for knd in range(nsegm):
        if order_strg == 'linear':
            k_order[:, knd] = np.linspace(0, npe_segm - 1, npe_segm).astype(np.int)

        elif order_strg == 'low_high':

            for ind in range(npe_segm // 2):
                k_order[2 * ind, knd] = np.int(npe_segm // 2 + ind)
                k_order[2 * ind + 1, knd] = np.int(npe_segm // 2 - ind - 1)

        elif order_strg == 'high_low':

            for ind in range(npe_segm // 2):
                k_order[2 * ind, knd] = np.int(ind)
                k_order[2 * ind + 1, knd] = np.int(npe_segm - ind - 1)

        else:
            raise NameError('order_strg {:} not recognised'.format(order_strg,))
    return(k_order)


# Map signal to 1D k-space
def map_sig_2_kspace(tss_mag, k_order, npulse_start_acq=0):
    if len(tss_mag) < len(k_order) + npulse_start_acq:
        raise KeyError('Length of tss_mag has to be >= len(k_order) + npulse_start_acq')

    k_space = np.zeros(len(k_order))
    for ind in range(len(k_order)):
        k_space[k_order[ind]] = tss_mag[npulse_start_acq+ind]
    return(k_space)


# Create phantom data
def create_ph_data(nxy=200):
    # Four circles: fat, gray matter, white matter, csf
    val = [300, 600, 1000, 3000]

    # Create phantom
    ph = np.zeros((nxy, nxy), dtype=np.float32)
    xx, yy = np.meshgrid(np.linspace(-nxy // 2, nxy // 2 - 1, nxy), np.linspace(-nxy // 2, nxy // 2 - 1, nxy))


    # Calculate size of tubes
    rad = nxy // 8

    # Go through tubes
    start_ctr = [- 2 * rad, - 2 * rad]
    for ind in range(2):
        start_ctr[0] = - 2 * rad
        for jnd in range(2):
            # Circles
            idx = np.where(((xx - start_ctr[0]) ** 2 + (yy - start_ctr[1]) ** 2) < rad ** 2)
            ph[idx[0], idx[1]] += val[jnd + 2 * ind]

            # Structure inside circles
            nlines = 5
            gap_pix = 3
            for knd in range(5):
                ph[nxy//2 + start_ctr[1]-rad//4*3: nxy//2 + start_ctr[1],
                    nxy//2 + start_ctr[0] - (nlines-1)//2*gap_pix + knd*gap_pix] = 100
                ph[nxy // 2 + start_ctr[1] + (knd + 1) * gap_pix,
                    nxy // 2 + start_ctr[0] - rad // 2: nxy // 2 + start_ctr[0] + rad // 2] = 100


            start_ctr[0] += 3 * rad
        start_ctr[1] += 3 * rad
    return (ph)


# Simulate data acquisition of a 2D phantom
def ph_2_kspace(ph, k_order, npulse_start_acq, M0, Ms, Tr, alpha_degree):
    if ph.shape[0] != ph.shape[1]:
        raise KeyError('Image has to be square')

    if ph.shape[0] != len(k_order):
        raise KeyError('Image dimension has to be the same as len(k_order)')

    unique_t1 = np.unique(ph)
    kspace = np.zeros(ph.shape, np.complex128)
    for knd in range(len(unique_t1)):
        if unique_t1[knd] > 0:
            # Create an image containing only voxels for a given T1 value
            cim = np.zeros(ph.shape)
            cim[np.where(ph == unique_t1[knd])] = ph[np.where(ph == unique_t1[knd])]

            # Transform this image to k-space
            ckspace = i2k(cim)

            # Calculate weighting of k-space based on transient signal variation
            Msig = transient_steady_state_mag(M0, Ms, Tr, unique_t1[knd], alpha_degree, npulse_start_acq + len(k_order))

            # Calculate weighting given the k-space order
            kweight = map_sig_2_kspace(Msig, k_order, npulse_start_acq)

            # Apply weighting to k-space
            ckspace = ckspace * np.tile(kweight[:,np.newaxis], (1,len(k_order)))

            # Add to k-space
            kspace += ckspace

    return(kspace)

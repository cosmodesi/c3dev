#!/usr/bin/env python
# This code is located at: /project/projectdirs/desi/users/cblake/lensing
# First run: source /project/projectdirs/desi/software/desi_environment.sh

import sys
sys.path.insert(0, '/project/projectdirs/desi/mocks/desiqa/cori/lib/python3.6/site-packages/')
import numpy as np
import healpy as hp
from scipy.interpolate import splev, splrep
from scipy.spatial import cKDTree
from astropy.io import fits

def main():
  istage = 1 # Stage of mock challenge
# Loop over Buzzard regions
#  ireglst = [0] # Stage 0
#  ireglst = [5] # Stage 1
#  for ireg in ireglst:
# Generate Stage 0 mock challenge
#    if (istage == 0):
# Write out catalogues for 1000 sq deg Buzzard regions
#      writestage0rawcat(ireg)
# Generate generic source and lens mock catalogue
#      genstage0mock(ireg)
# Generate Stage 1 mock challenge
#    elif (istage == 1):
# Write out catalogues for 1000 sq deg Buzzard regions
#      writestage1rawcat(ireg)
# Generate generic source and lens mock catalogue
#      genstage1mock(ireg)
# Generate calibration sample
  if (istage == 0):
    genstage0calsamp()
  else:
    genstage1calsamp()
  return

########################################################################
# Write out raw fits catalogues for 1000 sq deg Buzzard regions.       # 
########################################################################

def writestage0rawcat(ireg):
  print('\nReading in raw Buzzard catalogue for region',ireg+1,'...')
#  stem = '/Users/cblake/Data/desi/buzzard/'
  stem = '/project/projectdirs/desi/mocks/buzzard/buzzard_v1.6_desicut/lensed/8/'
  buzzardfile = 'Buzzard_v1.6_lensed'
  ipixlst = getdesibuzzpix(ireg+1)
  rasall,decall,zspecall,g1all,g2all = np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
  for ipix in ipixlst:
    grp = int(ipix/100)
#    infile = stem + buzzardfile + '-8-' + str(ipix) + '.fits'
    infile = stem + str(grp) + '/' + str(ipix) + '/' + buzzardfile + '-8-' + str(ipix) + '.fits'
    print(infile)
    hdulist = fits.open(infile)
    table = hdulist[1].data
    ras = table.field('RA')
    dec = table.field('DEC')
    zspec = table.field('Z')
    g1 = table.field('GAMMA1')
    g2 = table.field('GAMMA2')
    hdulist.close()
    rasall = np.concatenate((rasall,ras))
    decall = np.concatenate((decall,dec))
    zspecall = np.concatenate((zspecall,zspec))
    g1all = np.concatenate((g1all,g1))
    g2all = np.concatenate((g2all,g2))
  col1 = fits.Column(name='RA',format='D',array=rasall)
  col2 = fits.Column(name='DEC',format='D',array=decall)
  col3 = fits.Column(name='Z',format='E',array=zspecall)
  col4 = fits.Column(name='GAMMA1',format='E',array=g1all)
  col5 = fits.Column(name='GAMMA2',format='E',array=g2all)
  hdulist = fits.BinTableHDU.from_columns([col1,col2,col3,col4,col5])
  outfile = buzzardfile + '-merge-reg' + str(ireg+1) + '.fits'
  print('Writing out combined data file...')
  print(outfile)
  hdulist.writeto(outfile)
  return

########################################################################
# Write out raw fits catalogues for 1000 sq deg Buzzard regions        #
# including magnitudes.                                                #
########################################################################

def writestage1rawcat(ireg):
  print('\nReading in raw Buzzard catalogue for region',ireg+1,'...')
#  stem = '/Users/cblake/Data/desi/buzzard/'
  stem = '/project/projectdirs/desi/mocks/buzzard/buzzard_v2.0/buzzard-4/i25_lensing_cat/8/'
  buzzardfile = 'Buzzard_v2.0_lensed'
  ipixlst = getdesibuzzpix(ireg+1)
  rasall,decall,zspecall,g1all,g2all,gmagall,rmagall,imagall,zmagall,ymagall = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
  for ipix in ipixlst:
    grp = int(ipix/100)
#    infile = stem + buzzardfile + '-8-' + str(ipix) + '.fits'
    infile = stem + str(grp) + '/' + str(ipix) + '/' + buzzardfile + '-8-' + str(ipix) + '.fits'
    print(infile)
    hdulist = fits.open(infile)
    table = hdulist[1].data
    ras = table.field('RA')
    dec = table.field('DEC')
    zspec = table.field('Z')
    g1 = table.field('GAMMA1')
    g2 = table.field('GAMMA2')
    mags = table.field('TMAG')
    hdulist.close()
    rasall = np.concatenate((rasall,ras))
    decall = np.concatenate((decall,dec))
    zspecall = np.concatenate((zspecall,zspec))
    g1all = np.concatenate((g1all,g1))
    g2all = np.concatenate((g2all,g2))
    gmagall = np.concatenate((gmagall,mags[:,0]))
    rmagall = np.concatenate((rmagall,mags[:,1]))
    imagall = np.concatenate((imagall,mags[:,2]))
    zmagall = np.concatenate((zmagall,mags[:,3]))
    ymagall = np.concatenate((ymagall,mags[:,4]))
  col1 = fits.Column(name='RA',format='D',array=rasall)
  col2 = fits.Column(name='DEC',format='D',array=decall)
  col3 = fits.Column(name='Z',format='E',array=zspecall)
  col4 = fits.Column(name='GAMMA1',format='E',array=g1all)
  col5 = fits.Column(name='GAMMA2',format='E',array=g2all)
  col6 = fits.Column(name='GMAG',format='E',array=gmagall)
  col7 = fits.Column(name='RMAG',format='E',array=rmagall)
  col8 = fits.Column(name='IMAG',format='E',array=imagall)
  col9 = fits.Column(name='ZMAG',format='E',array=zmagall)
  col10 = fits.Column(name='YMAG',format='E',array=ymagall)
  hdulist = fits.BinTableHDU.from_columns([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10])
  outfile = buzzardfile + '-merge-reg' + str(ireg+1) + '.fits'
  print('Writing out combined data file...')
  print(outfile)
  hdulist.writeto(outfile)
  return

########################################################################
# Generate generic source mock catalogue for each 1000 sq deg region.  #
########################################################################

def genstage0mock(ireg):
  print('\nGenerating Stage 0 mock for region',ireg+1,'...')
# Read in Buzzard BGS and LRG data and random catalogues in this region
  rasbgsdat,decbgsdat,zspecbgsdat,nbgsdat = readrawlensdatcat(1,1,ireg+1)
  raslrgdat,declrgdat,zspeclrgdat,nlrgdat = readrawlensdatcat(1,2,ireg+1)
  rasbgsran,decbgsran,zspecbgsran,nbgsran = readrawlensrancat(1,1,ireg+1)
  raslrgran,declrgran,zspeclrgran,nlrgran = readrawlensrancat(1,2,ireg+1)
# Read in Buzzard source catalogue in this region
  rassource,decsource,zspecsource,g1source,g2source,nsource = readstage0rawsourcecat(ireg+1)
# Cut source catalogue within redshift limits
  zminsource,zmaxsource = 0.,2.
  keep = (zspecsource > zminsource) & (zspecsource < zmaxsource)
  rassource,decsource,zspecsource,g1source,g2source = rassource[keep],decsource[keep],zspecsource[keep],g1source[keep],g2source[keep]
  nsource = len(rassource)
  print('Cut to',nsource,'sources with',zminsource,'< z <',zmaxsource)
# Subsample source catalogue to required redshift distribution
#  area = 53.7
  area = 1020.58
  keep = dosubsamp(zspecsource,zminsource,zmaxsource,area)
  rassource,decsource,zspecsource,g1source,g2source = rassource[keep],decsource[keep],zspecsource[keep],g1source[keep],g2source[keep]
  nsource = len(rassource)
# Apply shape noise    
  sige = 0.28
  e1source,e2source = appshapenoise(g1source,g2source,sige)
# Apply Gaussian photo-z error
  sigz = 0.1
  zphotsource = dophotzscat(zspecsource,sigz,zminsource,zmaxsource)
# Write lens catalogue in tomographic bins
  for itom in range(4):
    print('Generating lens catalogue in tomographic bin',itom+1)
    if (itom == 0):
      zmin,zmax = 0.1,0.3
      ext = '_BGS_zs0pt1_0pt3'
    elif (itom == 1):
      zmin,zmax = 0.3,0.5
      ext = '_BGS_zs0pt3_0pt5'
    elif (itom == 2):
      zmin,zmax = 0.5,0.7
      ext = '_LRG_zs0pt5_0pt7'
    elif (itom == 3):
      zmin,zmax = 0.7,0.9
      ext = '_LRG_zs0pt7_0pt9'
    if ((itom == 0) or (itom == 1)):
      raslensdat,declensdat,zspeclensdat = rasbgsdat,decbgsdat,zspecbgsdat
      raslensran,declensran,zspeclensran = rasbgsran,decbgsran,zspecbgsran
    else:
      raslensdat,declensdat,zspeclensdat = raslrgdat,declrgdat,zspeclrgdat
      raslensran,declensran,zspeclensran = raslrgran,declrgran,zspeclrgran
    cut = (zspeclensdat > zmin) & (zspeclensdat < zmax)
    nlensdat1 = len(raslensdat[cut])
    weilensdat = np.ones(nlensdat1)
    print('Cut to',nlensdat1,'data lenses in range',zmin,'< z <',zmax)
    outfile = 'stage0mock_reg' + str(ireg+1) + '_lenses' + ext + '.dat'
    writemocklensascii(outfile,raslensdat[cut],declensdat[cut],zspeclensdat[cut],weilensdat)
    cut = (zspeclensran > zmin) & (zspeclensran < zmax)
    nlensran1 = len(raslensran[cut])
    weilensran = np.ones(nlensran1)
    print('Cut to',nlensran1,'random lenses in range',zmin,'< z <',zmax)
    outfile = 'stage0mock_reg' + str(ireg+1) + '_randlenses' + ext + '.dat'
    writemocklensascii(outfile,raslensran[cut],declensran[cut],zspeclensran[cut],weilensran)
# Write source catalogue in tomographic bins
  for itom in range(4):
    print('Generating source catalogue in tomographic bin',itom+1)
    if (itom == 0):
      zmin,zmax = 0.5,0.7
      ext = '_zp0pt5_0pt7'
    elif (itom == 1):
      zmin,zmax = 0.7,0.9
      ext = '_zp0pt7_0pt9'
    elif (itom == 2):
      zmin,zmax = 0.9,1.1
      ext = '_zp0pt9_1pt1'
    elif (itom == 3):
      zmin,zmax = 1.1,1.5
      ext = '_zp1pt1_1pt5'
    cut = (zphotsource > zmin) & (zphotsource < zmax)
    nsource1 = len(rassource[cut])
    weisource = np.ones(nsource1)
    print('Cut to',nsource1,'sources in range',zmin,'< z <',zmax)
    outfile = 'stage0mock_reg' + str(ireg+1) + '_sources' + ext + '.dat'
    writestage0mocksourceascii(outfile,rassource[cut],decsource[cut],zspecsource[cut],zphotsource[cut],g1source[cut],g2source[cut],e1source[cut],e2source[cut],weisource)
  return

########################################################################
# Generate source mock catalogue for each 1000 sq deg region           #
# tailored to weak lensing surveys (KiDS, DES, HSC).                   #
########################################################################

def genstage1mock(ireg):
  print('\nGenerating Stage 1 mock for region',ireg+1,'...')
  area = 53.7
#  area = 1020.58
# Read in Buzzard BGS and LRG data and random catalogues in this region
  rasbgsdat,decbgsdat,zspecbgsdat,nbgsdat = readrawlensdatcat(2,1,ireg+1)
  raslrgdat,declrgdat,zspeclrgdat,nlrgdat = readrawlensdatcat(2,2,ireg+1)
  rasbgsran,decbgsran,zspecbgsran,nbgsran = readrawlensrancat(2,1,ireg+1)
  raslrgran,declrgran,zspeclrgran,nlrgran = readrawlensrancat(2,2,ireg+1)
# Read in Buzzard source catalogue in this region
  rassource,decsource,zspecsource,g1source,g2source,gmagsource,rmagsource,imagsource,zmagsource,ymagsource,nsource = readstage1rawsourcecat(ireg+1)
# Cut source catalogue within redshift limits
  zminsource,zmaxsource = 0.,2.
  keep = (zspecsource > zminsource) & (zspecsource < zmaxsource)
  rassource,decsource,zspecsource,g1source,g2source,gmagsource,rmagsource,imagsource,zmagsource,ymagsource = rassource[keep],decsource[keep],zspecsource[keep],g1source[keep],g2source[keep],gmagsource[keep],rmagsource[keep],imagsource[keep],zmagsource[keep],ymagsource[keep]
  nsource = len(rassource)
  print('Cut to',nsource,'sources with',zminsource,'< z <',zmaxsource)
# Loop over weak lensing surveys
  for isurv in range(3):
# Read in KiDS data
    if (isurv == 0):
      print('\nGenerating tailored mock for KiDS...')
      gmagsurv,rmagsurv,imagsurv,zmagsurv,ymagsurv,weisurv,zphotsurv,nsurv = readkidssurv()
      weical,zspeccal,zphotcal,ncal = readkidscal()
# Read in DES data
    elif (isurv == 1):
      print('\nGenerating tailored mock for DES...')
      rmagsurv,imagsurv,zmagsurv,weisurv,r11surv,r22surv,zphotsurv,nsurv = readdessurv()
      weical,zspeccal,zphotcal,ncal = readdescal()
# Read in HSC data
    elif (isurv == 2):
      print('\nGenerating tailored mock for HSC...')
      gmagsurv,rmagsurv,imagsurv,zmagsurv,ymagsurv,weisurv,mcorrsurv,ermssurv,zphotsurv,nsurv = readhscsurv()
      weical,zspeccal,zphotcal,ncal = readhsccal()
# Construct spec-z vs phot-z scatter from calibration sample
    zsmin,zsmax,nzs,zpmin,zpmax,nzp = 0.,2.,40,0.,2.,40
    probzszp = getprobzszp(zspeccal,zphotcal,zsmin,zsmax,nzs,zpmin,zpmax,nzp)
# Apply this scatter pattern to mocks
    zphotsource = dophotzdraw(zspecsource,probzszp,zsmin,zsmax,nzs,zpmin,zpmax,nzp)
# Find weight for Buzzard sources in nearest neighbour data using KDTree
    neigh = 10
    mcorrsource,ermssource,r11source,r22source = np.empty(nsource),np.empty(nsource),np.empty(nsource),np.empty(nsource)
    if (isurv == 0):
      magssource = np.dstack([gmagsource,rmagsource,imagsource,zmagsource,ymagsource])[0]
      magssurv = np.dstack([gmagsurv,rmagsurv,imagsurv,zmagsurv,ymagsurv])[0]
      weisource,temp1,temp2 = findnearestsurv(magssource,magssurv,weisurv,weisurv,weisurv,neigh)
    elif (isurv == 1):
      magssource = np.dstack([rmagsource,imagsource,zmagsource])[0]
      magssurv = np.dstack([rmagsurv,imagsurv,zmagsurv])[0]
      weisource,r11source,r22source = findnearestsurv(magssource,magssurv,weisurv,r11surv,r22surv,neigh)
      weisource = np.ones(nsource)
    elif (isurv == 2):
      magssource = np.dstack([gmagsource,rmagsource,imagsource,zmagsource,ymagsource])[0]
      magssurv = np.dstack([gmagsurv,rmagsurv,imagsurv,zmagsurv,ymagsurv])[0]
      weisource,mcorrsource,ermssource = findnearestsurv(magssource,magssurv,weisurv,mcorrsurv,ermssurv,neigh)
# Randomly subsample mocks in each tomographic bin
    print('\nRandomly subsample mocks in each tomographic bin...')
# Survey properties for KiDS
    if (isurv == 0):
      ntom = 5
      zplims = np.array([0.1,0.3,0.5,0.7,0.9,1.2])
      neffcal = np.array([0.836,1.464,2.962,2.149,2.139])
      sigetom = np.array([0.276,0.269,0.290,0.281,0.294])
      mcorrtom = np.array([-0.017,-0.008,-0.015,0.010,0.006])
# Survey properties for DES
    elif (isurv == 1):
      ntom = 4
      zplims = np.array([0.2,0.43,0.63,0.9,1.3])
      neffcal = np.array([1.51,1.54,1.62,0.83])
      sigetom = np.array([0.26,0.29,0.27,0.29])
# Survey properties for HSC
    elif (isurv == 2):
      ntom = 4
      zplims = np.array([0.3,0.6,0.9,1.2,1.5])
      neffcal = np.array([5.5,5.5,4.2,2.4])
    zpbin = np.digitize(zphotsource,zplims) - 1
    keep = np.repeat(False,nsource)
    for itom in range(ntom):
      nsource1 = len(weisource[zpbin==itom])
      neff = float(nsource1)/(3600.*area)
      prob = neffcal[itom]/neff
      print(zplims[itom],zplims[itom+1],neff,neffcal[itom])
      ran = np.random.rand(nsource1)
      keep1 = (ran < prob)
      keep[zpbin==itom] = keep1
    rassource1,decsource1,zspecsource1,g1source1,g2source1,zphotsource1,weisource1,mcorrsource1,ermssource1,r11source1,r22source1 = rassource[keep],decsource[keep],zspecsource[keep],g1source[keep],g2source[keep],zphotsource[keep],weisource[keep],mcorrsource[keep],ermssource[keep],r11source[keep],r22source[keep]
    nsource1 = len(rassource1)
    print('Cut to',nsource1,'sources')
# Apply shape noise and calibration corrections
    if (isurv == 0):
# KiDS: x=mcorr, y=dummy
      e1source1,e2source1,xsource1 = appshapecalkids(g1source1,g2source1,zphotsource1,mcorrtom,sigetom,zplims)
      ysource1 = np.zeros(nsource1)
      csurv = '_kids'
    elif (isurv == 1):
# DES: x=R11, y=R22
      e1source1,e2source1 = appshapecaldes(g1source1,g2source1,zphotsource1,r11source1,r22source1,sigetom,zplims)
      xsource1,ysource1 = r11source1,r22source1
      csurv = '_des'
    elif (isurv == 2):
# HSC: x=mcorr, y=erms
      e1source1,e2source1 = appshapecalhsc(g1source1,g2source1,weisource1,mcorrsource1,ermssource1)
      xsource1,ysource1 = mcorrsource1,ermssource1
      csurv = '_hsc'
# Write source catalogue in tomographic bins
    for itom in range(ntom):
      print('Generating source catalogue in tomographic bin',itom+1,'...')
      zmin,zmax = zplims[itom],zplims[itom+1]
      czmin1 = str(int(zmin))
      i = int(100.*zmin)-100*int(zmin)
      czmin2 = str(np.where(i>0,str(i),'00'))
      czmax1 = str(int(zmax))
      i = int(100.*zmax)-100*int(zmax)
      czmax2 = str(np.where(i>0,str(i),'00'))
      ext = '_zp' + czmin1 + 'pt' + czmin2 + '_' + czmax1 + 'pt' + czmax2
      cut = (zphotsource1 > zmin) & (zphotsource1 < zmax)
      nsource1 = len(rassource1[cut])
      print('Cut to',nsource1,'sources in range',zmin,'< z <',zmax)
      outfile = 'stage1mock_reg' + str(ireg+1) + '_sources' + csurv + ext + '.dat'
      writestage1mocksourceascii(outfile,rassource1[cut],decsource1[cut],zspecsource1[cut],zphotsource1[cut],g1source1[cut],g2source1[cut],e1source1[cut],e2source1[cut],weisource1[cut],xsource1[cut],ysource1[cut],isurv)
# Write lens catalogue in tomographic bins
  for itom in range(4):
    print('Generating lens catalogue in tomographic bin',itom+1)
    if (itom == 0):
      zmin,zmax = 0.1,0.3
      ext = '_BGS_zs0pt1_0pt3'
    elif (itom == 1):
      zmin,zmax = 0.3,0.5
      ext = '_BGS_zs0pt3_0pt5'
    elif (itom == 2):
      zmin,zmax = 0.5,0.7
      ext = '_LRG_zs0pt5_0pt7'
    elif (itom == 3):
      zmin,zmax = 0.7,0.9
      ext = '_LRG_zs0pt7_0pt9'
    if ((itom == 0) or (itom == 1)):
      raslensdat,declensdat,zspeclensdat = rasbgsdat,decbgsdat,zspecbgsdat
      raslensran,declensran,zspeclensran = rasbgsran,decbgsran,zspecbgsran
    else:
      raslensdat,declensdat,zspeclensdat = raslrgdat,declrgdat,zspeclrgdat
      raslensran,declensran,zspeclensran = raslrgran,declrgran,zspeclrgran
    cut = (zspeclensdat > zmin) & (zspeclensdat < zmax)
    nlensdat1 = len(raslensdat[cut])
    weilensdat = np.ones(nlensdat1)
    print('Cut to',nlensdat1,'data lenses in range',zmin,'< z <',zmax)
    outfile = 'stage1mock_reg' + str(ireg+1) + '_lenses' + ext + '.dat'
    writemocklensascii(outfile,raslensdat[cut],declensdat[cut],zspeclensdat[cut],weilensdat)
    cut = (zspeclensran > zmin) & (zspeclensran < zmax)
    nlensran1 = len(raslensran[cut])
    weilensran = np.ones(nlensran1)
    print('Cut to',nlensran1,'random lenses in range',zmin,'< z <',zmax)
    outfile = 'stage1mock_reg' + str(ireg+1) + '_randlenses' + ext + '.dat'
    writemocklensascii(outfile,raslensran[cut],declensran[cut],zspeclensran[cut],weilensran)
  return

########################################################################
# Generate stage 0 calibration sample (subset of spec-z and photo-z).  #
########################################################################

def genstage0calsamp():
  ncal = 100000 # Number of sources in calibration sample
  ireg = 0      # Region of calibration sample
  for itom in range(4):
    print('\nGenerating calibration sample for tomographic bin',itom+1)
    if (itom == 0):
      ext = '_zp0pt5_0pt7'
    elif (itom == 1):
      ext = '_zp0pt7_0pt9'
    elif (itom == 2):
      ext = '_zp0pt9_1pt1'
    elif (itom == 3):
      ext = '_zp1pt1_1pt5'
    infile = 'stage0mocks/stage0mock_reg' + str(ireg+1) + '_sources' + ext + '.dat'
# Read in source catalogue
    print('Reading in source catalogue...')
    print(infile)
    f = open(infile,'r')
    lines = f.readlines()[3:]
    zspec,zphot = [],[]
    for line in lines:
      fields = line.split()
      zspec.append(float(fields[2]))
      zphot.append(float(fields[3]))
    f.close()
    nsrc = len(zspec)
    print(nsrc,'sources read in')
    zspec,zphot = np.array(zspec),np.array(zphot)
# Randomly subsample source catalogue to form calibration sample
    keep = np.random.choice(nsrc,ncal,replace=False)
    zspeccal,zphotcal = zspec[keep],zphot[keep]
# Write out calibration sample
    outfile = 'stage0mock_cal_sources' + ext + '.dat'
    print('Writing out calibration sample...')
    print(outfile)
    f = open(outfile,'w')
    f.write('# Calibration sample\n')
    f.write('# z_spec, z_phot\n')
    f.write('{:d}'.format(ncal) + '\n')
    for i in range(ncal):
      f.write('{:7.5f} {:7.5f}'.format(zspeccal[i],zphotcal[i]) + '\n')
    f.close()
  return

########################################################################
# Generate stage 1 calibration sample (subset of spec-z and photo-z).  #
########################################################################

def genstage1calsamp():
  ncal = 100000 # Number of sources in calibration sample
  ireg = 5      # Region of calibration sample
  stem = 'stage1mocks/'
#  stem = '/Users/cblake/Data/desi/buzzard/'
  for isurv in range(3):
    if (isurv == 0):
      print('\nGenerating calibration sample for KiDS...')
      ntom = 5
      zplims = np.array([0.1,0.3,0.5,0.7,0.9,1.2])
      csurv = '_kids'
    elif (isurv == 1):
      print('\nGenerating calibration sample for DES...')
      ntom = 4
      zplims = np.array([0.2,0.43,0.63,0.9,1.3])
      csurv = '_des'
    elif (isurv == 2):
      print('\nGenerating calibration sample for HSC...')
      ntom = 4
      zplims = np.array([0.3,0.6,0.9,1.2,1.5])
      csurv = '_hsc'
    for itom in range(ntom):
      print('\nGenerating calibration sample for tomographic bin',itom+1)
      zmin,zmax = zplims[itom],zplims[itom+1]
      czmin1 = str(int(zmin))
      i = int(100.*zmin)-100*int(zmin)
      czmin2 = str(np.where(i>0,str(i),'00'))
      czmax1 = str(int(zmax))
      i = int(100.*zmax)-100*int(zmax)
      czmax2 = str(np.where(i>0,str(i),'00'))
      ext = '_zp' + czmin1 + 'pt' + czmin2 + '_' + czmax1 + 'pt' + czmax2
      infile = stem + 'stage1mock_reg' + str(ireg+1) + '_sources' + csurv + ext + '.dat'
# Read in source catalogue
      print('Reading in source catalogue...')
      print(infile)
      f = open(infile,'r')
      lines = f.readlines()[3:]
      zspec,zphot,wei,mcorr,r11,r22,erms = [],[],[],[],[],[],[]
      for line in lines:
        fields = line.split()
        zspec.append(float(fields[2]))
        zphot.append(float(fields[3]))
        if (isurv == 0):
          wei.append(float(fields[8]))
          mcorr.append(float(fields[9]))
          r11.append(0.)
          r22.append(0.)
          erms.append(0.)
        elif (isurv == 1):
          r11.append(float(fields[9]))
          r22.append(float(fields[10]))
          wei.append(0.)
          mcorr.append(0.)
          erms.append(0.)
        elif (isurv == 2):
          wei.append(float(fields[8]))
          mcorr.append(float(fields[9]))
          erms.append(float(fields[10]))
          r11.append(0.)
          r22.append(0.)
      f.close()
      nsrc = len(zspec)
      print(nsrc,'sources read in')
      zspec,zphot,wei,mcorr,r11,r22,erms = np.array(zspec),np.array(zphot),np.array(wei),np.array(mcorr),np.array(r11),np.array(r22),np.array(erms)
# Randomly subsample source catalogue to form calibration sample
      keep = np.random.choice(nsrc,ncal,replace=False)
      zspeccal,zphotcal,weical,mcorrcal,r11cal,r22cal,ermscal = zspec[keep],zphot[keep],wei[keep],mcorr[keep],r11[keep],r22[keep],erms[keep]
# Write out calibration sample
      outfile = 'stage1mock_cal_sources' + csurv + ext + '.dat'
      print('Writing out calibration sample...')
      print(outfile)
      f = open(outfile,'w')
      if (isurv == 0):
        f.write('# KiDS calibration sample\n')
        f.write('# z_spec, z_phot, wei, mcorr\n')
        f.write('{:d}'.format(ncal) + '\n')
        for i in range(ncal):
          f.write('{:7.5f} {:7.5f} {:8.5f} {:8.5f}'.format(zspeccal[i],zphotcal[i],weical[i],mcorrcal[i]) + '\n')
      elif (isurv == 1):
        f.write('# DES calibration sample\n')
        f.write('# z_spec, z_phot, R_11, R_22\n')
        f.write('{:d}'.format(ncal) + '\n')
        for i in range(ncal):
          f.write('{:7.5f} {:7.5f} {:8.5f} {:8.5f}'.format(zspeccal[i],zphotcal[i],r11cal[i],r22cal[i]) + '\n')
      elif (isurv == 2):
        f.write('# HSC calibration sample\n')
        f.write('# z_spec, z_phot, wei, mcorr, e_rms\n')
        f.write('{:d}'.format(ncal) + '\n')
        for i in range(ncal):
          f.write('{:7.5f} {:7.5f} {:8.5f} {:8.5f} {:8.5f}'.format(zspeccal[i],zphotcal[i],weical[i],mcorrcal[i],ermscal[i]) + '\n')
      f.close()
  return

########################################################################
# Divide Buzzard mocks into 1000 sq deg regions.                       #
########################################################################

def getdesibuzzpix(ireg):
  if (ireg == 0):
    ipixlst = [128]
  elif (ireg == 1):
    ipixlst = [340,341,342,395,396,398,399,417,418,419,420,421,422,424,425,426,637,638,639]
  elif (ireg == 2):
    ipixlst = [64,65,66,67,68,72,343,349,351,423,427,428,429,430,431,432,434,435,440]
  elif (ireg == 3):
    ipixlst = [69,70,71,73,74,75,76,77,78,80,81,82,96,97,98,373,441,442,443]
  elif (ireg == 4):
    ipixlst = [79,83,86,88,89,90,91,92,99,100,101,102,103,104,105,106,107,108,112]
  elif (ireg == 5):
    ipixlst = [94,95,109,110,111,113,114,115,116,117,118,119,120,121,122,123,124,125,126]
  elif (ireg == 6):
    ipixlst = [391,397,400,401,402,403,404,405,406,408,409,410,483,488,489,490,701,702,703]
  elif (ireg == 7):
    ipixlst = [128,129,130,131,136,407,411,412,413,414,415,433,436,486,487,491,492,493,494]
  elif (ireg == 8):
    ipixlst = [132,133,134,135,137,138,139,140,141,142,160,161,162,437,438,439,444,445,446]
  elif (ireg == 9):
    ipixlst = [84,85,87,143,152,154,155,163,164,165,166,167,168,169,170,171,172,176,447]
  elif (ireg == 10):
    ipixlst = [93,158,173,174,175,177,178,179,180,181,182,183,184,185,186,187,188,189,190]
  elif (ireg == 11):
    ipixlst = [144,145,146,147,148,150,153,232,495,498,499,504,505,506,507,508,509,510,511]
  elif (ireg == 12):
    ipixlst = [149,151,156,157,159,234,235,236,237,238,239,248,249,250,251]
  elif (ireg == 13):
    ipixlst = [262,263,265,266,267,268,269,270,288,289,290,291,296,468,759,764,765,766,767]
  elif (ireg == 14):
    ipixlst = [260,261,272,273,274,352,353,354,559,565,566,567,568,569,570,571,572,573,574]
  elif (ireg == 15):
    ipixlst = [192,193,194,271,282,292,293,294,295,297,298,299,300,301,302,304,306,469,471]
  elif (ireg == 16):
    ipixlst = [0,275,276,277,278,279,280,281,283,284,285,286,305,355,360,361,362,363,575]
  elif (ireg == 17):
    ipixlst = [195,196,197,198,199,208,209,210,211,212,303,307,312,313,314,315,316,318,319]
  elif (ireg == 18):
    ipixlst = [2,3,8,9,10,11,12,14,32,33,34,35,36,40,287,308,309,310,311,317]
  else:
    print('Unknown region!!')
    sys.exit()
  return ipixlst

########################################################################
# Read in Buzzard source data version 1, for generic survey.           #
########################################################################

def readstage0rawsourcecat(ireg):
  stem = 'mergedcats/'
  buzzardfile = 'Buzzard_v1.6_lensed'
  infile = buzzardfile + '-merge-reg' + str(ireg) + '.fits'
  print('\nReading in Buzzard source catalogue...')
  print(stem+infile)
  hdulist = fits.open(stem+infile)
  table = hdulist[1].data
  ras = table.field('RA')
  dec = table.field('DEC')
  zspec = table.field('Z')
  g1 = table.field('GAMMA1')
  g2 = table.field('GAMMA2')
  hdulist.close()
  ngal = len(ras)
  print('Read in',ngal,'objects')
  return ras,dec,zspec,g1,g2,ngal

########################################################################
# Read in Buzzard source data version 2, for tailored survey.          #
########################################################################

def readstage1rawsourcecat(ireg):
#  stem = '/Users/cblake/Data/desi/buzzard/'
  stem = 'mergedcats/'
  buzzardfile = 'Buzzard_v2.0_lensed'
  infile = buzzardfile + '-merge-reg' + str(ireg) + '.fits'
  print('\nReading in Buzzard source catalogue...')
  print(stem+infile)
  hdulist = fits.open(stem+infile)
  table = hdulist[1].data
  ras = table.field('RA')
  dec = table.field('DEC')
  zspec = table.field('Z')
  g1 = table.field('GAMMA1')
  g2 = table.field('GAMMA2')
  gmag = table.field('GMAG')
  rmag = table.field('RMAG')
  imag = table.field('IMAG')
  zmag = table.field('ZMAG')
  ymag = table.field('YMAG')
  hdulist.close()
  ngal = len(ras)
  print('Read in',ngal,'objects')
  return ras,dec,zspec,g1,g2,gmag,rmag,imag,zmag,ymag,ngal

########################################################################
# Read in Buzzard lens data.                                           #
########################################################################

def readrawlensdatcat(iver,lensopt,ireg):
#  stem = '/Users/cblake/Data/desi/buzzard/'
  if (iver == 1):
    stem = '/project/projectdirs/desi/users/shadaba/Buzzard/'
  else:
    stem = '/project/projectdirs/desi/mocks/buzzard/buzzard_v2.0/buzzard-4/DESI_tracers/'
  if (lensopt == 1):
    ctype = 'BGS'
  else:
    ctype = 'LRG'
  print('\nReading in DESI Buzzard',ctype,'data...')
  infile = stem + 'buzzard_' + ctype + '.fits'
  print(infile)
  hdulist = fits.open(infile)
  table = hdulist[1].data
  ras = table.field('RA')
  dec = table.field('DEC')
  red = table.field('Z_COSMO') + table.field('DZ_RSD')
  hdulist.close()
  ngal = len(ras)
  print('Read in',ngal,'data lenses')
  theta,phi = np.radians(90.-dec),np.radians(ras)
  nside = 8
  ipix = hp.ang2pix(nside,theta,phi,nest=True)
  ipixlst = getdesibuzzpix(ireg)
  cut = np.isin(ipix,ipixlst)
  ras,dec,red = ras[cut],dec[cut],red[cut]
  ngal = len(ras)
  print('Cut to',ngal,'objects inside pixels')
  return ras,dec,red,ngal

########################################################################
# Read in Buzzard lens randoms.                                        #
########################################################################

def readrawlensrancat(iver,lensopt,ireg):
#  stem = '/Users/cblake/Data/desi/buzzard/'
  if (iver == 1):
    stem = '/project/projectdirs/desi/users/shadaba/Buzzard/'
  else:
    stem = '/project/projectdirs/desi/mocks/buzzard/buzzard_v2.0/buzzard-4/DESI_tracers/'
  if (lensopt == 1):
    ctype = 'BGS'
  else:
    ctype = 'LRG'
  print('\nReading in DESI Buzzard',ctype,'randoms...')
  ipixlst = getdesibuzzpix(ireg)
  nside = 8
  rsets = 9
  rasall,decall,redall = np.array([]),np.array([]),np.array([])
  for iset in range(rsets):
    print('Reading in DESI Buzzard',ctype,'random set',iset+1,'...')
    infile = stem + 'buzzard_' + ctype + '_rand.0' + str(iset+1) + '.fits'
    print(infile)
    hdulist = fits.open(infile)
    table = hdulist[1].data
    ras = table.field('RA')
    dec = table.field('DEC')
    red = table.field('Z_COSMO') + table.field('DZ_RSD')
    hdulist.close()
    nran = len(ras)
    print('Read in',nran,'random lenses')
    theta,phi = np.radians(90.-dec),np.radians(ras)
    ipix = hp.ang2pix(nside,theta,phi,nest=True)
    cut = np.isin(ipix,ipixlst)
    ras,dec,red = ras[cut],dec[cut],red[cut]
    nran = len(ras)
    print('Cut to',nran,'objects inside pixels')
    rasall = np.concatenate((rasall,ras))
    decall = np.concatenate((decall,dec))
    redall = np.concatenate((redall,red))
  nranall = len(rasall)
  print(nranall,'total randoms')
  return rasall,decall,redall,nranall

########################################################################
# Read in KiDS source magnitudes and weights.                          #
########################################################################

def readkidssurv():
  nsamp = 1000000
#  stem = '/Users/cblake/Data/desi/buzzard/'
  stem = 'lenscats/'
  infile = stem + 'kids_mag.fits'
  print('\nReading in KiDS source catalogue...')
  print(infile)
  hdulist = fits.open(infile)
  table = hdulist[1].data
  gmag = table.field('MAG_GAAP_g')
  rmag = table.field('MAG_GAAP_r')
  imag = table.field('MAG_GAAP_i')
  zmag = table.field('MAG_GAAP_Z')
  ymag = table.field('MAG_GAAP_Y')
  wei = table.field('weight')
  zb = table.field('Z_B')
  hdulist.close()
  cut = (zb > 0.1) & (zb < 1.2)
  gmag,rmag,imag,zmag,ymag,wei,zb = gmag[cut],rmag[cut],imag[cut],zmag[cut],ymag[cut],wei[cut],zb[cut]
  ngal = len(gmag)
  print(ngal,'KiDS sources read in')
  cut = np.random.choice(ngal,nsamp,replace=False)
  gmag,rmag,imag,zmag,ymag,wei,zb = gmag[cut],rmag[cut],imag[cut],zmag[cut],ymag[cut],wei[cut],zb[cut]
  ngal = len(gmag)
  print('Cut to',ngal,'sources')
  zb += np.random.uniform(-0.005,0.005,ngal)
  return gmag,rmag,imag,zmag,ymag,wei,zb,ngal

########################################################################
# Read in DES metacal source magnitudes and weights.                   #
########################################################################

def readdessurv():
  nsamp = 1000000
#  stem = '/Users/cblake/Data/desi/buzzard/'
  stem = 'lenscats/'
  infile = stem + 'des_metacal_mag.fits'
  print('\nReading in DES source catalogue...')
  print(infile)
  hdulist = fits.open(infile)
  table = hdulist[1].data
  rmag = table.field('flux_r')
  rmag = 30. - 2.5*np.log10(rmag)
  imag = table.field('flux_i')
  imag = 30. - 2.5*np.log10(imag)
  zmag = table.field('flux_z')
  zmag = 30. - 2.5*np.log10(zmag)
  wei = table.field('weight')
  r11 = table.field('R11')
  r22 = table.field('R22')
  zphot = table.field('zphotmof')
  hdulist.close()
  wei = 0.5*(r11+r22)
  cut = (zphot > 0.2) & (zphot < 1.3) & (r11 > -1000.) & (r22 > -1000.)
  rmag,imag,zmag,wei,r11,r22,zphot = rmag[cut],imag[cut],zmag[cut],wei[cut],r11[cut],r22[cut],zphot[cut]
  ngal = len(rmag)
  print(ngal,'DES sources read in')
  cut = np.random.choice(ngal,nsamp,replace=False)
  rmag,imag,zmag,wei,r11,r22,zphot = rmag[cut],imag[cut],zmag[cut],wei[cut],r11[cut],r22[cut],zphot[cut]
  ngal = len(rmag)
  print('Cut to',ngal,'sources')
  return rmag,imag,zmag,wei,r11,r22,zphot,ngal

########################################################################
# Read in HSC source magnitudes and weights.                           #
########################################################################

def readhscsurv():
  nsamp = 1000000
#  stem = '/Users/cblake/Data/desi/buzzard/'
  stem = 'lenscats/'
  infile = stem + 'hsc_mag.fits'
  print('\nReading in HSC source catalogue...')
  print(infile)
  hdulist = fits.open(infile)
  table = hdulist[1].data
  gmag = table.field('gcmodel_mag')
  rmag = table.field('rcmodel_mag')
  imag = table.field('icmodel_mag')
  zmag = table.field('zcmodel_mag')
  ymag = table.field('ycmodel_mag')
  wei = table.field('weight')
  mcorr = table.field('mcorr')
  erms = table.field('erms')
  zphot = table.field('photoz_best')
  hdulist.close()
  ngal = len(gmag)
  cut = (zphot > 0.3) & (zphot < 1.5)
  gmag,rmag,imag,zmag,ymag,wei,mcorr,erms,zphot = gmag[cut],rmag[cut],imag[cut],zmag[cut],ymag[cut],wei[cut],mcorr[cut],erms[cut],zphot[cut]
  ngal = len(gmag)
  print(ngal,'HSC sources read in')
  cut = np.random.choice(ngal,nsamp,replace=False)
  gmag,rmag,imag,zmag,ymag,wei,mcorr,erms,zphot = gmag[cut],rmag[cut],imag[cut],zmag[cut],ymag[cut],wei[cut],mcorr[cut],erms[cut],zphot[cut]
  ngal = len(gmag)
  print('Cut to',ngal,'sources')
  return gmag,rmag,imag,zmag,ymag,wei,mcorr,erms,zphot,ngal

########################################################################
# Read in KiDS calibration data.                                       #
########################################################################

def readkidscal():
#  stem = '/Users/cblake/Data/desi/buzzard/'
  stem = 'lenscats/'
  infile = stem + 'kids_cal.fits'
  print('\nReading in KiDS calibration sample...')
  print(infile)
  hdulist = fits.open(infile)
  table = hdulist[1].data
  zspec = table.field('z_spec')
  zb = table.field('z_B')
  wei = table.field('spec_weight_CV') # includes lensfit weight
  hdulist.close()
  ngal = len(zspec)
  print(ngal,'calibration sources read in')
  zb += np.random.uniform(-0.005,0.005,ngal)
  return wei,zspec,zb,ngal

########################################################################
# Read in DES metacal calibration data.                                #
########################################################################

def readdescal():
#  stem = '/Users/cblake/Data/desi/buzzard/'
  stem = 'lenscats/'
  infile = stem + 'des_metacal_cal.fits'
  print('\nReading in DES calibration sample...')
  print(infile)
  hdulist = fits.open(infile)
  table = hdulist[1].data
  zphot = table.field('zphot')
  zspec = table.field('zmc')
  wei = table.field('weinz')
  hdulist.close()
  ngal = len(zphot)
  print(ngal,'calibration sources read in')
  return wei,zspec,zphot,ngal

########################################################################
# Read in HSC calibration data.                                        #
########################################################################

def readhsccal():
#  stem = '/Users/cblake/Data/desi/buzzard/'
  stem = 'lenscats/'
  infile = stem + 'hsc_cal.fits'
  print('\nReading in HSC calibration sample...')
  print(infile)
  hdulist = fits.open(infile)
  table = hdulist[1].data
  zphot = table.field('redhsc')
  zspec = table.field('redcosmos')
  weisom = table.field('weisom')
  weilens = table.field('weilens')
  hdulist.close()
  ngal = len(zphot)
  print(ngal,'calibration sources read in')
  wei = weisom*weilens
  return wei,zspec,zphot,ngal

########################################################################
# Assign source properties by randomly drawing amongst the "neigh"     #
# nearest neighbours using a KDTree in magnitude space.                #
########################################################################

def findnearestsurv(mags,magssrc,p1src,p2src,p3src,neigh):
  print('\nAssigning properties from',neigh,'nearest neighbours...')
# Setting up KDTree on KiDS magnitudes
  tree = cKDTree(magssrc)
# Indices of nearest neighbours for each Buzzard magnitude
  ineighlst = tree.query(mags,k=neigh)[1]
# Random draw from neigh indices
  ineighdraw = np.random.randint(0,neigh,size=len(ineighlst))
  nmock = mags.shape[0]
# Assign properties
  p1,p2,p3 = np.empty(nmock),np.empty(nmock),np.empty(nmock)
  for i,j in enumerate(ineighdraw):
    k = ineighlst[i,j]
    p1[i],p2[i],p3[i] = p1src[k],p2src[k],p3src[k]
  return p1,p2,p3

########################################################################
# Apply shape noise to simulated catalogue.                            #
########################################################################

def appshapenoise(g1,g2,sige):
  print('\nApplying shape noise =',sige,'...')
  ngal = len(g1)
  n1 = sige*np.random.normal(size=ngal)
  n2 = sige*np.random.normal(size=ngal)
  a1 = g1 + n1
  a2 = g2 + n2
  a3 = 1. + g1*n1 + g2*n2
  a4 = g1*n2 - g2*n1
  e1 = (a1*a3 + a2*a4)/(a3*a3 + a4*a4)
  e2 = (a2*a3 - a1*a4)/(a3*a3 + a4*a4)
  return e1,e2

########################################################################
# Apply shape noise to simulated catalogue including shear bias.       #
########################################################################

def appshapecalkids(g1,g2,zphot,mcorrtom,sigetom,zplims):
  print('\nApplying KiDS shape noise and calibration bias...')
  ngal = len(g1)
  ibin = np.digitize(zphot,zplims) - 1
  mcorr = mcorrtom[ibin]
  sige = sigetom[ibin]
  g1nonoise = g1*(1.+mcorr)
  g2nonoise = g2*(1.+mcorr)
  n1 = sige*np.random.normal(size=ngal)
  n2 = sige*np.random.normal(size=ngal)
  a1 = g1nonoise + n1
  a2 = g2nonoise + n2
  a3 = 1. + g1nonoise*n1 + g2nonoise*n2
  a4 = g1nonoise*n2 - g2nonoise*n1
  e1 = (a1*a3 + a2*a4)/(a3*a3 + a4*a4)
  e2 = (a2*a3 - a1*a4)/(a3*a3 + a4*a4)
  return e1,e2,mcorr

def appshapecalhsc(g1,g2,wei,mcorr,erms):
  print('\nApplying HSC shape noise and calibration bias...')
  ngal = len(g1)
  r = 1. - erms**2
  sige = 1./np.sqrt(wei)
  g1nonoise = g1*2.*r*(1.+mcorr)
  g2nonoise = g2*2.*r*(1.+mcorr)
  n1 = sige*np.random.normal(size=ngal)
  n2 = sige*np.random.normal(size=ngal)
  a1 = g1nonoise + n1
  a2 = g2nonoise + n2
  a3 = 1. + g1nonoise*n1 + g2nonoise*n2
  a4 = g1nonoise*n2 - g2nonoise*n1
  e1 = (a1*a3 + a2*a4)/(a3*a3 + a4*a4)
  e2 = (a2*a3 - a1*a4)/(a3*a3 + a4*a4)
  return e1,e2

def appshapecaldes(g1,g2,zphot,r11,r22,sigetom,zplims):
  print('\nApplying DES shape noise and calibration bias...')
  ngal = len(g1)
  g1nonoise = g1*0.5*(r11+r22)
  g2nonoise = g2*0.5*(r11+r22)
  ibin = np.digitize(zphot,zplims) - 1
  sige = sigetom[ibin]
  n1 = sige*np.random.normal(size=ngal)
  n2 = sige*np.random.normal(size=ngal)
  a1 = g1nonoise + n1
  a2 = g2nonoise + n2
  a3 = 1. + g1nonoise*n1 + g2nonoise*n2
  a4 = g1nonoise*n2 - g2nonoise*n1
  e1 = (a1*a3 + a2*a4)/(a3*a3 + a4*a4)
  e2 = (a2*a3 - a1*a4)/(a3*a3 + a4*a4)
  return e1,e2

########################################################################
# Sub-sample source catalogue to match redshift distribution.          #
########################################################################

def dosubsamp(red,zmin,zmax,area):
  print('\nSubsampling source catalogue to match model N(z)...')
  sig0,z0 = 10.,0.2 # Parameters of model
  dz = 0.01
  nbin = int((zmax-zmin)/dz)
  nzdat,zlims = np.histogram(red,weights=np.ones_like(red),bins=nbin,range=[zmin,zmax],normed=False)
  zcen = zlims[:-1] + 0.5*dz
  nzdat *= 1./(area*3600.*dz)
  fact = 1. - (((zmax**2)/(2.*(z0**2)))+(zmax/z0)+1.)*np.exp(-zmax/z0)
  nzmod = ((sig0*(zcen**2))/(2.*(z0**3)*fact))*np.exp(-zcen/z0)
  sfrac = nzmod/nzdat
  ibin = np.digitize(red,zlims) - 1
  redprob = sfrac[ibin]
  keep = np.where(np.random.uniform(0.,1.,len(red))<redprob,True,False)
  print('Keeping',len(keep[keep]),'sources')
  return keep

########################################################################
# Apply photo-z scatter to spectroscopic redshifts.                    #
########################################################################

def dophotzscat(zspec,sigz,zpmin,zpmax):
  print('\nApplying photo-z scatter with sigz =',sigz,'...')
  ngal = len(zspec)
  dz = sigz*(1.+zspec)*np.random.normal(size=ngal)
  zphot = zspec + dz
#  zphot = np.where(zphot>zpmin,zphot,zpmin)
#  zphot = np.where(zphot<zpmax,zphot,zpmax)
# Repeatedly draw to ensure no redshifts outside range
  for irep in range(100):
    ind = np.invert((zphot > zpmin) & (zphot < zpmax))
    ncorr = len(zphot[ind])
    if (ncorr > 0):
      dz = sigz*(1.+zspec[ind])*np.random.normal(size=ncorr)
      zphot[ind] = zspec[ind] + dz
  return zphot

########################################################################
# Get 2D (z_phot,z_spec) distribution.                                 #
########################################################################

def getprobzszp(zspec,zphot,zsmin,zsmax,nzs,zpmin,zpmax,nzp):
  print('\nGet (z_phot,z_spec) probability distribution...')
  probzszp,edges = np.histogramdd(np.vstack([zspec,zphot]).transpose(),bins=(nzs,nzp),range=((zsmin,zsmax),(zpmin,zpmax)))
  return probzszp

########################################################################
# Draw photo-z values from 2D (z_phot,z_spec) distribution.            #
########################################################################

def dophotzdraw(zspec,probzszp,zsmin,zsmax,nzs,zpmin,zpmax,nzp):
  print('\nDrawing photo-z values from probability distribution...')
  dzp = (zpmax-zpmin)/float(nzp)
  zpcen = np.linspace(zpmin+0.5*dzp,zpmax-0.5*dzp,nzp)
  zpmod = zpcen
  zphotarr = np.arange(zpmin,zpmax,0.0001)
  zsbin = np.digitize(zspec,np.linspace(zsmin,zsmax,nzs+1)) - 1
  zphot = np.zeros(len(zspec))
  for izs in range(nzs):
    zsind = (zsbin == izs)
    ngal = len(zspec[zsind])
    pzmod = probzszp[izs,:]
    tck = splrep(zpmod,pzmod)
    Ns = np.cumsum(splev(zphotarr,tck))
    Ns /= Ns[-1]
    N = np.random.rand(ngal)
    zpind = np.searchsorted(Ns,N)
    zphot[zsind] = zphotarr[zpind]
  return zphot

########################################################################
# Write out lens file as ascii catalogue.                              #
########################################################################

def writemocklensascii(outfile,ras,dec,zspec,wei):
  print('\nWriting out lens mock catalogue...')
  print(outfile)
  f = open(outfile,'w')
  ngal = len(ras)
  f.write('# Lens catalogue\n')
  f.write('# R.A., Dec., z_spec, wei\n')
  f.write('{:d}'.format(ngal) + '\n')
  for i in range(ngal):
    f.write('{:10.6f} {:10.6f} {:7.5f} {:7.5f}'.format(ras[i],dec[i],zspec[i],wei[i]) + '\n')
  f.close()
  return

########################################################################
# Write out stage 0 source file as ascii catalogue.                    #
########################################################################

def writestage0mocksourceascii(outfile,ras,dec,zspec,zphot,g1,g2,e1,e2,wei):
  print('\nWriting out source mock catalogue...')
  print(outfile)
  f = open(outfile,'w')
  ngal = len(ras)
  f.write('# Source catalogue\n')
  f.write('# R.A., Dec., z_spec, z_phot, gamma_1, gamma_2, e_1, e_2, wei\n')
  f.write('{:d}'.format(ngal) + '\n')
  for i in range(ngal):
    f.write('{:10.6f} {:10.6f} {:7.5f} {:7.5f} {:11.8f} {:11.8f} {:11.8f} {:11.8f} {:7.5f}'.format(ras[i],dec[i],zspec[i],zphot[i],g1[i],g2[i],e1[i],e2[i],wei[i]) + '\n')
  f.close()
  return

########################################################################
# Write out stage 1 source file as ascii catalogue.                    #
########################################################################

def writestage1mocksourceascii(outfile,ras,dec,zspec,zphot,g1,g2,e1,e2,wei,x,y,isurv):
  if (isurv == 0):
    print('\nWriting out KiDS source mock catalogue...')
  elif (isurv == 1):
    print('\nWriting out DES source mock catalogue...')
  elif (isurv == 2):
    print('\nWriting out HSC source mock catalogue...')
  print(outfile)
  f = open(outfile,'w')
  ngal = len(ras)
  f.write('# Source catalogue\n')
  if (isurv == 0):
    f.write('# R.A., Dec., z_spec, z_phot, gamma_1, gamma_2, e_1, e_2, wei, m, dummy\n')
  elif (isurv == 1):
    f.write('# R.A., Dec., z_spec, z_phot, gamma_1, gamma_2, e_1, e_2, wei, R11, R22\n')
  elif (isurv == 2):
    f.write('# R.A., Dec., z_spec, z_phot, gamma_1, gamma_2, e_1, e_2, wei, m, e_rms\n')
  f.write('{:d}'.format(ngal) + '\n')
  for i in range(ngal):
    f.write('{:10.6f} {:10.6f} {:7.5f} {:7.5f} {:11.8f} {:11.8f} {:11.8f} {:11.8f} {:8.5f} {:8.5f} {:8.5f}'.format(ras[i],dec[i],zspec[i],zphot[i],g1[i],g2[i],e1[i],e2[i],wei[i],x[i],y[i]) + '\n')
  f.close()
  return

if __name__ == '__main__':
  main()

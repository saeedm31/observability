import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
from scipy import interpolate
import time


from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import matplotlib.pyplot as plt
from astropy.coordinates import get_sun
import datetime

from astropy import units as u
from astropy.coordinates import SkyCoord
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool
plt.close('all')

start = time.time()

def observibility_in_night(x,ra,dec): # paranal VLT. x: number of night 1 to 365
	alt_airmass,sun_cut = 41.8 , -18.0
	a = datetime.datetime(2017, 1, 1, 0, 0, 0, 703890)
	date = a + datetime.timedelta(days = x)
	date_c = str(date.year) + '-' + str(date.month) + '-' + str(date.day) + ' ' + '00:00:00'
	c = SkyCoord(ra, dec, frame='icrs')
	bear_mountain = EarthLocation(lat=-24.3*u.deg, lon=-70*u.deg, height=390*u.m) # VLT
	utcoffset = -4*u.hour  # Eastern Daylight Time
	midnight = Time(date) - utcoffset
	delta_midnight = np.linspace(-12, 12, 500)*u.hour
	target_altazs = c.transform_to(AltAz(obstime=midnight+delta_midnight, location=bear_mountain))  
	delta_midnight = np.linspace(-12, 12, 500)*u.hour
	times = midnight + delta_midnight
	altazframe = AltAz(obstime=times, location=bear_mountain)
	sunaltazs = get_sun(times).transform_to(altazframe)
	target_altazs = c.transform_to(altazframe)
	delta_midnight_star = -delta_midnight
	index = np.where(sunaltazs.alt.deg < sun_cut)  
	sun_time = delta_midnight[index]
	sun = sunaltazs.alt.deg[index]
	delta_midnight_star = delta_midnight_star[index]
	deg_star = target_altazs.alt[index]
	index_airmass = np.where(deg_star.deg > alt_airmass) # airmass 1.50 cut-off h=41.5
	sun_time = sun_time[index_airmass]
	sun = sun[index_airmass]
	delta_midnight_star = delta_midnight_star[index_airmass]
	deg_star = deg_star[index_airmass]
	try:
		dur = np.abs(delta_midnight_star.min() - delta_midnight_star.max())
		dur = dur.value
	except:
		dur = 0
	return dur

# alt_airmass = 41.8 # deg airmass = 2.0 = 30.0  1.5 = 41.8
# sun_cut = -18.0 # deg
stars_list = np.genfromtxt('stars-list.txt', delimiter='\t', skip_header=0, dtype=None, names=True,usemask=False)
inputs = range(365)



resul = []
i = 0
for item in stars_list[range(1)]:
	c = SkyCoord(item['RA'], item['DEC'], frame='icrs')
	dec = c.dec
	dec = dec.deg
	num_cores = multiprocessing.cpu_count()
	resul = Parallel(n_jobs=num_cores)(delayed(observibility_in_year)(x,item['RA'],item['DEC']) for x in inputs)
	# for day in inputs:
	# 	r = observibility_in_year(day,item['RA'],item['DEC'])
	# 	resul.append(r)
	j = 0
	for i in resul: # result: list with 365 items. Each item is hour(s) which the target is visibible at the airmass
		j = i + j
	print(item['name'], '\t',j)


DATE = []
for d in inputs:
	a = datetime.datetime(2017, 1, 1, 0, 0, 0, 703890)
	date = a + datetime.timedelta(days = d)
	DATE.append(date)
# plt.plot_date(DATE, resul, fmt="bo", tz=None, xdate=True)
plt.plot(DATE, resul)
plt.gcf().autofmt_xdate()
plt.show()


end = time.time()
# print(end - start)

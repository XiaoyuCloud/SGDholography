import matplotlib.pyplot as pyplot
import json

# psnrs_time = json.loads('./recon/psnrs_time.json')

with open('./recon/psnrs_time_single.json','r') as f:
        psnrs_time_single = json.load(f)
with open('./recon/ssims_time_single.json','r') as f:
        ssims_time_single = json.load(f)
with open('./recon/psnrs_time_time.json','r') as f:
        psnrs_time_time = json.load(f)
with open('./recon/ssims_time_time.json','r') as f:
        ssims_time_time = json.load(f)
with open('./recon/psnrs_time_dir.json', 'r') as f:
        psnrs_time_dir = json.load(f)
with open('./recon/ssims_time_dir.json', 'r') as f:
        ssims_time_dir = json.load(f)
pyplot.figure(1),pyplot.plot(psnrs_time_single['amp']),pyplot.plot(psnrs_time_time['amp']),pyplot.plot(psnrs_time_dir['amp']),pyplot.legend(['individual','time','dir']),pyplot.show()
pyplot.figure(2),pyplot.plot(ssims_time_single['amp']),pyplot.plot(ssims_time_time['amp']),pyplot.plot(ssims_time_dir['amp']),pyplot.legend(['individual','time','dir']),pyplot.show()
with open('./recon/psnrs_time_time_enc.json','r') as f:
        psnrs_time_time_enc = json.load(f)
with open('./recon/ssims_time_time_enc.json','r') as f:
        ssims_time_time_enc = json.load(f)
pyplot.figure(3),pyplot.plot(psnrs_time_time['amp']),pyplot.plot(psnrs_time_time_enc['amp']),pyplot.legend(['without encryption sampling','with encryption sampling']),pyplot.show()
pyplot.figure(4),pyplot.plot(ssims_time_time['amp']),pyplot.plot(ssims_time_time_enc['amp']),pyplot.legend(['without encryption sampling','with encryption sampling']),pyplot.show()
cc=1
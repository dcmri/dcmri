import dcmri as dc
import pydmr
#
# # fetch dmr file
file = dc.fetch('tristan_humans_healthy_rifampicin')
#
# # read dmr file
data = pydmr.read(file)


# Longitudinal magnetization in a 2-compartment system
# with kinetic transport and free relaxation

# v1*dm1/dt = fi1*mi1 - fo1*m1 + f12*m2 - f21*m1 + R1_1*v1*(m01-m1) 
# v2*dm2/dt = fi2*mi2 - fo2*m2 + f21*m1 - f12*m2 + R1_2*v2*(m02-m2) 

# v1*dm1/dt = fi1*mi1 - (fo1+f21+R1_1*v1)*m1 + f12*m2 + R1_1*v1*m01
# v2*dm2/dt = fi2*mi2 - (fo2+f12+R1_2*v2)*m2 + f21*m1 + R1_2*v2*m02

# f1 = fo1 + f21 + R1_1*v1 > 0
# f2 = fo2 + f12 + R1_2*v2 > 0

# J1(t) = fi1*mi1(t) + R1_1*v1*m01 > 0
# J2(t) = fi2*mi2(t) + R1_2*v2*m02 > 0

# v1*dm1/dt = J1 - f1*m1 + f12*m2 
# v2*dm2/dt = J2 - f2*m2 + f21*m1 

# K1 = (fo1 + f21)/v1 + R1_1
# K2 = (fo2 + f12)/v2 + R1_2
# K12 = f12/v2
# K21 = f21/v1

# dM1/dt = J1 - K1*M1 + K12*M2 
# dM2/dt = J2 - K2*M2 + K21*M1 

# K = [[K1, -K12],[-K21, K2]
# dM/dt = J - KM

# Use generic solution for n-comp system to solve for M(t)
# Note with R1(t) this is not a stationary system

# Solution with K and J constant in time:

# M(t) = exp(-tK)M(0) + exp(-tK)*J
# M(t) = exp(-tK)M(0) + (1-exp(-tK)) K^-1 J

# Check

# dM/dt 
# = - K exp(-tK)M(0) + exp(-tK) J
# J - KM 
# = J - Kexp(-tK)M(0) - (1-exp(-tK))J 
# = - K exp(-tK)M(0) + exp(-tK) J

# Spoiled gradient echo steady state:
# M = exp(-TR*K) cosFA M + (1-exp(-TR*K)) K^-1 J
# (1 - cosFA exp(-TR*K)) M = (1-exp(-TR*K)) K^-1 J
# M = (1 - cosFA exp(-TR*K))^-1 (1-exp(-TR*K)) K^-1 J
# M = (1 - cosFA exp(-TR*K))^-1 K^-1 (1-exp(-TR*K))  J
# M = [K(1 - cosFA exp(-TR*K))]^-1 (1-exp(-TR*K)) J
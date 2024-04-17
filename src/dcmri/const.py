import math

def ca_conc(agent):
    """Contrast agent concentration in the bottle"""
    if (agent=='Primovist') or (agent=='Gadoxetate'):
        # https://www.bayer.com/sites/default/files/2020-11/primovist-pm-en.pdf
        return 0.25     # mmol/mL
    if (agent=='Dotarem') or (agent=='Gadoterate'):
        # https://www.accessdata.fda.gov/drugsatfda_docs/label/2018/204781s008lbl.pdf
        return 0.5      # mmol/mL
    if (agent=='Gadovist') or (agent=='Gadobutrol'):
        # https://www.medicines.org.uk/emc/product/2876/smpc#gref
        return 1.0      # mmol/mL
    raise ValueError('No concentration data for contrast agent ' + agent)


def ca_std_dose(agent:str):
    """Standard dose in mL/kg""" # better in mmol/kg, or offer it as an option
    if (agent=='Dotarem') or (agent=='Gadoterate'):
        # https://www.accessdata.fda.gov/drugsatfda_docs/label/2018/204781s008lbl.pdf
        return 0.2      # mL/kg
    if (agent=='Gadovist') or (agent=='Gadobutrol'):
        return 0.1      # mL/kg
    raise ValueError('No dosage data for contrast agent ' + agent)


def relaxivity(field_strength=3.0, tissue='plasma', agent='Primovist', type='T1'): 
    """Relaxivity values in units of Hz/M"""
# https://journals.lww.com/investigativeradiology/FullText/2005/11000/Comparison_of_Magnetic_Properties_of_MRI_Contrast.5.aspx
# Gadoxetate
# Szomolanyi P, Rohrer M, Frenzel T, Noebauer-Huhmann IM, Jost G, Endrikat J, Trattnig S, Pietsch H. Comparison of the Relaxivities of Macrocyclic Gadolinium-Based Contrast Agents in Human Plasma at 1.5, 3, and 7 T, and Blood at 3 T. Invest Radiol. 2019 Sep;54(9):559-564. doi: 10.1097/RLI.0000000000000577.
    field = math.floor(field_strength)
    if type=='T1':
        if tissue == 'plasma': # 37 degrees
            if (agent=='Primovist') or (agent=='Gadoxetate'):
                if field == 1.5: return 8.1*1000
                if field == 3.0: return 6.4*1000
                if field == 4.0: return 6.4*1000
                if field == 7.0: return 6.2*1000
                if field == 9.0: return 6.1*1000
            if agent in ['Dotarem','Gadoterate']:
                if field == 1.5: return 3.6*1000
                if field == 3.0: return 3.5*1000
            if agent in ['Gadovist','Gadobutrol']:
                if field == 1.5: return 5.2*1000
                if field == 3.0: return 5.0*1000
            if agent in ['Magnevist']:
                if field == 1.5: return 4.1*1000
                if field == 3.0: return 3.7*1000
            if agent in ['Omniscan']:
                if field == 1.5: return 4.3*1000
                if field == 3.0: return 4.0*1000
        if tissue == 'hepatocytes':
            if field == 1.5: return 14.6*1000    
            if field == 3.0: return 9.8*1000    
            if field == 4.0: return 7.6*1000   
            if field == 7.0: return 6.0*1000  
            if field == 9.0: return 6.1*1000 

    msg = 'No relaxivity data for ' + agent + ' at ' + str(field_strength) + ' T.'
    raise ValueError(msg)


def T1(field_strength=3.0, tissue='blood', Hct=0.45):
    """T1 in seconds"""
    field = math.floor(field_strength)
    if tissue=='blood':
        if field == 1.5: return 1480.0/1000.0    
        if field == 3.0: return 1/(0.52 * Hct + 0.38)  # Lu MRM 2004 
    if tissue=='liver':
        if field == 1.5: return 602.0/1000.0    # liver R1 in 1/sec (Waterton 2021)
        if field == 3.0: return 752.0/1000.0     # liver R1 in 1/sec (Waterton 2021)
        if field == 4.0: return 1/1.281     # liver R1 in 1/sec (Changed from 1.285 on 06/08/2020)
        if field == 7.0: return 1/1.109     # liver R1 in 1/sec (Changed from 0.8350 on 06/08/2020)
        if field == 9.0: return 1/0.920     # per sec - liver R1 (https://doi.org/10.1007/s10334-021-00928-x)
    if tissue=='kidney':
        # Reference values average over cortext and medulla from Cox et al
        # https://academic.oup.com/ndt/article/33/suppl_2/ii41/5078406
        if field == 1.5: return ((1024+1272)/2) / 1000
        if field == 3.0: return ((1399+1685)/2) / 1000
    msg = 'No T1 values for ' + tissue + ' at ' + str(field_strength) + ' T.'
    raise ValueError(msg)
.. _imaging-sequences:

Signals
-------

.. _sequence-params:
.. list-table:: **List of sequence parameters**
    :widths: 15 20 40 10 10
    :header-rows: 1

    * - Short name
      - Full name
      - Definition
      - Units
      - Sequences
    * - TS
      - Sampling time
      - Duration of the signal readout for a single time point. If TS=0, the 
        signals are sampled by interpolation. If TS is a finite value, the 
        signals are averaged over a time TS around the acquisition time. 
        Defaults to None.
      - sec
      - SS, SR
    * - S0
      - Signal scale factor
      - Amplitude of the signal model
      - a.u.
      - SS, SR
    * - FA
      - Flip angle
      - Angle of magnetization after excitation
      - degrees
      - SS, SR
    * - TR
      - Repetition time
      - Time between excitation pulses
      - sec
      - SS, SR
    * - TC
      - Time to center
      - Time between the preparation pulse and the k-space center
      - sec
      - SR
    * - TP
      - Preparation delay
      - Time between the preparation pulse and the first k-space line
      - sec
      - SR
    * - B1corr
      - B1 correction factor
      - Factor to correct flip angles for B1-variations
      - dimensionless
      - SR, SS


.. _params-per-sequence:
.. list-table:: **Sequence parameters**
    :widths: 20 40
    :header-rows: 1 

    * - sequence
      - parameters
    * - SS
      - S0, FA, TR
    * - SR
      - S0, FA, TR, TC, TP









#!/usr/bin/env python3
import sys
sys.path.append('/home/onurs/edge_ai_web/ppg_web')
from pan_tompkins import PanTompkins
import numpy as np
import csv

# Load latest CSV
csv_path = '/home/onurs/edge_ai_web/ppg_web/outputs/session.csv'
ir = []
with open(csv_path, 'r') as f:
    r = csv.reader(f)
    next(r, None)
    for row in r:
        if len(row) >= 3:
            ir.append(float(row[2]))
ir = np.array(ir)

processor = PanTompkins(fs=100.0)
result = processor.process(ir)

print('Total samples:', len(ir))
print('BPM:', result['bpm'])
print('QRS peaks count:', len(result['qrs_peaks']))
print('RR intervals (ms):', result['rr_intervals_ms'][:10], '...' if len(result['rr_intervals_ms']) > 10 else '')
print('First 10 QRS peak indices:', result['qrs_peaks'][:10])
print('Integrated signal stats: min', min(result['integrated']), 'max', max(result['integrated']))

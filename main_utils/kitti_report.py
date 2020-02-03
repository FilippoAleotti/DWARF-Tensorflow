import os
import argparse
from terminaltables import AsciiTable

def _format(number):
    return '{:.4f}'.format(float(number))

parser = argparse.ArgumentParser(description='Display kitti results')
parser.add_argument('--results', type=str, required=True, help='path to a kitti result folder')
parser.add_argument('--noc',  action='store_true')
args = parser.parse_args()

results = ['stats_flow_occ.txt', 'stats_disp_occ_0.txt', 'stats_disp_occ_1.txt', 'stats_scene_flow_occ.txt'] 
metrics = ['background', 'foreground', 'all', 'density']

table_data = [['FILE','BACKGROUND', 'FOREGROUND', 'ALL', 'DENSITY']]
if args.noc:
    results = [x.replace('occ','noc') for x in results]
for r in results:
    with open(os.path.join(args.results, r),'r') as result_file:
        lines = result_file.readlines()
    background, _, foreground, _, all, _, density = lines[0].strip().split(' ')
    values = [r, _format(background), _format(foreground), _format(all), _format(density)]
    table_data.append(values)
table = AsciiTable(table_data)

print('\nEvaluation results of {}:'.format(args.results))
print(table.table)

with open(os.path.join(args.results,'report.txt'),'w') as f:
    for data in table_data:
        for i,value in enumerate(data):
            if i > 0:
                value = value.replace('.',',')
            f.write('{};'.format(value))
        f.write('\n')

print('report.txt has been written in {}'.format(args.results, 'KITTI'))


    

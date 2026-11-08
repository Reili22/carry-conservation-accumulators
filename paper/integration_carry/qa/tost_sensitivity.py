import json
from pathlib import Path

root = Path(__file__).resolve().parents[3]
source = root / 'python/accrate/cnn_cal_10k_tost.json'
data = json.loads(source.read_text())
out = {name: {'ci90': item['ci90'], 'equivalent_half_point':
        item['ci90'][0] > -0.005 and item['ci90'][1] < 0.005}
       for name, item in data.items()}
target = source.with_name('cnn_cal_10k_tost_half_point.json')
target.write_text(json.dumps(out, indent=2) + '\n')
print(target)
for name, item in out.items():
    print(name, item['equivalent_half_point'])

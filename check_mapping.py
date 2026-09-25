import json
p1 = r'C:\Users\Emabe\Documents\GitHub\xlstm\rust\python\model_test_mapping.json'
p2 = r'C:\Users\Emabe\Documents\GitHub\xlstm\rust\model_test_mapping.json'
for p in [p1, p2]:
    d = json.load(open(p))
    ks = list(d['parameters'].keys())
    print(f'{p}:')
    print(f'  keys={len(d["parameters"])}')
    print(f'  first={ks[0]} shape={d["parameters"][ks[0]]["shape"]}')

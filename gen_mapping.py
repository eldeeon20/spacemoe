from safetensors import safe_open
import json

with safe_open(r'C:\Users\Emabe\Documents\GitHub\xlstm\rust\model_test.safetensors', framework='pt') as f:
    params = {}
    for k in f.keys():
        t = f.get_tensor(k)
        params[k] = {
            'shape': list(t.shape),
            'dtype': 'f32',
            'burn_name': k
        }

mapping = {'parameters': params}
outpath = r'C:\Users\Emabe\Documents\GitHub\xlstm\rust\model_test_mapping.json'
with open(outpath, 'w') as out:
    json.dump(mapping, out, indent=2)
print(f'Generated {len(params)} keys -> {outpath}')

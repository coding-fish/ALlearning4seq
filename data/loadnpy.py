import numpy as np
import json
import pdb

def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

data = np.load("samples.npy", allow_pickle=True)
print(data.shape)  # 打印数据形状
print(data.dtype)

## 将数据转换为字符串格式
with open('output.json', 'w') as f:
    for d in data:
        line = json.dumps(d, ensure_ascii=False, default=default_dump)
        print(line, file=f)

# data_str = np.array2string(data, separator=',')
# data_list = data.tolist()
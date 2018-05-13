import pickle
import numpy as np

with open("/run/media/jacob/SSD/Development/thesis/FACTsourceFinding/test/sep_all.p", "rb") as talapas:
    talapas_aucs = pickle.load(talapas)

with open("/run/media/jacob/SSD/Development/thesis/FACTsourceFinding/sep_talapas.p", "rb") as f:
    dortmund_auc = pickle.load(f)


tal1 = talapas_aucs[0][0::2]
tal2 = talapas_aucs[0][1::2]

print(talapas_aucs)
print(dortmund_auc)
dort1 = dortmund_auc[0]
dort2 = [e for e in dortmund_auc[0] if isinstance(e, float)]
print(tal1)
print(tal2)
print(dort2)
#print(np.max(tal1))
print(np.max(tal2))
#print(np.max(dort1))
print(np.max(dort2))


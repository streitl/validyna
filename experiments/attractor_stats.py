from collections import Counter

import dysts.flows
from dysts.base import get_attractor_list

counter = Counter()
for attractor_idx, attractor_name in enumerate(get_attractor_list()):
    attractor = getattr(dysts.flows, attractor_name)()
    counter.update({len(attractor.ic): 1})

print(counter)  # shows Counter({3: 100, 4: 19, 10: 7, 6: 3, 5: 2})

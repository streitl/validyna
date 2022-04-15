from collections import Counter

import dysts.flows
from dysts.base import get_attractor_list, DynSys

if __name__ == '__main__':
    counter = Counter()
    for attractor_name in get_attractor_list():
        attractor: DynSys = getattr(dysts.flows, attractor_name)()
        counter.update({len(attractor.ic): 1})

    print(counter)  # shows Counter({3: 100, 4: 19, 10: 7, 6: 3, 5: 2})

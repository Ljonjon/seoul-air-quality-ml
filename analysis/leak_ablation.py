"""
Factorial ablation: what is each revision worth, in accuracy points?

Three binary choices define the cells:

    leak    : PM2.5_roll6 = rolling mean over t-5..t  (course version)
              vs. t-6..t-1                            (this repo)
    filter  : instrument-status filter applied, or not
    protocol: random-stratified (course) vs chronological (this repo)

Cell E reproduces the graded submission end to end, so the ablation is
calibrated against the reported numbers before any factor is varied.

    python analysis/leak_ablation.py --data /path/to/AirPollutionSeoul
"""

import argparse

from _common import (RESULTS_DIR, add_argument, build_features, leaky_roll6,
                     load_without_status_filter, resolve_data_path, SILENT,
                     evaluate_variant, load_and_clean, write_json)

PARSED = {}


def cleaned(path):
    PARSED.setdefault('cleaned', load_and_clean(path, log=SILENT))
    return PARSED['cleaned']


def raw(path):
    PARSED.setdefault('raw', load_without_status_filter(path))
    return PARSED['raw']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_argument(ap)
    args = ap.parse_args()
    path = resolve_data_path(args.data)

    cells = [
        # tag                              leak  filter  protocol
        ('E_course_reproduction',          True,  False, 'random'),
        ('A_leaky_but_filtered',           True,  True,  'random'),
        ('C_fixed_but_unfiltered',         False, False, 'random'),
        ('B_fixed_filtered_random',        False, True,  'random'),
        ('D_fixed_filtered_chronological', False, True,  'chronological'),
        ('L_leaky_filtered_chronological', True,  True,  'chronological'),
    ]

    cache = {}
    rows = []
    for tag, leaky, filtered, protocol in cells:
        key = (leaky, filtered)
        if key not in cache:
            base = cleaned(path) if filtered else raw(path)
            cache[key] = (leaky_roll6(base.copy()) if leaky
                          else build_features(base.copy(), log=SILENT))
            print(f'[features] leaky={leaky} filtered={filtered} '
                  f'-> {len(cache[key]):,} rows', flush=True)
        rows += evaluate_variant(tag, cache[key], protocol)

    write_json(args.out or RESULTS_DIR / 'leak_ablation.json',
               {'cells': [{'tag': t, 'leak': l, 'status_filter': fl, 'protocol': p}
                          for t, l, fl, p in cells],
                'runs': rows})


if __name__ == '__main__':
    main()

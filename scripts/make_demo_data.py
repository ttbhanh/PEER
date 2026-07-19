#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import random
from pathlib import Path

from peer.utils import ensure_dir, write_jsonl

TOPICS = {
    'baby': ['bottle', 'stroller', 'diaper bag'],
    'cellphone': ['phone case', 'charger', 'screen protector'],
    'musical': ['guitar strings', 'microphone', 'keyboard stand'],
}
ASPECT_SENTENCES = {
    'battery': ['The battery lasts all day.', 'Battery life is disappointing.', 'The charger works fast.'],
    'sound': ['The sound quality is clear.', 'Bass is weak but vocals are clean.', 'It sounds much better than expected.'],
    'screen': ['The screen is bright and easy to read.', 'Display quality is sharp.', 'The screen scratches too easily.'],
    'price': ['The price is affordable for the quality.', 'It is expensive for what it offers.', 'Great value for the money.'],
    'comfort': ['It feels comfortable during long use.', 'The material is soft and comfortable.', 'Comfort could be better.'],
    'durability': ['Build quality feels durable.', 'It broke after a few days.', 'The material seems strong.'],
    'shipping': ['Shipping was fast and packaging was clean.', 'The package arrived late.', 'Delivery was quick.'],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='data/raw')
    ap.add_argument('--datasets', nargs='+', default=['baby', 'cellphone', 'musical'])
    ap.add_argument('--users', type=int, default=30)
    ap.add_argument('--items', type=int, default=12)
    ap.add_argument('--reviews-per-user', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    out = ensure_dir(args.output)
    ts = 1700000000000
    for ds in args.datasets:
        reviews = []
        metadata = []
        items = [f'{ds}_item_{i}' for i in range(args.items)]
        item_aspects = {}
        for i, iid in enumerate(items):
            aspects = random.sample(list(ASPECT_SENTENCES.keys()), 3)
            item_aspects[iid] = aspects
            metadata.append({
                'parent_asin': iid,
                'title': f'{TOPICS.get(ds, ["product"])[i % len(TOPICS.get(ds, ["product"]))]} {i}',
                'main_category': ds,
                'features': [f'Feature about {a}' for a in aspects],
                'description': [f'This item focuses on {", ".join(aspects)}.'],
                'brand': f'Brand{i%4}',
            })
        for u in range(args.users):
            user_aspects = random.sample(list(ASPECT_SENTENCES.keys()), 3)
            user_items = random.sample(items, min(args.reviews_per_user, len(items)))
            for rpos, iid in enumerate(user_items):
                aspects = list(dict.fromkeys(random.sample(user_aspects, 2) + random.sample(item_aspects[iid], 2)))
                sentences = []
                for a in aspects[:3]:
                    sentences.append(random.choice(ASPECT_SENTENCES[a]))
                text = ' '.join(sentences)
                rating = 5 if any('Great' in s or 'clear' in s or 'durable' in s for s in sentences) else random.choice([3,4])
                reviews.append({
                    'user_id': f'user_{u}',
                    'parent_asin': iid,
                    'rating': rating,
                    'text': text,
                    'timestamp': ts,
                    'helpful_vote': random.randint(0, 20),
                })
                ts += random.randint(1000, 5000)
        write_jsonl(reviews, out / f'{ds}_reviews.jsonl')
        write_jsonl(metadata, out / f'{ds}_metadata.jsonl')
        print(f'Wrote demo {ds}: {len(reviews)} reviews, {len(metadata)} metadata rows')


if __name__ == '__main__':
    main()

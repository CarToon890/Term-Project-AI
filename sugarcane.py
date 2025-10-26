#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sugarcane Sweetness Prediction (no sklearn version)
---------------------------------------------------
Model: Naive Bayes (manual implementation)
Input files:
 - train.csv : contains all attributes + Sweet (target)
 - test.csv  : contains all attributes without Sweet
Output:
 - <student_id>.txt : each line is the predicted sweetness (High/Medium/Low)
"""

import csv, math, statistics
from collections import defaultdict, Counter

STUDENT_ID = "67160378"  #เปลี่ยนเป็นรหัสนิสิตของคุณ
TARGET = "Sweet"
CAT_TOKEN = "UNK"
EPS = 1e-9


# --- Helper functions ---
def try_float(x):
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(x)
    except:
        return None


def read_csv(file):
    with open(file, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def detect_types(rows):
    """ตรวจว่าคอลัมน์ไหนเป็นตัวเลข (num) หรือกลุ่ม (cat)"""
    sample = rows[0]
    types = {}
    for col in sample:
        if col == TARGET: continue
        vals = [try_float(r[col]) for r in rows]
        ratio = sum(v is not None for v in vals) / len(vals)
        types[col] = "num" if ratio > 0.7 else "cat"
    return types


# --- Naive Bayes model ---
class NaiveBayes:
    def __init__(self, types):
        self.types = types
        self.classes = []
        self.class_counts = Counter()
        self.priors = {}
        self.num_stats = defaultdict(dict)  # {cls: {feat: (mean,var)}}
        self.cat_counts = defaultdict(lambda: defaultdict(Counter))
        self.cat_vocab = defaultdict(set)
        self.global_median = {}

    def fit(self, rows):
        # median สำหรับ missing numeric
        for feat, t in self.types.items():
            if t == "num":
                vals = [try_float(r[feat]) for r in rows if try_float(r[feat]) is not None]
                self.global_median[feat] = statistics.median(vals) if vals else 0.0

        # แยกข้อมูลตามคลาส
        by_cls = defaultdict(list)
        for r in rows:
            y = str(r[TARGET]).strip()
            if not y: continue
            self.class_counts[y] += 1
            by_cls[y].append(r)
            for feat, t in self.types.items():
                if t == "cat":
                    v = r[feat].strip() if r[feat] else CAT_TOKEN
                    self.cat_vocab[feat].add(v)

        total = sum(self.class_counts.values())
        self.classes = sorted(self.class_counts.keys())
        self.priors = {c: self.class_counts[c] / total for c in self.classes}

        # คำนวณ mean/var
        for c in self.classes:
            for feat, t in self.types.items():
                if t == "num":
                    vals = [try_float(r[feat]) for r in by_cls[c]]
                    vals = [self.global_median[feat] if v is None else v for v in vals]
                    if not vals:
                        mu, var = 0.0, 1.0
                    else:
                        mu = sum(vals)/len(vals)
                        var = sum((x-mu)**2 for x in vals)/(len(vals)-1 or 1)
                        var = var if var > 0 else EPS
                    self.num_stats[c][feat] = (mu, var)
                else:
                    for r in by_cls[c]:
                        v = r[feat].strip() if r[feat] else CAT_TOKEN
                        self.cat_counts[c][feat][v] += 1

    def _log_gauss(self, x, mean, var):
        return -0.5*math.log(2*math.pi*var) - (x-mean)**2/(2*var)

    def predict_row(self, r):
        scores = {}
        for c in self.classes:
            logp = math.log(self.priors[c] + EPS)
            for feat, t in self.types.items():
                if t == "num":
                    x = try_float(r[feat]) or self.global_median[feat]
                    mean, var = self.num_stats[c][feat]
                    logp += self._log_gauss(x, mean, var)
                else:
                    v = r[feat].strip() if r[feat] else CAT_TOKEN
                    cnt = self.cat_counts[c][feat][v]
                    V = len(self.cat_vocab[feat])
                    prob = (cnt + 1) / (self.class_counts[c] + V)
                    logp += math.log(prob)
            scores[c] = logp
        return max(scores, key=scores.get)

    def predict(self, rows):
        return [self.predict_row(r) for r in rows]


# --- Main ---
def main():
    train = read_csv("C:\\Users\\USER\\OneDrive\\Desktop\\BUU68\\AI 68\\Term Project\\train.csv")
    test = read_csv("C:\\Users\\USER\\OneDrive\\Desktop\\BUU68\\AI 68\\Term Project\\test.csv")

    types = detect_types(train)
    model = NaiveBayes(types)
    model.fit(train)
    preds = model.predict(test)

    out = f"{STUDENT_ID}.txt"
    with open(out, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(p + "\n")
    print(f"เขียนผลลัพธ์แล้วที่ {out} ({len(preds)} แถว)")


if __name__ == "__main__":
    main()

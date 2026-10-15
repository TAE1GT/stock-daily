#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
매일 아침 종목 발굴 대시보드 (코스피 · 코스닥)

설치 (최초 1회)
    pip install pandas numpy

실행
    python stock_dashboard.py --serve   # 추천: http://localhost:8000 에서 보기, 새로고침하면 최신 데이터 반영
    python stock_dashboard.py           # dashboard.html 파일만 만들고 브라우저로 열기
    python stock_dashboard.py --demo    # 가상 데이터로 화면 미리보기 (인터넷 불필요)

동작
    - 네이버 금융에서 종목 목록과 최근 160 거래일 시세를 받음 (약 1~2분)
    - 장 마감 데이터는 16:00 이후 확정본으로 간주. --serve 모드에서는 16:00 이후 첫 새로고침 때 자동 재계산
    - 3가지 전략(추세 눌림목 / 거래량 돌파 / 과매도 반등) 점수 중 최고점으로 순위 산정
"""
import argparse
import json
import math
import os
import re
import sys
import threading
import webbrowser
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

CFG = dict(
    days=160,              # 수집 거래일 수
    min_amount=30e8,       # 최소 거래대금 30억
    min_mcap=1000e8,       # 최소 시가총액 1,000억
    min_price=2000,        # 최소 주가
    min_history=120,       # 상장 후 최소 거래일 (약 6개월)
    chart_days=100,        # 차트 표시 기간
    top_n=16,              # 1~3 / 4~6 / 7~16
)

STOP_LABEL = {1: "20일선 -3%", 2: "돌파 기준가(60일 고가)", 3: "최근 5일 저점"}


# ───────────────────────────── 데이터 수집 ─────────────────────────────
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://finance.naver.com/",
}
ETF_BRANDS = {"KODEX", "TIGER", "KBSTAR", "RISE", "ACE", "SOL", "HANARO", "KOSEF", "ARIRANG", "PLUS", "KIWOOM",
              "TIMEFOLIO", "WOORI", "BNK", "파워", "마이다스", "히어로즈", "1Q", "ITF", "에셋플러스", "트러스톤",
              "UNICORN", "FOCUS", "KTOP", "VITA", "TRUSTON", "마이티", "DAISHIN343", "TREX", "HK", "더제이"}


def http_get(url, retries=3):
    import time
    import urllib.request
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.read().decode("cp949", errors="replace")
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(1.5 * (k + 1))


def _strip(s):
    return re.sub(r"<[^>]+>", "", s).replace("&nbsp;", " ").strip()


def _num(s):
    try:
        return float(_strip(s).replace(",", "").replace("%", ""))
    except ValueError:
        return np.nan


def is_excluded_name(name):
    first = name.split()[0] if name.split() else ""
    return first in ETF_BRANDS or "ETN" in name or "스팩" in name


def naver_listing():
    """네이버 금융 시가총액 페이지에서 코스피·코스닥 종목 목록"""
    rows = []
    for sosok, mk in ((0, "KOSPI"), (1, "KOSDAQ")):
        seen = set()
        for page in range(1, 90):
            html = http_get(f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}")
            heads = [_strip(h) for h in re.findall(r'<th[^>]*scope="col"[^>]*>(.*?)</th>', html, re.S)]
            new = 0
            for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
                mm = re.search(r'<a[^>]*code=(\d{6})[^>]*>(.*?)</a>', tr, re.S)
                if not mm or not _strip(mm.group(2)):
                    continue
                code, name = mm.group(1), _strip(mm.group(2))
                if code in seen:
                    continue
                seen.add(code)
                new += 1
                cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
                rec = dict(code=code, name=name, market=mk)
                for key in ("현재가", "시가총액", "거래량"):
                    i = heads.index(key) if key in heads else -1
                    rec[key] = _num(cells[i]) if 0 <= i < len(cells) else np.nan
                rows.append(rec)
            if new == 0:
                break
        print(f"  {mk} 종목 목록 {len(seen)}개", flush=True)
    if not rows:
        raise RuntimeError("네이버 종목 목록을 읽지 못했습니다 (페이지 형식 변경 가능성)")
    return pd.DataFrame(rows).set_index("code")


def naver_ohlcv(code, count):
    xml = http_get(f"https://fchart.stock.naver.com/sise.nhn?symbol={code}&timeframe=day&count={count}&requestType=0")
    recs = re.findall(r'data="(\d{8})\|([\d.]+)\|([\d.]+)\|([\d.]+)\|([\d.]+)\|([\d.]+)"', xml)
    if not recs:
        return None
    df = pd.DataFrame(recs, columns=["d", "o", "h", "l", "c", "v"])
    df.index = pd.to_datetime(df.pop("d"), format="%Y%m%d")
    return df.astype(float)


def load_naver(days):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print("종목 목록 받는 중…", flush=True)
    lst = naver_listing()
    lst = lst[[c.endswith("0") and not is_excluded_name(n) for c, n in zip(lst.index, lst["name"])]]
    # 목록의 전일 기준 값으로 1차 거르기 (값을 못 읽으면 통과)
    amt = lst["현재가"] * lst["거래량"]
    keep = (amt.isna() | (amt >= CFG["min_amount"] * 0.4)) & (lst["현재가"].isna() | (lst["현재가"] >= CFG["min_price"] * 0.8))
    keep &= lst["시가총액"].isna() | (lst["시가총액"] * 1e8 >= CFG["min_mcap"] * 0.8)
    lst = lst[keep]
    print(f"시세 받는 중… ({len(lst)}종목)", flush=True)

    data = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(naver_ohlcv, code, days + 5): code for code in lst.index}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                df = f.result()
                if df is not None and len(df):
                    data[futs[f]] = df
            except Exception:
                pass
            if i % 200 == 0:
                print(f"  {i}/{len(futs)}", flush=True)
    if not data:
        raise RuntimeError("네이버 시세를 하나도 받지 못했습니다")

    P = {k: pd.DataFrame({code: df[k] for code, df in data.items()}).sort_index() for k in "ohlcv"}
    now = datetime.now()
    today = pd.Timestamp(now.date())
    if P["c"].index[-1] == today and now.hour < 16:      # 장중 미확정 데이터 제외
        P = {k: df.iloc[:-1] for k, df in P.items()}
    P = {k: df.iloc[-days:] for k, df in P.items()}
    halted = P["o"] <= 0
    for key in ("o", "h", "l"):
        P[key] = P[key].mask(halted, P["c"])
    P["a"] = P["c"] * P["v"]                                # 거래대금(종가×거래량 근사)
    mcap = lst["시가총액"] * 1e8
    market = lst["market"].to_dict()
    names = lst["name"].to_dict()
    return P, mcap, market, names.get


def load_demo(days, n=180):
    """화면 확인용 가상 데이터"""
    rng = np.random.default_rng(int(datetime.now().strftime("%Y%m%d")))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=days)
    A = ["한빛", "대성", "미래", "동방", "세원", "태광", "청람", "가온", "새롬", "누리", "금강", "백두", "서진", "해솔"]
    B = ["전자", "반도체", "바이오", "화학", "정밀", "테크", "에너지", "소재", "제약", "중공업", "시스템", "로보틱스", "모빌리티"]
    tickers, names, mk = [], {}, {}
    data = {k: np.zeros((days, n)) for k in "ohlcva"}
    for j in range(n):
        t = f"{(j + 1) * 10 + 100000:06d}"
        tickers.append(t)
        names[t] = A[j % len(A)] + B[(j // len(A)) % len(B)]
        mk[t] = "KOSPI" if rng.random() < 0.4 else "KOSDAQ"
        p0, drift, vol = rng.uniform(3000, 150000), rng.normal(0.0003, 0.0015), rng.uniform(0.012, 0.032)
        r = rng.normal(drift, vol, days)
        pat = rng.choice(4, p=[0.82, 0.06, 0.06, 0.06])
        if pat == 1:      # 수축 후 돌파
            r[-41:-1] = rng.normal(0.0002, vol * 0.35, 40)
            r[-1] = rng.uniform(0.01, 0.09)
        elif pat == 2:    # 급락 후 반등
            r[-12:-1] = rng.normal(-0.013, vol * 0.5, 11)
            r[-1] = rng.uniform(0.005, 0.025)
        elif pat == 3:    # 상승 추세 중 눌림
            r[:-8] += 0.0018
            r[-8:] = rng.normal(-0.004, vol * 0.4, 8)
        c = p0 * np.exp(np.cumsum(r))
        o = c / np.exp(r) * (1 + rng.normal(0, vol * 0.3, days))
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol * 0.45, days)))
        l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol * 0.45, days)))
        v = rng.uniform(1e5, 3e6) * np.exp(rng.normal(0, 0.35, days)) * (1 + 15 * np.abs(r))
        if pat == 1:
            v[-41:-1] *= 0.6
            v[-1] *= rng.uniform(1.2, 5)
        elif pat == 2:
            v[-1] *= rng.uniform(1.8, 4)
            o[-1] = c[-1] * 0.985
            l[-1] = min(l[-1], o[-1] * 0.96)
        elif pat == 3:
            v[-8:] *= 0.6
        for key, arr in zip("ohlcv", (o, h, l, c, v)):
            data[key][:, j] = np.round(arr) if key != "v" else np.round(arr)
        data["a"][:, j] = data["c"][:, j] * data["v"][:, j]
    P = {k: pd.DataFrame(data[k], index=dates, columns=tickers) for k in "ohlcva"}
    mcap = pd.Series(np.exp(rng.uniform(np.log(800e8), np.log(8e12), n)), index=tickers)
    return P, mcap, mk, names.get


# ───────────────────────────── 지표 계산 ─────────────────────────────
def indicators(P):
    c, h, v = P["c"], P["h"], P["v"]
    I = {}
    I["ma20"] = c.rolling(20).mean()
    I["ma60"] = c.rolling(60).mean()
    I["ma120"] = c.rolling(120).mean()
    sd = c.rolling(20).std(ddof=0)
    I["bbu"] = I["ma20"] + 2 * sd
    I["bbl"] = I["ma20"] - 2 * sd
    I["bw"] = (I["bbu"] - I["bbl"]) / I["ma20"]
    I["bw_pct"] = I["bw"].rolling(60).rank(pct=True)
    d = c.diff()
    au = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    ad = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    I["rsi"] = 100 - 100 / (1 + au / ad)
    I["vma20"] = v.shift(1).rolling(20).mean()     # 당일 제외 20일 평균
    I["hi60"] = h.shift(1).rolling(60).max()        # 당일 제외 60일 고가
    I["chg"] = c.pct_change(fill_method=None) * 100
    return I


def near(x, lo, hi, tol):
    """범위 안이면 1, 벗어나면 tol 만큼 멀어질 때까지 선형으로 0까지 감소"""
    x = x.astype(float)
    dist = (lo - x).clip(lower=0) + (x - hi).clip(lower=0)
    return (1 - dist / tol).clip(0, 1).fillna(0)


def evaluate(P, I):
    idx = P["c"].columns
    S = lambda arr: pd.Series(arr, index=idx, dtype=float)
    c, o, h, l, v = (P[k].iloc[-1] for k in "cohlv")
    ma20, ma60, ma120 = I["ma20"].iloc[-1], I["ma60"].iloc[-1], I["ma120"].iloc[-1]
    ma60_5 = I["ma60"].iloc[-6]
    rsi, rsi_min5 = I["rsi"].iloc[-1], I["rsi"].iloc[-5:].min()
    vr = v / I["vma20"].iloc[-1]
    vr3 = P["v"].iloc[-3:].mean() / I["vma20"].iloc[-3]
    chg, hi60 = I["chg"].iloc[-1], I["hi60"].iloc[-1]
    bw, bw5, bwp5 = I["bw"].iloc[-1], I["bw"].iloc[-6], I["bw_pct"].iloc[-6]
    bbu, bbl = I["bbu"].iloc[-1], I["bbl"].iloc[-1]
    body, lsh = (c - o).abs(), np.minimum(o, c) - l
    bull = c > o
    tail = lsh >= body
    touched = (P["l"].iloc[-3:] <= I["bbl"].iloc[-3:] * 1.01).any()

    d60, d20 = (c / ma60 - 1) * 100, (c / ma20 - 1) * 100
    dhi, dh, d120 = (c / hi60 - 1) * 100, (c / h - 1) * 100, (c / ma120 - 1) * 100
    dbl = (c / bbl - 1) * 100
    pb = (c - bbl) / (bbu - bbl)
    candle = lambda t: "양봉" if bull[t] else ("아래꼬리" if tail[t] else "음봉")

    checks = {
        1: [
            ("60일선 위 · 우상향", 25, near(d60, 0, np.inf, 3) * S(np.where(ma60 > ma60_5, 1.0, 0.5)),
             lambda t: f"{d60[t]:+.1f}% {'↗' if ma60[t] > ma60_5[t] else '↘'}"),
            ("20일선 근접 (-3 ~ +2%)", 20, near(d20, -3, 2, 3), lambda t: f"{d20[t]:+.1f}%"),
            ("RSI 40 ~ 55", 20, near(rsi, 40, 55, 10), lambda t: f"{rsi[t]:.1f}"),
            ("조정 중 거래량 감소", 20, near(vr3, 0, 1.0, 0.6), lambda t: f"평균의 {vr3[t]:.2f}배"),
            ("양봉 또는 아래꼬리", 15, (bull | tail).astype(float), candle),
        ],
        2: [
            ("60일 신고가 돌파", 25, near(dhi, 0, np.inf, 3), lambda t: f"{dhi[t]:+.1f}%"),
            ("거래량 2.5배 이상", 25, near(vr, 2.5, np.inf, 1.5), lambda t: f"{vr[t]:.1f}배"),
            ("상승률 +3 ~ 12%", 15, near(chg, 3, 12, 3), lambda t: f"{chg[t]:+.1f}%"),
            ("밴드 수축 후 확장", 20, near(bwp5, 0, 0.3, 0.3) * S(np.where(bw > bw5, 1.0, 0.4)),
             lambda t: f"폭 하위 {bwp5[t] * 100:.0f}% → {'확장' if bw[t] > bw5[t] else '수축'}"),
            ("윗꼬리 짧음 (고가 -3% 이내)", 15, near(dh, -3, np.inf, 3), lambda t: f"고가 대비 {dh[t]:+.1f}%"),
        ],
        3: [
            ("120일선 -10% 이내", 15, near(c / ma120, 0.9, np.inf, 0.1), lambda t: f"{d120[t]:+.1f}%"),
            ("BB 하단 터치 후 회복", 25,
             S(np.where(touched & (c > bbl), 1.0, np.where(touched, 0.4, near(dbl, -np.inf, 3, 5) * 0.6))),
             lambda t: ("터치 후 회복" if c[t] > bbl[t] else "하단 아래") if touched[t] else f"하단 +{dbl[t]:.1f}%"),
            ("RSI 30 상향 돌파", 20,
             S(np.where((rsi_min5 < 30) & (rsi >= 30), 1.0, near(rsi, 25, 35, 10) * 0.6)),
             lambda t: f"{rsi_min5[t]:.0f} → {rsi[t]:.0f}"),
            ("거래량 2배 이상", 20, near(vr, 2, np.inf, 1), lambda t: f"{vr[t]:.1f}배"),
            ("양봉 + 긴 아래꼬리", 20, S(np.where(bull & tail, 1.0, np.where(bull | tail, 0.5, 0.0))), candle),
        ],
    }
    scores = pd.DataFrame({f"s{k}": sum(w * sc for _, w, sc, _ in v_) for k, v_ in checks.items()})
    stops = {1: ma20 * 0.97, 2: hi60, 3: P["l"].iloc[-5:].min()}
    metrics = dict(c=c, chg=chg, vr=vr, rsi=rsi, pb=pb, a=P["a"].iloc[-1])
    return dict(scores=scores.fillna(0), checks=checks, stops=stops, m=metrics)


# ───────────────────────────── 결과 조립 ─────────────────────────────
def num(x, nd=0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, nd) if nd else int(round(x))


def build(demo=False, mode="file"):
    print("데이터 준비 중…", flush=True)
    P, mcap, market, namefn = load_demo(CFG["days"]) if demo else load_naver(CFG["days"])
    I = indicators(P)
    ev = evaluate(P, I)
    m = ev["m"]
    c = m["c"]

    hist = P["c"].iloc[-CFG["min_history"]:].notna().sum() >= CFG["min_history"]
    codes = pd.Series(c.index, index=c.index)
    mask = (hist & (m["a"] >= CFG["min_amount"]) & (c >= CFG["min_price"]) & (P["v"].iloc[-1] > 0)
            & codes.map(lambda t: t in market and t.endswith("0"))
            & (mcap.reindex(c.index).fillna(np.inf) >= CFG["min_mcap"]))
    names = {t: namefn(t) for t in c.index[mask]}
    mask &= codes.map(lambda t: t in names and "스팩" not in str(names[t]))
    uni = c.index[mask]

    sc = ev["scores"].loc[uni]
    best = sc.max(axis=1)
    strat = sc.values.argmax(axis=1) + 1
    order = pd.DataFrame({"s": best, "a": m["a"][uni], "st": strat}, index=uni).sort_values(["s", "a"], ascending=False)
    top = order.head(CFG["top_n"])

    def series(t):
        sl = slice(-CFG["chart_days"], None)
        arr = lambda s, nd=0: [num(x, nd) for x in s[t].iloc[sl]]
        return dict(
            d=[x.strftime("%Y-%m-%d") for x in P["c"].index[sl]],
            o=arr(P["o"]), h=arr(P["h"]), l=arr(P["l"]), c=arr(P["c"]), v=arr(P["v"]),
            bbu=arr(I["bbu"], 1), bbm=arr(I["ma20"], 1), bbl=arr(I["bbl"], 1),
            ma60=arr(I["ma60"], 1), vma=arr(I["vma20"]), rsi=arr(I["rsi"], 1),
        )

    def safe(fn, t):
        try:
            return fn(t)
        except Exception:
            return "-"

    stocks = []
    for rank, (t, row) in enumerate(top.iterrows(), 1):
        st = int(row["st"])
        e = dict(
            rank=rank, code=t, name=str(names[t]), market=market.get(t, ""),
            close=num(c[t]), chg=num(m["chg"][t], 2), amount=num(m["a"][t]), mcap=num(mcap.get(t)),
            vr=num(m["vr"][t], 2), rsi=num(m["rsi"][t], 1), pb=num(m["pb"][t], 2),
            score=num(row["s"]), strat=st, scores=[num(sc.loc[t, k]) for k in ("s1", "s2", "s3")],
            checks=[dict(label=lb, s=num(s_[t], 2), v=safe(fn, t)) for lb, _, s_, fn in ev["checks"][st]],
            stop=dict(price=num(ev["stops"][st][t]), label=STOP_LABEL[st]),
        )
        if rank <= 6:
            e["series"] = series(t)
        stocks.append(e)

    above20 = (c[uni] > I["ma20"].iloc[-1][uni]).mean() * 100
    above60 = (c[uni] > I["ma60"].iloc[-1][uni]).mean() * 100
    rising = (m["chg"][uni] > 0).mean() * 100
    return dict(
        asof=P["c"].index[-1].strftime("%Y-%m-%d"),
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        mode=mode, demo=demo, universe=int(len(uni)),
        breadth=dict(above20=num(above20, 1), above60=num(above60, 1), rising=num(rising, 1)),
        stocks=stocks,
    )


def write_html(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def data_boundary(now=None):
    now = now or datetime.now()
    b = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return b if now >= b else b - timedelta(days=1)


def needs_rebuild(path):
    return not os.path.exists(path) or datetime.fromtimestamp(os.path.getmtime(path)) < data_boundary()


def serve(port, demo, out, open_browser):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path, _, query = self.path.partition("?")
            if path not in ("/", "/index.html"):
                self.send_response(404)
                self.end_headers()
                return
            with lock:
                try:
                    if "force=1" in query or needs_rebuild(out):
                        write_html(build(demo, mode="serve"), out)
                        print(f"[{datetime.now():%H:%M}] 대시보드 갱신 완료", flush=True)
                    with open(out, "rb") as f:
                        body = f.read()
                except Exception as ex:
                    body = (f"<meta charset='utf-8'><h2>데이터를 불러오지 못했습니다</h2><pre>{ex}</pre>"
                            "<p>인터넷 연결을 확인한 뒤 다시 새로고침해 보세요.</p>").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    url = f"http://localhost:{port}/"
    print(f"대시보드 서버 실행 중: {url}  (종료: Ctrl+C)")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser(description="매일 아침 종목 발굴 대시보드")
    ap.add_argument("--serve", action="store_true", help="로컬 서버 모드 (새로고침 시 자동 갱신)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--demo", action="store_true", help="가상 데이터로 미리보기")
    ap.add_argument("--out", default=None, help="저장할 HTML 경로")
    ap.add_argument("--no-open", action="store_true", help="브라우저 자동 열기 끄기")
    args = ap.parse_args()
    out = args.out or os.path.join(BASE, "dashboard_demo.html" if args.demo else "dashboard.html")

    if args.serve:
        serve(args.port, args.demo, out, not args.no_open)
    else:
        write_html(build(args.demo), out)
        print(f"완료: {out}")
        if not args.no_open:
            webbrowser.open("file://" + os.path.abspath(out))


# ───────────────────────────── HTML 템플릿 ─────────────────────────────
TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>오늘의 종목 발굴</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">
<style>
:root{
  --bg:#eceff3; --surface:#ffffff; --sunk:#f5f7f9; --line:#dde2e8; --ink:#16202a; --muted:#5d6874; --faint:#9aa4af;
  --up:#d42a3c; --down:#1d5bd8; --band:rgba(29,91,216,.06); --bb:#8090a8; --ma:#b45309; --rsi:#7c3aed;
  --vp:rgba(93,104,116,.18); --poc:rgba(180,83,9,.42);
  --ok:#0f766e; --mid:#b7791f; --no:#b8c0c8;
  --s1:#0f766e; --s2:#6d28d9; --s3:#b45309;
  color-scheme:light;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0f1418; --surface:#171d22; --sunk:#1c2329; --line:#2a333b; --ink:#e7ecf0; --muted:#98a3ad; --faint:#5f6a74;
    --up:#ff5a68; --down:#5b95ff; --band:rgba(91,149,255,.08); --bb:#7f8ea5; --ma:#f59e0b; --rsi:#a78bfa;
    --vp:rgba(152,163,173,.16); --poc:rgba(245,158,11,.40);
    --ok:#2dd4bf; --mid:#fbbf24; --no:#46515b;
    --s1:#2dd4bf; --s2:#a78bfa; --s3:#f59e0b;
    color-scheme:dark;
  }
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink)}
body{font-family:"Pretendard","Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",system-ui,sans-serif;font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
.tn{font-variant-numeric:tabular-nums}
.wrap{max-width:1240px;margin:0 auto;padding:28px 20px 64px}
.upc{color:var(--up)} .downc{color:var(--down)}

header{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap}
h1{font-size:32px;line-height:1.15;margin:0;font-weight:800;letter-spacing:-.025em}
.meta{color:var(--muted);font-size:13px;margin-top:6px}
.btn{font:inherit;font-weight:700;font-size:14px;color:var(--surface);background:var(--ink);border:0;border-radius:10px;padding:10px 16px;cursor:pointer}
.btn:focus-visible{outline:3px solid var(--down);outline-offset:2px}
.demo{margin:16px 0 0;padding:10px 14px;border-radius:10px;background:var(--sunk);border:1px dashed var(--mid);color:var(--muted);font-size:13px}

.pulse{display:grid;grid-template-columns:repeat(3,minmax(0,1fr)) minmax(0,1.6fr);gap:1px;margin:20px 0 0;background:var(--line);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.pulse>div{background:var(--surface);padding:12px 16px}
.pulse dt{font-size:12px;color:var(--muted)}
.pulse dd{margin:2px 0 0;font-size:22px;font-weight:800;letter-spacing:-.02em}
.pulse .gauge{height:4px;border-radius:2px;background:var(--sunk);margin-top:6px;overflow:hidden}
.pulse .gauge i{display:block;height:100%;background:var(--ink)}
.pulse .regime dd{font-size:15px;font-weight:700;letter-spacing:0}
.pulse .regime p{margin:2px 0 0;font-size:12.5px;color:var(--muted)}
@media(max-width:820px){.pulse{grid-template-columns:repeat(3,1fr)}.pulse .regime{grid-column:1/-1}}

h2{font-size:19px;font-weight:800;margin:40px 0 4px;letter-spacing:-.015em}
.sub{color:var(--muted);font-size:13px;margin:0 0 14px}

.tag{display:inline-block;font-size:12px;font-weight:700;padding:2px 9px;border-radius:999px;border:1.5px solid currentColor;white-space:nowrap}
.s1{color:var(--s1)} .s2{color:var(--s2)} .s3{color:var(--s3)}

.pick{display:grid;grid-template-columns:310px minmax(0,1fr);background:var(--surface);border:1px solid var(--line);border-radius:16px;overflow:hidden;margin-bottom:16px}
.pick .info{padding:20px;background:var(--sunk);border-right:1px solid var(--line);position:relative}
.pick .big{position:absolute;right:16px;top:6px;font-size:84px;font-weight:900;line-height:1;letter-spacing:-.06em;color:var(--line)}
.pick .chartbox{padding:12px 6px 8px 8px;min-width:0}
@media(max-width:880px){.pick{grid-template-columns:1fr}.pick .info{border-right:0;border-bottom:1px solid var(--line)}}
.name{font-size:22px;font-weight:800;letter-spacing:-.02em;margin:10px 0 0;position:relative}
.code{font-size:12.5px;color:var(--muted);font-weight:500}
.price{font-size:26px;font-weight:800;letter-spacing:-.02em;margin:6px 0 0}
.price span{font-size:15px;font-weight:700;margin-left:6px}
.score{display:flex;align-items:center;gap:10px;margin:14px 0 4px}
.score .bar{flex:1;height:8px;background:var(--line);border-radius:4px;overflow:hidden}
.score .bar i{display:block;height:100%;border-radius:4px;background:currentColor}
.score b{font-size:20px;font-weight:800;color:var(--ink)} .score small{color:var(--muted)}
.strats{display:grid;grid-template-columns:repeat(3,auto);justify-content:space-between;gap:6px;font-size:11.5px;color:var(--muted)}
.strats b{font-weight:700}
.checks{list-style:none;padding:0;margin:14px 0 0;display:grid;gap:5px}
.checks li{display:grid;grid-template-columns:14px 1fr auto;gap:8px;align-items:baseline;font-size:13px}
.checks .ic{font-size:11px}
.checks .ok .ic{color:var(--ok)} .checks .mid .ic{color:var(--mid)} .checks .no .ic{color:var(--no)}
.checks .no .lb{color:var(--muted)}
.checks .vl{font-size:12px;color:var(--muted)}
.kv{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0 0;padding-top:14px;border-top:1px dashed var(--line)}
.kv dt{font-size:11.5px;color:var(--muted)} .kv dd{margin:0;font-weight:700;font-size:13.5px}
.stop{margin-top:14px;font-size:12.5px;display:flex;justify-content:space-between;gap:8px;padding:9px 12px;border-radius:9px;background:var(--surface);border:1px solid var(--line)}
.stop b{color:var(--down)}

.watch{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
@media(max-width:920px){.watch{grid-template-columns:1fr}}
.w{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px}
.w .top{display:flex;justify-content:space-between;align-items:center}
.w .rk{font-weight:800;font-size:13px;color:var(--muted)}
.w .name{font-size:18px;margin-top:6px}
.w .price{font-size:20px}
.w .mini{margin:10px -4px 0}

.tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:14px}
table{border-collapse:collapse;width:100%;min-width:760px;font-size:13.5px}
th{font-size:12px;color:var(--muted);font-weight:600;text-align:right;padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
tr:last-child td{border-bottom:0}
th:nth-child(-n+2),td:nth-child(-n+2){text-align:left}
td.nm b{font-weight:700} td.nm span{color:var(--muted);font-size:12px;margin-left:6px}
td.rk{color:var(--muted);font-weight:700;width:44px}

footer{margin-top:44px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:12.5px;display:grid;grid-template-columns:repeat(3,1fr);gap:20px}
footer h4{margin:0 0 4px;font-size:13px;color:var(--ink)}
footer p{margin:0}
footer .note{grid-column:1/-1}
@media(max-width:820px){footer{grid-template-columns:1fr}}

svg text{font-family:inherit}
.chart .grid{stroke:var(--line);stroke-width:1}
.chart .axis{fill:var(--faint);font-size:11px}
.chart .up{fill:var(--up);stroke:var(--up)}
.chart .down{fill:var(--down);stroke:var(--down)}
.chart .band{fill:var(--band)}
.chart .bb{fill:none;stroke:var(--bb);stroke-width:1.1}
.chart .bbm{fill:none;stroke:var(--bb);stroke-width:1;stroke-dasharray:3 3}
.chart .ma{fill:none;stroke:var(--ma);stroke-width:1.4}
.chart .vma{fill:none;stroke:var(--ink);stroke-width:1;opacity:.55}
.chart .vp{fill:var(--vp)} .chart .poc{fill:var(--poc)}
.chart .rsi{fill:none;stroke:var(--rsi);stroke-width:1.6}
.chart .zone{fill:var(--band)}
.chart .lvl{stroke:var(--faint);stroke-dasharray:3 3}
.chart .cross{stroke:var(--muted);stroke-dasharray:2 3}
.chart .lg{fill:var(--muted);font-size:11.5px}
.chart .lg .k{fill:var(--ink);font-weight:700}
.chart .tagbg{fill:var(--ink)} .chart .tagtx{fill:var(--surface);font-size:11px;font-weight:700}
.chart .poctx{fill:var(--ma);font-size:10.5px;font-weight:700}
.chart .cl{fill:none;stroke:var(--ink);stroke-width:1.5}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>오늘의 종목 발굴</h1>
      <div class="meta tn" id="meta"></div>
    </div>
    <button class="btn" id="refresh">새로고침</button>
  </header>
  <div class="demo" id="demo" hidden>가상 데이터로 만든 미리보기입니다. 실제 종목이 아닙니다.</div>
  <dl class="pulse" id="pulse"></dl>

  <h2>매수 후보 1–3위</h2>
  <p class="sub">세 가지 전략 중 가장 잘 맞는 전략의 점수 순입니다. 차트: 캔들 · 볼린저밴드(20,2) · 60일선 · 오른쪽 매물대(진한 막대가 최대 거래 가격대) · 거래량 · RSI(14)</p>
  <div id="picks"></div>

  <h2>지켜볼 종목 4–6위</h2>
  <p class="sub">조건에 근접했지만 아직 덜 갖춘 종목입니다. 빠진 조건(○, ◐)이 채워지는지 지켜보세요.</p>
  <div class="watch" id="watch"></div>

  <h2>그 외 후보 7–16위</h2>
  <p class="sub">점수 순 목록입니다.</p>
  <div class="tablewrap"><table id="rest"></table></div>

  <footer>
    <div><h4 class="s1">추세 눌림목</h4><p>60일선 위 상승 추세에서 20일선까지 쉬어 내려온 종목. 거래량이 줄며 조정되면 가점.</p></div>
    <div><h4 class="s2">거래량 돌파</h4><p>밴드가 좁아진 뒤 거래량을 동반해 60일 고가를 넘긴 종목. 윗꼬리가 짧을수록 가점.</p></div>
    <div><h4 class="s3">과매도 반등</h4><p>볼린저 하단을 찍고 회복하며 RSI가 30을 넘어선 종목. 대량 거래 양봉·아래꼬리가 핵심.</p></div>
    <p class="note">● 충족 ◐ 근접 ○ 미충족 · 손절 참고가는 전략별 기준입니다. 관리종목·투자경고 여부와 공시는 매수 전 HTS에서 직접 확인하세요. 기술적 지표 기반 참고 자료이며 투자 판단과 책임은 본인에게 있습니다.</p>
  </footer>
</div>

<script>
const D = __DATA__;
const STRAT = {1:{n:'추세 눌림목',c:'s1'},2:{n:'거래량 돌파',c:'s2'},3:{n:'과매도 반등',c:'s3'}};
const NS = 'http://www.w3.org/2000/svg';
const f0 = x => x==null ? '-' : Math.round(x).toLocaleString('ko-KR');
const pct = (x,d=2) => x==null ? '-' : `<span class="${x>0?'upc':x<0?'downc':''}">${x>0?'+':''}${x.toFixed(d)}%</span>`;
const won = x => x==null ? '-' : x>=1e12 ? (x/1e12).toFixed(1)+'조' : f0(x/1e8)+'억';
const ckIcon = s => s>=.99 ? ['ok','●'] : s>=.4 ? ['mid','◐'] : ['no','○'];

function el(tag, a, p){ const e=document.createElementNS(NS,tag); for(const k in a) e.setAttribute(k,a[k]); if(p) p.appendChild(e); return e; }
function tx(p, x, y, s, cls, anchor){ const t=el('text',{x,y,class:cls||'',...(anchor?{'text-anchor':anchor}:{})},p); t.textContent=s; return t; }
function path(p, arr, X, Y, cls){
  let d='', pen=false;
  arr.forEach((v,i)=>{ if(v==null){pen=false;return;} d+=(pen?'L':'M')+X(i).toFixed(1)+','+Y(v).toFixed(1); pen=true; });
  return el('path',{d,class:cls},p);
}
function ticks(lo, hi, n){
  const raw=(hi-lo)/n, mag=10**Math.floor(Math.log10(raw)), f=raw/mag;
  const st=(f<1.5?1:f<3?2:f<7?5:10)*mag, out=[];
  for(let v=Math.ceil(lo/st)*st; v<=hi; v+=st) out.push(v);
  return out;
}
function legend(t, parts){ t.textContent=''; parts.forEach(([s,c])=>{ const sp=document.createElementNS(NS,'tspan'); if(c) sp.setAttribute('class',c); sp.textContent=s; t.appendChild(sp); }); }

function drawChart(box, s){
  box.innerHTML='';
  const W=Math.max(300, box.clientWidth), small=W<560;
  const m={l:4,r:66,t:24,b:20}, hP=small?230:300, hV=64, hR=80, g=14;
  const H=m.t+hP+g+hV+g+hR+m.b;
  const svg=el('svg',{width:W,height:H,viewBox:`0 0 ${W} ${H}`,class:'chart',role:'img','aria-label':'가격 차트'},box);
  const n=s.c.length, pw=W-m.l-m.r, step=pw/n, cw=Math.max(1,Math.min(9,step*.64));
  const X=i=>m.l+step*(i+.5);
  const pT=m.t, pB=pT+hP, vT=pB+g, vB=vT+hV, rT=vB+g, rB=rT+hR;

  let lo=Infinity, hi=-Infinity;
  for(let i=0;i<n;i++){ lo=Math.min(lo,s.l[i],s.bbl[i]??Infinity); hi=Math.max(hi,s.h[i],s.bbu[i]??-Infinity); }
  const pad=(hi-lo)*.06; lo-=pad; hi+=pad;
  const YP=p=>pB-(p-lo)/(hi-lo)*hP;

  for(const t of ticks(lo,hi,small?4:5)){ const y=YP(t); el('line',{x1:m.l,x2:m.l+pw,y1:y,y2:y,class:'grid'},svg); tx(svg,m.l+pw+6,y+4,f0(t),'axis'); }

  // 매물대 (가격대별 거래량)
  const bins=28, bh=(hi-lo)/bins, vp=new Array(bins).fill(0);
  for(let i=0;i<n;i++){
    const a=Math.max(0,Math.floor((s.l[i]-lo)/bh)), b=Math.min(bins-1,Math.floor((s.h[i]-lo)/bh)), k=b-a+1;
    for(let j=a;j<=b;j++) vp[j]+=s.v[i]/k;
  }
  const vmx=Math.max(...vp), poc=vp.indexOf(vmx), vpw=pw*.26;
  vp.forEach((x,j)=>{ const w=x/vmx*vpw, y1=YP(lo+(j+1)*bh), y2=YP(lo+j*bh);
    el('rect',{x:m.l+pw-w,y:y1+.5,width:w,height:Math.max(.5,y2-y1-1),class:j===poc?'poc':'vp'},svg); });
  if(!small) tx(svg, m.l+pw-vpw-4, YP(lo+(poc+.5)*bh)+4, '매물대 최대', 'poctx', 'end');

  // 볼린저밴드 & 60일선
  const up=[], dn=[];
  s.bbu.forEach((v,i)=>{ if(v!=null&&s.bbl[i]!=null){ up.push(X(i).toFixed(1)+','+YP(v).toFixed(1)); dn.unshift(X(i).toFixed(1)+','+YP(s.bbl[i]).toFixed(1)); } });
  if(up.length) el('path',{d:'M'+up.join('L')+'L'+dn.join('L')+'Z',class:'band'},svg);
  path(svg,s.bbu,X,YP,'bb'); path(svg,s.bbl,X,YP,'bb'); path(svg,s.bbm,X,YP,'bbm'); path(svg,s.ma60,X,YP,'ma');

  // 캔들
  for(let i=0;i<n;i++){
    const cls=s.c[i]>=s.o[i]?'up':'down', x=X(i);
    el('line',{x1:x,x2:x,y1:YP(s.h[i]),y2:YP(s.l[i]),class:cls,'stroke-width':1},svg);
    const y1=YP(Math.max(s.o[i],s.c[i])), y2=YP(Math.min(s.o[i],s.c[i]));
    el('rect',{x:x-cw/2,y:y1,width:cw,height:Math.max(1,y2-y1),class:cls,'stroke-width':0},svg);
  }
  const ly=YP(s.c[n-1]);
  el('rect',{x:m.l+pw+1,y:ly-9,width:m.r-2,height:18,rx:4,class:'tagbg'},svg);
  tx(svg,m.l+pw+6,ly+4,f0(s.c[n-1]),'tagtx');

  // 거래량
  const vMax=Math.max(...s.v)*1.08, YV=v=>vB-v/vMax*hV;
  el('line',{x1:m.l,x2:m.l+pw,y1:vB,y2:vB,class:'grid'},svg);
  for(let i=0;i<n;i++) el('rect',{x:X(i)-cw/2,y:YV(s.v[i]),width:cw,height:vB-YV(s.v[i]),class:(s.c[i]>=s.o[i]?'up':'down'),'stroke-width':0,opacity:.75},svg);
  path(svg,s.vma,X,YV,'vma');

  // RSI
  const YR=r=>rB-r/100*hR;
  el('rect',{x:m.l,y:YR(30),width:pw,height:YR(0)-YR(30),class:'zone'},svg);
  [30,70].forEach(v=>{ el('line',{x1:m.l,x2:m.l+pw,y1:YR(v),y2:YR(v),class:'lvl'},svg); tx(svg,m.l+pw+6,YR(v)+4,v,'axis'); });
  el('line',{x1:m.l,x2:m.l+pw,y1:rB,y2:rB,class:'grid'},svg);
  path(svg,s.rsi,X,YR,'rsi');

  const every=Math.ceil(n/(small?4:6));
  for(let i=every;i<n;i+=every) tx(svg,X(i),H-4,s.d[i].slice(5).replace('-','.'),'axis','middle');

  const L1=tx(svg,m.l+2,14,'','lg'), L2=tx(svg,m.l+2,vT+11,'','lg'), L3=tx(svg,m.l+2,rT+11,'','lg');
  function show(i){
    const prev=i>0?s.c[i-1]:s.o[i], ch=(s.c[i]/prev-1)*100;
    const chp=[(ch>0?'+':'')+ch.toFixed(2)+'%',ch>=0?'k up':'k down'];
    legend(L1, small ? [[s.d[i].slice(5).replace('-','.')+'  ','k'],['종 ',''],[f0(s.c[i])+' ','k'],chp] :
      [[s.d[i]+'  ','k'],['시 ',''],[f0(s.o[i])+'  ','k'],['고 ',''],[f0(s.h[i])+'  ','k'],['저 ',''],[f0(s.l[i])+'  ','k'],['종 ',''],[f0(s.c[i])+' ','k'],chp,
       ['   BB ',''],[`${f0(s.bbu[i])} / ${f0(s.bbl[i])}`,'k']]);
    const vr=s.vma[i]?s.v[i]/s.vma[i]:null;
    legend(L2,[['거래량 ',''],[f0(s.v[i]),'k'],...(vr?[[`  (평균의 ${vr.toFixed(1)}배)`,'']]:[])]);
    legend(L3,[['RSI(14) ',''],[s.rsi[i]==null?'-':s.rsi[i].toFixed(1),'k']]);
  }
  show(n-1);
  const vl=el('line',{y1:pT,y2:rB,class:'cross',visibility:'hidden'},svg);
  const ov=el('rect',{x:m.l,y:pT,width:pw,height:rB-pT,fill:'transparent'},svg);
  ov.addEventListener('pointermove',e=>{
    const r=svg.getBoundingClientRect(), x=(e.clientX-r.left)*W/r.width;
    const i=Math.max(0,Math.min(n-1,Math.floor((x-m.l)/step)));
    vl.setAttribute('x1',X(i)); vl.setAttribute('x2',X(i)); vl.setAttribute('visibility','visible'); show(i);
  });
  ov.addEventListener('pointerleave',()=>{ vl.setAttribute('visibility','hidden'); show(n-1); });
}

function drawMini(box, s){
  box.innerHTML='';
  const k=60, W=Math.max(240,box.clientWidth), H=96, m=6;
  const c=s.c.slice(-k), u=s.bbu.slice(-k), b=s.bbl.slice(-k), n=c.length;
  const lo=Math.min(...c,...b.filter(x=>x!=null)), hi=Math.max(...c,...u.filter(x=>x!=null));
  const X=i=>m+(W-2*m)*i/(n-1), Y=v=>H-m-(v-lo)/(hi-lo)*(H-2*m);
  const svg=el('svg',{width:W,height:H,viewBox:`0 0 ${W} ${H}`,class:'chart',role:'img','aria-label':'최근 60일 종가와 볼린저밴드'},box);
  const up=[],dn=[]; u.forEach((v,i)=>{ if(v!=null&&b[i]!=null){ up.push(X(i)+','+Y(v)); dn.unshift(X(i)+','+Y(b[i])); } });
  el('path',{d:'M'+up.join('L')+'L'+dn.join('L')+'Z',class:'band'},svg);
  path(svg,u,X,Y,'bb'); path(svg,b,X,Y,'bb'); path(svg,c,X,Y,'cl');
  el('circle',{cx:X(n-1),cy:Y(c[n-1]),r:3.5,class:c[n-1]>=c[n-2]?'up':'down'},svg);
}

function checksHTML(list){
  return '<ul class="checks">'+list.map(k=>{ const [c,i]=ckIcon(k.s); return `<li class="${c}"><span class="ic">${i}</span><span class="lb">${k.label}</span><span class="vl tn">${k.v}</span></li>`; }).join('')+'</ul>';
}
function scoreHTML(x){
  const st=STRAT[x.strat];
  return `<div class="score ${st.c}"><div class="bar"><i style="width:${x.score}%"></i></div><b class="tn">${x.score}</b><small>/100</small></div>
    <div class="strats tn">${[1,2,3].map(k=>`<span class="${k===x.strat?STRAT[k].c:''}">${STRAT[k].n} <b>${x.scores[k-1]}</b></span>`).join('')}</div>`;
}

function render(){
  document.getElementById('meta').textContent = `${D.asof} 종가 기준 · 분석 ${D.universe.toLocaleString()}종목 · ${D.generated} 생성`;
  document.getElementById('demo').hidden = !D.demo;
  const B=D.breadth;
  const regime = B.above20>=60 ? ['상승 우위','추세 눌림목·거래량 돌파 위주로 보세요.'] :
                 B.above20>=40 ? ['중립','전략을 섞되 종목 수를 줄이세요.'] :
                                 ['약세','과매도 반등 위주로, 비중을 줄이세요.'];
  const g=(t,v)=>`<div><dt>${t}</dt><dd class="tn">${v==null?'-':v.toFixed(0)+'%'}</dd><div class="gauge"><i style="width:${v||0}%"></i></div></div>`;
  document.getElementById('pulse').innerHTML = g('20일선 위 종목',B.above20)+g('60일선 위 종목',B.above60)+g('상승 마감 종목',B.rising)+
    `<div class="regime"><dt>시장 분위기</dt><dd>${regime[0]}</dd><p>${regime[1]}</p></div>`;

  const picks=D.stocks.slice(0,3), watch=D.stocks.slice(3,6), rest=D.stocks.slice(6);
  document.getElementById('picks').innerHTML = picks.length ? picks.map((x,i)=>`
    <article class="pick">
      <div class="info">
        <span class="big tn" aria-hidden="true">${x.rank}</span>
        <span class="tag ${STRAT[x.strat].c}">${STRAT[x.strat].n}</span>
        <div class="name">${x.name} <span class="code tn">${x.code} ${x.market}</span></div>
        <div class="price tn">${f0(x.close)}<span>${pct(x.chg)}</span></div>
        ${scoreHTML(x)}
        ${checksHTML(x.checks)}
        <dl class="kv tn">
          <div><dt>거래대금</dt><dd>${won(x.amount)}</dd></div>
          <div><dt>시가총액</dt><dd>${won(x.mcap)}</dd></div>
          <div><dt>거래량</dt><dd>${x.vr==null?'-':x.vr.toFixed(1)+'배'}</dd></div>
          <div><dt>RSI</dt><dd>${x.rsi??'-'}</dd></div>
          <div><dt>밴드 위치 %B</dt><dd>${x.pb??'-'}</dd></div>
          <div><dt>순위</dt><dd>${x.rank}위</dd></div>
        </dl>
        <div class="stop tn"><span>손절 참고 · ${x.stop.label}</span><b>${f0(x.stop.price)}</b></div>
      </div>
      <div class="chartbox" data-i="${i}"></div>
    </article>`).join('') : '<p class="sub">조건을 통과한 종목이 없습니다. 필터 기준(거래대금·시가총액)을 낮춰 보세요.</p>';

  document.getElementById('watch').innerHTML = watch.map((x,i)=>`
    <article class="w">
      <div class="top"><span class="rk">${x.rank}위</span><span class="tag ${STRAT[x.strat].c}">${STRAT[x.strat].n}</span></div>
      <div class="name">${x.name} <span class="code tn">${x.code}</span></div>
      <div class="price tn">${f0(x.close)}<span>${pct(x.chg)}</span></div>
      <div class="mini" data-i="${i+3}"></div>
      ${scoreHTML(x)}
      ${checksHTML(x.checks)}
      <div class="stop tn"><span>손절 참고 · ${x.stop.label}</span><b>${f0(x.stop.price)}</b></div>
    </article>`).join('');

  document.getElementById('rest').innerHTML = `<thead><tr><th>순위</th><th>종목</th><th>종가</th><th>등락률</th><th>거래량</th><th>RSI</th><th>%B</th><th>거래대금</th><th>전략</th><th>점수</th></tr></thead><tbody>`+
    rest.map(x=>`<tr class="tn"><td class="rk">${x.rank}</td><td class="nm"><b>${x.name}</b><span>${x.code} ${x.market}</span></td>
      <td>${f0(x.close)}</td><td>${pct(x.chg)}</td><td>${x.vr==null?'-':x.vr.toFixed(1)+'배'}</td><td>${x.rsi??'-'}</td><td>${x.pb??'-'}</td>
      <td>${won(x.amount)}</td><td><span class="tag ${STRAT[x.strat].c}">${STRAT[x.strat].n}</span></td><td><b>${x.score}</b></td></tr>`).join('')+'</tbody>';
  drawAll();
}
function drawAll(){
  document.querySelectorAll('.chartbox').forEach(b=>drawChart(b, D.stocks[+b.dataset.i].series));
  document.querySelectorAll('.mini').forEach(b=>drawMini(b, D.stocks[+b.dataset.i].series));
}
let rt; addEventListener('resize',()=>{ clearTimeout(rt); rt=setTimeout(drawAll,150); });
document.getElementById('refresh').addEventListener('click',()=>location.reload());
render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()

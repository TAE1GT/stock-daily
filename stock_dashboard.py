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
    - 네이버 금융에서 종목 목록과 최근 160 거래일 시세를 받음 (약 2~4분)
    - 환경변수 ANTHROPIC_API_KEY 가 있으면 전략별 1~3위를 Claude가 뉴스 검색 후 분석 (없으면 버튼 방식)
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
    cand_n=25,             # 전략별 2차 검토(수급·실적 조회) 후보 수
)

STRATS = {1: "추세 눌림목", 2: "거래량 돌파", 3: "과매도 반등"}
STOP_LABEL = {1: "20일선 -3%", 2: "돌파 기준가", 3: "최근 5일 저점"}
TARGET_RULE = {
    1: ("최근 20일 고점", "현재가 + 2.5×ATR"),
    2: ("현재가 + 2×ATR", "돌파 기준가 + 박스권 높이"),
    3: ("20일선(볼린저 중심선)", "볼린저 상단"),
}


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
                raw = r.read()
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("cp949", errors="replace")
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(1.5 * (k + 1))


def _strip(s):
    return re.sub(r"<[^>]+>", "", str(s)).replace("&nbsp;", " ").strip()


def _num(s):
    try:
        return float(_strip(s).replace(",", "").replace("%", ""))
    except ValueError:
        return np.nan


def is_excluded_name(name):
    first = name.split()[0] if name.split() else ""
    return first in ETF_BRANDS or "ETN" in name or "스팩" in name


def _diag(src, text):
    snippet = re.sub(r"\s+", " ", text[:300])
    print(f"  [{src}] 읽기 실패. 응답 앞부분: {snippet}", flush=True)


def listing_mobile_api():
    """네이버 증권 모바일 API (JSON)"""
    rows = []
    for mk in ("KOSPI", "KOSDAQ"):
        n0 = len(rows)
        for page in range(1, 60):
            text = http_get(f"https://m.stock.naver.com/api/stocks/marketValue/{mk}?page={page}&pageSize=100")
            try:
                data = json.loads(text)
            except ValueError:
                _diag("모바일 API", text)
                return []
            items = data.get("stocks") if isinstance(data, dict) else None
            if not items:
                if page == 1:
                    _diag("모바일 API", text)
                break
            for it in items:
                code = str(it.get("itemCode", ""))
                if not re.fullmatch(r"[0-9A-Z]{6}", code):
                    continue
                if it.get("stockEndType", "stock") != "stock":
                    continue
                eng = it.get("stockNameEng") or next((v for k, v in it.items() if "eng" in k.lower() and isinstance(v, str)), "")
                ENG[code] = str(eng or "")
                rows.append(dict(code=code, name=str(it.get("stockName", "")), market=mk,
                                 현재가=_num(it.get("closePrice", "")), 거래량=_num(it.get("accumulatedTradingVolume", "")),
                                 시가총액=_num(it.get("marketValue", ""))))
        print(f"  {mk} 종목 목록 {len(rows) - n0}개 (모바일 API)", flush=True)
    if rows:
        df = pd.DataFrame(rows).drop_duplicates("code").set_index("code")
        # 시가총액 단위 자동 판별: 삼성전자가 수백조 원이 되도록 맞춤 (억 단위 기준)
        ref = df["시가총액"].get("005930", np.nan)
        if ref == ref:
            if ref > 1e8:
                df["시가총액"] = df["시가총액"] / 100      # 백만원 → 억
            elif not (1e5 < ref < 1e8):
                df["시가총액"] = np.nan
        return df
    return []


def listing_pc_page():
    """네이버 금융 PC 시가총액 페이지 (HTML)"""
    rows = []
    for sosok, mk in ((0, "KOSPI"), (1, "KOSDAQ")):
        seen = set()
        for page in range(1, 90):
            html = http_get(f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}")
            heads = [_strip(h) for h in re.findall(r'<th[^>]*scope="col"[^>]*>(.*?)</th>', html, re.S)]
            new = 0
            for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
                mm = re.search(r'<a[^>]*code=([0-9A-Z]{6})[^>]*>(.*?)</a>', tr, re.S)
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
                if page == 1:
                    _diag("PC 페이지", html)
                break
        print(f"  {mk} 종목 목록 {len(seen)}개 (PC 페이지)", flush=True)
    return pd.DataFrame(rows).set_index("code") if rows else []


def listing_kind():
    """한국거래소 KIND 상장법인 목록 (ETF 제외 원본)"""
    rows = []
    for mt, mk in (("stockMkt", "KOSPI"), ("kosdaqMkt", "KOSDAQ")):
        html = http_get(f"https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&marketType={mt}")
        n0 = len(rows)
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
            cells = [_strip(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
            code = next((c for c in cells if re.fullmatch(r"[0-9A-Z]{6}", c)), None)
            if code and cells[0]:
                rows.append(dict(code=code, name=cells[0], market=mk, 현재가=np.nan, 거래량=np.nan, 시가총액=np.nan))
        if len(rows) == n0:
            _diag("KIND", html)
        print(f"  {mk} 종목 목록 {len(rows) - n0}개 (KIND)", flush=True)
    return pd.DataFrame(rows).drop_duplicates("code").set_index("code") if rows else []


def naver_listing():
    for fn in (listing_mobile_api, listing_pc_page, listing_kind):
        try:
            df = fn()
        except Exception as ex:
            print(f"  {fn.__doc__.strip()}: 접속 오류 {ex}", flush=True)
            continue
        if len(df) > 500:
            return df
    raise RuntimeError("종목 목록을 어느 곳에서도 받지 못했습니다. 위 로그의 '응답 앞부분'을 확인하세요.")


def naver_ohlcv(code, count):
    xml = http_get(f"https://fchart.stock.naver.com/sise.nhn?symbol={code}&timeframe=day&count={count}&requestType=0")
    recs = re.findall(r'data="(\d{8})\|([\d.]+)\|([\d.]+)\|([\d.]+)\|([\d.]+)\|([\d.]+)"', xml)
    if not recs:
        return None
    df = pd.DataFrame(recs, columns=["d", "o", "h", "l", "c", "v"])
    df.index = pd.to_datetime(df.pop("d"), format="%Y%m%d")
    return df.astype(float)


def names_of(lst, t):
    try:
        return f"{lst.at[t, 'name']}({t})"
    except Exception:
        return t


ENG = {}   # 종목코드 → 영문명 (해외 뉴스 검색용)


def load_naver(days):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print("종목 목록 받는 중…", flush=True)
    lst = naver_listing()
    # ETF·ETN·스팩 제외 (보유 종목 점검용으로 우선주는 남김. 매수 후보 선정에서는 따로 제외)
    lst = lst[[not is_excluded_name(n) for n in lst["name"]]]
    print(f"시세 받는 중… ({len(lst)}종목)", flush=True)

    data = {}
    todo = list(lst.index)
    for rnd, workers in ((1, 10), (2, 4), (3, 2)):          # 실패한 종목은 천천히 다시 시도
        failed = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(naver_ohlcv, code, days + 5): code for code in todo}
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    df = f.result()
                    if df is not None and len(df):
                        data[futs[f]] = df
                    else:
                        failed.append(futs[f])
                except Exception:
                    failed.append(futs[f])
                if rnd == 1 and i % 300 == 0:
                    print(f"  {i}/{len(futs)}", flush=True)
        print(f"  {rnd}차 수집 후 실패 {len(failed)}종목", flush=True)
        if not failed:
            break
        todo = failed
    if todo and failed:
        print("  끝내 못 받은 종목(일부): " + ", ".join(f"{names_of(lst, t)}" for t in failed[:15]), flush=True)
    if not data:
        try:
            _diag("시세 차트", http_get("https://fchart.stock.naver.com/sise.nhn?symbol=005930&timeframe=day&count=5&requestType=0"))
        except Exception as ex:
            print(f"  시세 차트 접속 오류: {ex}", flush=True)
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
    c, h, l, v = P["c"], P["h"], P["l"], P["v"]
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
    pc = c.shift(1)
    tr = np.fmax(np.fmax((h - l).to_numpy(float), (h - pc).abs().to_numpy(float)), (l - pc).abs().to_numpy(float))
    I["atr"] = pd.DataFrame(tr, index=c.index, columns=c.columns).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    I["hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    I["obv"] = (np.sign(d).fillna(0) * v.fillna(0)).cumsum()
    I["r20"] = c / c.shift(20) - 1
    return I


def near(x, lo, hi, tol):
    """범위 안이면 1, 벗어나면 tol 만큼 멀어질 때까지 선형으로 0까지 감소"""
    x = x.astype(float)
    dist = (lo - x).clip(lower=0) + (x - hi).clip(lower=0)
    return (1 - dist / tol).clip(0, 1).fillna(0)


def evaluate(P, I, market):
    idx = P["c"].columns
    S = lambda arr: pd.Series(arr, index=idx, dtype=float)
    W = lambda cond, a, b: S(np.where(cond, a, b))
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
            ("60일선 위 · 우상향", 25, near(d60, 0, np.inf, 3) * W(ma60 > ma60_5, 1.0, 0.5),
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
            ("밴드 수축 후 확장", 20, near(bwp5, 0, 0.3, 0.3) * W(bw > bw5, 1.0, 0.4),
             lambda t: f"폭 하위 {bwp5[t] * 100:.0f}% → {'확장' if bw[t] > bw5[t] else '수축'}"),
            ("윗꼬리 짧음 (고가 -3% 이내)", 15, near(dh, -3, np.inf, 3), lambda t: f"고가 대비 {dh[t]:+.1f}%"),
        ],
        3: [
            ("120일선 -10% 이내", 15, near(c / ma120, 0.9, np.inf, 0.1), lambda t: f"{d120[t]:+.1f}%"),
            ("BB 하단 터치 후 회복", 25,
             W(touched & (c > bbl), 1.0, np.where(touched, 0.4, near(dbl, -np.inf, 3, 5) * 0.6)),
             lambda t: ("터치 후 회복" if c[t] > bbl[t] else "하단 아래") if touched[t] else f"하단 +{dbl[t]:.1f}%"),
            ("RSI 30 상향 돌파", 20,
             W((rsi_min5 < 30) & (rsi >= 30), 1.0, near(rsi, 25, 35, 10) * 0.6),
             lambda t: f"{rsi_min5[t]:.0f} → {rsi[t]:.0f}"),
            ("거래량 2배 이상", 20, near(vr, 2, np.inf, 1), lambda t: f"{vr[t]:.1f}배"),
            ("양봉 + 긴 아래꼬리", 20, W(bull & tail, 1.0, np.where(bull | tail, 0.5, 0.0)), candle),
        ],
    }
    scores = pd.DataFrame({f"s{k}": sum(w * sc for _, w, sc, _ in v_) for k, v_ in checks.items()}).fillna(0)

    # ── 보조 지표 (가감점) ──
    h0, h1, h2 = I["hist"].iloc[-1], I["hist"].iloc[-2], I["hist"].iloc[-3]
    rising = h0 > h1
    rising2 = rising & (h1 > h2)
    cross = (h0 > 0) & (h1 <= 0)

    def macd_txt(t):
        if cross[t]:
            return "0선 상향 돌파"
        if h0[t] < 0:
            return "하락 둔화" if rising[t] else "하락 확대"
        return "상승 확대" if rising[t] else "상승 둔화"

    cw = P["c"].iloc[-30:].to_numpy(float)
    ow = I["obv"].iloc[-30:].to_numpy(float)
    cf = np.where(np.isnan(cw), np.inf, cw)
    cols = np.arange(cw.shape[1])
    ia, ib = cf[:20].argmin(0), cf[20:].argmin(0) + 20
    with np.errstate(invalid="ignore"):
        div = S((cw[ib, cols] < cw[ia, cols]) & (ow[ib, cols] > ow[ia, cols])) > 0
    obv_up = I["obv"].iloc[-1] > I["obv"].iloc[-21]
    obv_txt = lambda t: "강세 다이버전스" if div[t] else ("누적 상승" if obv_up[t] else "누적 하락")

    r20 = I["r20"].iloc[-1]
    mk = pd.Series([market.get(t, "") for t in idx], index=idx)
    liquid = P["a"].iloc[-1] >= CFG["min_amount"]
    med = r20[liquid].groupby(mk[liquid]).median()
    rs = (r20 - mk.map(med).astype(float)) * 100
    rs_txt = lambda t: f"시장 대비 {rs[t]:+.1f}%p" if rs[t] == rs[t] else "-"
    rs_trend = W(rs > 5, 4, W(rs > 0, 2, W(rs < -10, -4, 0)))

    aux = {
        1: [("MACD 히스토그램", W(rising, 3, -2), macd_txt),
            ("OBV", W(div, 3, 0) + W(obv_up, 2, -2), obv_txt),
            ("상대강도 (20일)", rs_trend, rs_txt)],
        2: [("MACD 히스토그램", W(rising & (h0 > 0), 3, W(h0 < 0, -3, 0)), macd_txt),
            ("OBV", W(obv_up, 2, -2), obv_txt),
            ("상대강도 (20일)", rs_trend, rs_txt)],
        3: [("MACD 히스토그램", W(cross, 5, W((h0 < 0) & rising2, 5, W(rising, 2, -3))), macd_txt),
            ("OBV", W(div, 6, W(obv_up, 2, 0)), obv_txt),
            ("상대강도 (20일)", W(rs > -5, 3, W(rs < -15, -3, 0)), rs_txt)],
    }

    # ── 목표가 · 손절가 원안 ──
    atr = I["atr"].iloc[-1]
    hi20 = P["h"].iloc[-20:].max()
    lo60 = P["l"].iloc[-61:-1].min()
    tg = {
        1: (hi20.where(hi20 > c * 1.01, c + 1.5 * atr), c + 2.5 * atr, ma20 * 0.97),
        2: (c + 2 * atr, hi60 + (hi60 - lo60), hi60),
        3: (ma20, bbu, P["l"].iloc[-5:].min()),
    }
    metrics = dict(c=c, chg=chg, vr=vr, rsi=rsi, pb=pb, a=P["a"].iloc[-1], atr=atr, rs=rs,
                   h0=h0, h1=h1, ma20=ma20, ma60=ma60, ma60_5=ma60_5, bbu=bbu, bbl=bbl,
                   hi20=hi20, lo20=P["l"].iloc[-20:].min(), hi60=hi60)
    return dict(scores=scores, checks=checks, aux=aux, tg=tg, m=metrics)


def vp_levels(h, l, v, c0, bins=28):
    """매물대: 현재가 바로 위 저항, 바로 아래 지지 가격"""
    ok = ~(np.isnan(h) | np.isnan(l) | np.isnan(v))
    h, l, v = h[ok], l[ok], v[ok]
    if len(h) < 20 or not np.isfinite(c0):
        return None, None
    lo, hi = l.min(), h.max()
    bh = (hi - lo) / bins
    if bh <= 0:
        return None, None
    vp = np.zeros(bins)
    a = np.clip(((l - lo) / bh).astype(int), 0, bins - 1)
    b = np.clip(((h - lo) / bh).astype(int), 0, bins - 1)
    for ai, bi, vi in zip(a, b, v):
        vp[ai:bi + 1] += vi / (bi - ai + 1)
    heavy = vp >= max(np.percentile(vp, 75), 1.2 * np.median(vp))
    res = sup = None
    for j in range(bins):
        e_lo, e_hi = lo + j * bh, lo + (j + 1) * bh
        if heavy[j] and res is None and e_lo > c0 * 1.005:
            res = e_lo
        if heavy[j] and e_hi < c0 * 0.995:
            sup = e_hi
    return res, sup


def target_for(k, t, ev, res):
    m = ev["m"]
    c0, atr = float(m["c"][t]), float(m["atr"][t])
    if not (np.isfinite(atr) and atr > 0):
        atr = c0 * 0.03
    t1, t2, st = (float(x[t]) for x in ev["tg"][k])
    stop_label, note = STOP_LABEL[k], ""
    if not (np.isfinite(st) and st < c0 * 0.995):
        st, stop_label = c0 - 1.5 * atr, "현재가 -1.5×ATR"
    if not (np.isfinite(t1) and t1 > c0 * 1.01):
        t1 = c0 + 1.5 * atr
    if res is not None and res < t1:
        if res * 0.995 > c0 * 1.01:
            t1, note = res * 0.995, "위쪽 매물대에 맞춰 1차 목표 하향"
        else:
            note = "바로 위에 매물대 저항"
    if not (np.isfinite(t2) and t2 > t1 * 1.01):
        t2 = t1 + atr
    rr = (t1 - c0) / (c0 - st)
    return dict(t1=t1, t2=t2, stop=st, rr=rr, note=note, stop_label=stop_label,
                t1p=(t1 / c0 - 1) * 100, t2p=(t2 / c0 - 1) * 100, sp=(st / c0 - 1) * 100)


def rr_delta(rr):
    if not np.isfinite(rr):
        return -12
    return 8 if rr >= 2 else 4 if rr >= 1.5 else 0 if rr >= 1 else -12


def flow_delta(f, i):
    if f is None or i is None:
        return 0
    if f > 0 and i > 0:
        return 6
    if f < 0 and i < 0:
        return -6
    return 3 if f + i > 0 else 0


def man(x):
    return "-" if x is None else f"{x / 1e4:+,.1f}만"


# ───────────────────────────── 수급 · 실적 ─────────────────────────────
def fetch_flow(code):
    try:
        data = json.loads(http_get(f"https://m.stock.naver.com/api/stock/{code}/trend?pageSize=10", retries=2))
        items = data if isinstance(data, list) else next((x for x in data.values() if isinstance(x, list)), [])
        f = i = 0.0
        n = 0
        for it in items[:5]:
            fk = next((k for k in it if re.search(r"foreign.*pure", k, re.I)), None)
            ok = next((k for k in it if re.search(r"organ.*pure", k, re.I)), None)
            if fk is None or ok is None:
                break
            f += np.nan_to_num(_num(it[fk]))
            i += np.nan_to_num(_num(it[ok]))
            n += 1
        if n:
            return f, i
    except Exception:
        pass
    try:
        html = http_get(f"https://finance.naver.com/item/frgn.naver?code={code}", retries=2)
        f = i = 0.0
        n = 0
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
            cells = [_strip(x) for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
            if len(cells) >= 7 and re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", cells[0]):
                i += np.nan_to_num(_num(cells[5]))
                f += np.nan_to_num(_num(cells[6]))
                n += 1
                if n == 5:
                    break
        if n:
            return f, i
    except Exception:
        pass
    return None


def fetch_eps(code):
    try:
        txt = http_get(f"https://m.stock.naver.com/api/stock/{code}/integration", retries=2)
        vals = {}
        for obj in re.findall(r"\{[^{}]*\}", txt):
            mc = re.search(r'"code"\s*:\s*"(eps|per)"', obj, re.I)
            mv = re.search(r'"value"\s*:\s*"([^"]*)"', obj)
            if mc and mv:
                vals[mc.group(1).lower()] = _num(re.sub(r"[^\d.\-]", "", mv.group(1)))
        if vals.get("eps") is not None and vals["eps"] == vals["eps"]:
            return vals["eps"], vals.get("per")
    except Exception:
        pass
    try:
        html = http_get(f"https://finance.naver.com/item/main.naver?code={code}", retries=2)
        me = re.search(r'id="_eps"[^>]*>([^<]+)<', html)
        mp = re.search(r'id="_per"[^>]*>([^<]+)<', html)
        if me:
            eps = _num(me.group(1))
            if eps == eps:
                return eps, (_num(mp.group(1)) if mp else None)
    except Exception:
        pass
    return None


def fetch_extras(codes):
    from concurrent.futures import ThreadPoolExecutor
    print(f"수급·실적 확인 중… ({len(codes)}종목)", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        flows = dict(zip(codes, ex.map(fetch_flow, codes)))
        epss = dict(zip(codes, ex.map(fetch_eps, codes)))
    out = {}
    for t in codes:
        e = {}
        if flows[t]:
            e["f"], e["i"] = flows[t]
        if epss[t]:
            e["eps"], e["per"] = epss[t]
        out[t] = e
    nf = sum(1 for t in codes if flows[t])
    ne = sum(1 for t in codes if epss[t])
    print(f"  수급 {nf}/{len(codes)}, 실적 {ne}/{len(codes)} 확인", flush=True)
    return out, dict(flow=nf, eps=ne, total=len(codes))


def demo_extras(codes):
    rng = np.random.default_rng(7)
    out = {}
    for t in codes:
        eps = rng.normal(1500, 2000)
        out[t] = dict(f=rng.normal(0, 2e5), i=rng.normal(0, 1.5e5), eps=eps,
                      per=(rng.uniform(5, 40) if eps > 0 else None))
    return out, dict(flow=len(codes), eps=len(codes), total=len(codes))


# ───────────────────────────── 뉴스 ─────────────────────────────
def _rss(url, days=31):
    import html as _h
    from email.utils import parsedate_to_datetime
    xml = http_get(url, retries=2)
    out = []
    now = datetime.now().astimezone()
    for it in re.findall(r"<item>(.*?)</item>", xml, re.S):
        g = lambda tag: (re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", it, re.S) or [None, ""])[1]
        title = _h.unescape(re.sub(r"<!\[CDATA\[|\]\]>", "", g("title"))).strip()
        link = _h.unescape(re.sub(r"<!\[CDATA\[|\]\]>", "", g("link"))).strip()
        src = _h.unescape(_strip(g("source")))
        try:
            dt = parsedate_to_datetime(g("pubDate").strip()).astimezone()
        except Exception:
            continue
        if not title or not link.startswith("http") or (now - dt).days > days:
            continue
        if src and title.endswith(" - " + src):
            title = title[: -len(src) - 3]
        out.append(dict(dt=dt, t=title, u=link, s=src))
    return out


def fetch_news(code, name, market):
    from urllib.parse import quote
    ko, en = [], []
    try:
        ko = [x for x in _rss(f"https://news.google.com/rss/search?q={quote(chr(34) + name + chr(34) + ' when:30d')}&hl=ko&gl=KR&ceid=KR:ko")
              if name.replace(" ", "") in x["t"].replace(" ", "")]
    except Exception:
        pass
    eng = ENG.get(code, "")
    try:
        if eng:
            en = _rss(f"https://news.google.com/rss/search?q={quote(chr(34) + eng + chr(34) + ' when:30d')}&hl=en-US&gl=US&ceid=US:en")
    except Exception:
        pass
    if not en:
        try:
            sfx = ".KQ" if market == "KOSDAQ" else ".KS"
            en = _rss(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={code}{sfx}&region=US&lang=en-US")
        except Exception:
            pass
    seen, out = set(), []
    for x, is_en in [(x, 0) for x in sorted(ko, key=lambda x: x["dt"], reverse=True)[:6]] + \
                    [(x, 1) for x in sorted(en, key=lambda x: x["dt"], reverse=True)[:4]]:
        k = x["t"][:30]
        if k in seen:
            continue
        seen.add(k)
        out.append(dict(d=x["dt"].strftime("%m/%d"), t=x["t"][:120], u=x["u"], s=x["s"][:30], en=is_en, ts=x["dt"].timestamp()))
    out.sort(key=lambda x: x["ts"], reverse=True)
    for x in out:
        x.pop("ts")
    return out


def collect_news(stocks, demo):
    if demo:
        rng = np.random.default_rng(3)
        heads = ["3분기 실적 시장 기대치 상회", "신규 공급 계약 체결 공시", "Global peers expand capacity amid demand recovery",
                 "외국인 순매수 이어져", "대주주 지분 일부 매각", "Analysts raise price targets on AI demand"]
        return {e["code"]: [dict(d=f"10/{int(d):02d}", t=heads[j % 6], u="https://example.com", s="예시", en=int("A" <= heads[j % 6][0] <= "Z"))
                            for j, d in enumerate(sorted(rng.integers(1, 15, 4), reverse=True))] for e in stocks}
    from concurrent.futures import ThreadPoolExecutor
    print(f"뉴스 모으는 중… ({len(stocks)}종목)", flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(lambda e: fetch_news(e["code"], e["name"], e["market"]), stocks))
    news = {e["code"]: r for e, r in zip(stocks, res) if r}
    nk = sum(1 for r in res if any(not x["en"] for x in r))
    ne = sum(1 for r in res if any(x["en"] for x in r))
    print(f"  국내 뉴스 {nk}/{len(stocks)}, 해외 뉴스 {ne}/{len(stocks)} 종목 확인", flush=True)
    return news


# ───────────────────────────── AI 분석 ─────────────────────────────
AI_MODEL = "claude-sonnet-5-5"


def _w(x):
    return "-" if x is None else f"{x:,}"


def ai_prompt(e, asof, as_json=False):
    tg = e["tg"]
    miss = [k["label"] for k in e["checks"] if (k["s"] or 0) < 0.99]
    flow = next((x["v"] for x in e["aux"] if x["label"].startswith("수급")), "-")
    p = (f"{e['name']}({e['code']}) 주식 분석해줘.\n"
         f"- {asof} 종가 {_w(e['close'])}원 ({(e['chg'] or 0):+.2f}%)\n"
         f"- 차트: {STRATS[e['strat']]} 전략 {e['score']}점"
         + (f", 미충족: {', '.join(miss)}" if miss else ", 조건 모두 충족") + "\n"
         f"- 목표 {_w(tg['t1'])} / {_w(tg['t2'])}원, 손절 {_w(tg['stop'])}원, 수급 {flow}\n"
         "최근 한 달 국내 뉴스·공시와 해외 영문 기사(Reuters·Bloomberg 등, 한국어로 요약)를 검색해서 "
         "주요 이슈를 날짜순으로 정리하고, 차트 결과와 같은 방향인지 판단해 매수/관망/주의 의견과 근거 3가지, 리스크, 기사 링크를 알려줘.")
    if as_json:
        p += ('\n\n답변은 다른 말 없이 아래 JSON 하나로만 출력해줘:\n'
              '{"verdict":"매수|관망|주의","summary":"한 문장 결론",'
              '"issues":[{"date":"MM/DD","text":"한국어 한 줄 요약","lang":"ko 또는 en"}],'
              '"reasons":["근거1","근거2","근거3"],"risks":["리스크1"],'
              '"sources":[{"title":"기사 제목(영문 기사는 원제목)","url":"https://...","lang":"ko 또는 en"}]}')
    return p


def ai_analyze(e, asof, key):
    import urllib.request
    messages = [{"role": "user", "content": ai_prompt(e, asof, as_json=True)}]
    text = ""
    for _ in range(4):
        body = {"model": AI_MODEL, "max_tokens": 3000, "messages": messages,
                "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 7}]}
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                     headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                              "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=240) as r:
            resp = json.loads(r.read())
        text += "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
        if resp.get("stop_reason") != "pause_turn":
            break
        messages.append({"role": "assistant", "content": resp["content"]})
    m = re.search(r"\{.*\}", text, re.S)
    data = json.loads(m.group(0))
    verdict = data.get("verdict", "관망")
    if verdict not in ("매수", "관망", "주의"):
        verdict = "관망"
    return dict(verdict=verdict, summary=str(data.get("summary", ""))[:200],
                reasons=[str(x)[:160] for x in data.get("reasons", [])][:3],
                risks=[str(x)[:160] for x in data.get("risks", [])][:2],
                issues=[dict(date=str(x.get("date", ""))[:10], text=str(x.get("text", ""))[:160], lang=("en" if x.get("lang") == "en" else "ko"))
                        for x in data.get("issues", []) if isinstance(x, dict)][:6],
                sources=[dict(title=str(s.get("title", ""))[:120], url=s["url"], lang=("en" if s.get("lang") == "en" else "ko"))
                         for s in data.get("sources", []) if isinstance(s, dict) and str(s.get("url", "")).startswith("http")][:5],
                at=datetime.now().strftime("%m-%d %H:%M"))


def run_ai(tabs, asof):
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return "button"
    from concurrent.futures import ThreadPoolExecutor
    todo = [e for tab in tabs for e in tab["stocks"][:3]]
    print(f"AI 분석 중… ({len(todo)}종목)", flush=True)

    def work(e):
        try:
            e["ai"] = ai_analyze(e, asof, key)
        except Exception as ex:
            print(f"  {e['name']} AI 분석 실패: {str(ex)[:150]}", flush=True)

    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(work, todo))
    print(f"  완료 {sum(1 for e in todo if 'ai' in e)}/{len(todo)}", flush=True)
    return "auto"


# ───────────────────────────── 결과 조립 ─────────────────────────────
def num(x, nd=0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, nd) if nd else int(round(x))


def safe(fn, t):
    try:
        return fn(t)
    except Exception:
        return "-"


def build(demo=False, mode="file"):
    print("데이터 준비 중…", flush=True)
    P, mcap, market, namefn = load_demo(CFG["days"]) if demo else load_naver(CFG["days"])
    I = indicators(P)
    ev = evaluate(P, I, market)
    m = ev["m"]
    c = m["c"]
    asof = P["c"].index[-1].strftime("%Y-%m-%d")

    codes = pd.Series(c.index, index=c.index)
    names = {t: str(namefn(t) or t) for t in c.index}
    hist = P["c"].iloc[-CFG["min_history"]:].notna().sum() >= CFG["min_history"]
    mask = (hist & (m["a"] >= CFG["min_amount"]) & (c >= CFG["min_price"]) & (P["v"].iloc[-1] > 0)
            & codes.map(lambda t: t in market and t.endswith("0") and "스팩" not in names[t])
            & (mcap.reindex(c.index).fillna(np.inf) >= CFG["min_mcap"]))
    uni = c.index[mask]

    # 매물대는 전 종목 계산 (매수 후보 목표가 + 보유 종목 점검에 사용)
    sl = slice(-CFG["chart_days"], None)
    H, L, V = (P[k].iloc[sl] for k in "hlv")
    vp = {t: vp_levels(H[t].to_numpy(float), L[t].to_numpy(float), V[t].to_numpy(float), float(c[t]))
          for t in c.index if c[t] == c[t]}

    # 1차 후보: 기본 점수 + 차트 보조 점수 상위
    pre = {}
    for k in (1, 2, 3):
        s = ev["scores"][f"s{k}"][uni] + sum(d[uni] for _, d, _ in ev["aux"][k])
        pre[k] = list(s.sort_values(ascending=False).head(CFG["cand_n"]).index)
    cands = sorted(set().union(*pre.values()))
    extras, xinfo = demo_extras(cands) if demo else fetch_extras(cands)

    def series(t, tg):
        arr = lambda s, nd=0: [num(x, nd) for x in s[t].iloc[sl]]
        return dict(
            d=[x.strftime("%Y-%m-%d") for x in P["c"].index[sl]],
            o=arr(P["o"]), h=arr(P["h"]), l=arr(P["l"]), c=arr(P["c"]), v=arr(P["v"]),
            bbu=arr(I["bbu"], 1), bbm=arr(I["ma20"], 1), bbl=arr(I["bbl"], 1),
            ma60=arr(I["ma60"], 1), vma=arr(I["vma20"]), rsi=arr(I["rsi"], 1), hist=arr(I["hist"], 2),
            t1=num(tg["t1"]), t2=num(tg["t2"]), stop=num(tg["stop"]),
        )

    tabs, picks_of = [], {}
    for k in (3, 1, 2):
        rows, loss = [], 0
        for t in pre[k]:
            x = extras.get(t, {})
            if x.get("eps") is not None and x["eps"] < 0:
                loss += 1
                continue
            tg = target_for(k, t, ev, vp.get(t, (None, None))[0])
            aux = [dict(label=lb, v=safe(fn, t), d=int(d[t])) for lb, d, fn in ev["aux"][k]]
            aux.append(dict(label="손익비", v=f"1 : {tg['rr']:.1f}" if np.isfinite(tg["rr"]) else "-", d=rr_delta(tg["rr"])))
            f, i = x.get("f"), x.get("i")
            aux.append(dict(label="수급 (5일)", v=(f"외국인 {man(f)} · 기관 {man(i)}" if f is not None else "정보 없음"),
                            d=flow_delta(f, i)))
            per = x.get("per")
            aux.append(dict(label="실적", v=(f"흑자 · PER {per:.1f}" if per and per == per else "흑자")
                            if x.get("eps") is not None else "확인 불가", d=0))
            base = float(ev["scores"].loc[t, f"s{k}"])
            adj = sum(a["d"] for a in aux)
            rows.append(dict(t=t, base=base, adj=adj, final=max(0.0, min(100.0, base + adj)), aux=aux, tg=tg, x=x))
        rows.sort(key=lambda r: (r["final"], float(m["a"][r["t"]])), reverse=True)

        stocks = []
        for rank, r in enumerate(rows[:CFG["top_n"]], 1):
            t, tg = r["t"], r["tg"]
            e = dict(
                rank=rank, code=t, name=names[t], market=market.get(t, ""), strat=k,
                close=num(c[t]), chg=num(m["chg"][t], 2), amount=num(m["a"][t]), mcap=num(mcap.get(t)),
                vr=num(m["vr"][t], 2), rsi=num(m["rsi"][t], 1), pb=num(m["pb"][t], 2),
                base=num(r["base"]), adj=int(r["adj"]), score=num(r["final"]),
                checks=[dict(label=lb, s=num(s_[t], 2), v=safe(fn, t)) for lb, _, s_, fn in ev["checks"][k]],
                aux=r["aux"],
                tg=dict(t1=num(tg["t1"]), t2=num(tg["t2"]), stop=num(tg["stop"]), rr=num(tg["rr"], 2),
                        t1p=num(tg["t1p"], 1), t2p=num(tg["t2p"], 1), sp=num(tg["sp"], 1),
                        note=tg["note"], stop_label=tg["stop_label"]),
            )
            e["prompt"] = ai_prompt(e, asof)
            if rank <= 6:
                e["series"] = series(t, tg)
            stocks.append(e)
            picks_of.setdefault(t, f"{STRATS[k]} {rank}위")
        tabs.append(dict(k=k, name=STRATS[k], stocks=stocks, loss=loss))

    uniq = list({e["code"]: e for tab in tabs for e in tab["stocks"]}.values())
    news = collect_news(uniq, demo)
    ai_mode = "button" if demo else run_ai(tabs, asof)

    # 보유 종목 점검용 전 종목 요약
    allstk = {}
    for t in c.index:
        if c[t] != c[t]:
            continue
        res, sup = vp.get(t, (None, None))
        allstk[t] = [names[t], market.get(t, ""), num(c[t]), num(m["chg"][t], 2), num(m["ma20"][t]), num(m["ma60"][t]),
                     (1 if m["ma60"][t] > m["ma60_5"][t] else -1), num(m["bbu"][t]), num(m["bbl"][t]),
                     num(m["rsi"][t], 1), num(m["atr"][t]), num(m["h0"][t], 2), num(m["h1"][t], 2),
                     num(m["hi20"][t]), num(m["lo20"][t]), num(m["hi60"][t]), num(res), num(sup),
                     picks_of.get(t, ""), num(m["rs"][t], 1)]

    above20 = (c[uni] > m["ma20"][uni]).mean() * 100
    above60 = (c[uni] > m["ma60"][uni]).mean() * 100
    rising = (m["chg"][uni] > 0).mean() * 100
    return dict(
        asof=asof, generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        mode=mode, demo=demo, universe=int(len(uni)), ai_mode=ai_mode, extras=xinfo,
        cfg=dict(min_amount=CFG["min_amount"], min_mcap=CFG["min_mcap"], min_price=CFG["min_price"]),
        rules={k: [[lb, w] for lb, w, _, _ in ev["checks"][k]] for k in (1, 2, 3)},
        targets={k: list(TARGET_RULE[k]) + [STOP_LABEL[k]] for k in (1, 2, 3)},
        breadth=dict(above20=num(above20, 1), above60=num(above60, 1), rising=num(rising, 1)),
        default_tab=(3 if above20 < 45 else 1),
        tabs=tabs, all=allstk, news=news,
        names={t: str(n) for t, n in getattr(namefn, "__self__", {}).items()},
        worker=os.environ.get("AI_WORKER_URL", "").strip(),
    )


def write_html(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", payload)
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
                    with open(out, "rb") as f:
                        body = f.read()
                except Exception as ex:
                    body = (f"<meta charset='utf-8'><h2>데이터를 불러오지 못했습니다</h2><pre>{ex}</pre>").encode("utf-8")
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
  --ok:#0f766e; --mid:#b7791f; --no:#b8c0c8; --bad:#c2410c;
  --s1:#0f766e; --s2:#6d28d9; --s3:#b45309; --ai:#0e7490; --aibg:rgba(14,116,144,.07);
  color-scheme:light;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0f1418; --surface:#171d22; --sunk:#1c2329; --line:#2a333b; --ink:#e7ecf0; --muted:#98a3ad; --faint:#5f6a74;
    --up:#ff5a68; --down:#5b95ff; --band:rgba(91,149,255,.08); --bb:#7f8ea5; --ma:#f59e0b; --rsi:#a78bfa;
    --vp:rgba(152,163,173,.16); --poc:rgba(245,158,11,.40);
    --ok:#2dd4bf; --mid:#fbbf24; --no:#46515b; --bad:#fb923c;
    --s1:#2dd4bf; --s2:#a78bfa; --s3:#f59e0b; --ai:#22d3ee; --aibg:rgba(34,211,238,.07);
    color-scheme:dark;
  }
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink)}
body{font-family:"Pretendard","Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",system-ui,sans-serif;font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
.tn{font-variant-numeric:tabular-nums}
.wrap{max-width:1240px;margin:0 auto;padding:28px 20px 64px}
.upc{color:var(--up)} .downc{color:var(--down)}
button{font:inherit}
:focus-visible{outline:3px solid var(--down);outline-offset:2px}

header{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap}
h1{font-size:32px;line-height:1.15;margin:0;font-weight:800;letter-spacing:-.025em}
.meta{color:var(--muted);font-size:13px;margin-top:6px}
.btn{font-weight:700;font-size:14px;color:var(--surface);background:var(--ink);border:0;border-radius:10px;padding:10px 16px;cursor:pointer}
.btn.ghost{background:transparent;color:var(--ink);border:1.5px solid var(--line)}
.demo{margin:16px 0 0;padding:10px 14px;border-radius:10px;background:var(--sunk);border:1px dashed var(--mid);color:var(--muted);font-size:13px}

/* 추천 방식 */
.how{margin:20px 0 0;background:var(--surface);border:1px solid var(--line);border-radius:14px}
.how>summary{cursor:pointer;padding:14px 18px;font-weight:800;font-size:15px;list-style:none;display:flex;justify-content:space-between;align-items:center;gap:12px}
.how>summary::-webkit-details-marker{display:none}
.how>summary span{font-weight:500;font-size:13px;color:var(--muted)}
.how>summary::after{content:"펼치기";font-size:12px;font-weight:600;color:var(--muted)}
.how[open]>summary::after{content:"접기"}
.how .body{padding:0 18px 18px;display:grid;gap:16px}
.kinds{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.kind{padding:12px 14px;border-radius:10px;background:var(--sunk);font-size:13px}
.kind b{font-size:14px;display:block;margin-bottom:2px}
.kind.ai{background:var(--aibg)}
.how h4{margin:0 0 6px;font-size:13.5px}
.how p{margin:0;font-size:13px;color:var(--muted)}
.rules{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
.rule{border:1px solid var(--line);border-radius:10px;padding:12px 14px;font-size:12.5px}
.rule h5{margin:0 0 6px;font-size:14px}
.rule ol{margin:0;padding-left:18px;display:grid;gap:2px}
.rule li span{color:var(--muted)}
.rule .tg{margin-top:8px;padding-top:8px;border-top:1px dashed var(--line);color:var(--muted)}
.auxrules{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px 18px;font-size:12.5px;color:var(--muted)}
.auxrules b{color:var(--ink)}
@media(max-width:860px){.kinds,.rules,.auxrules{grid-template-columns:1fr}}

.pulse{display:grid;grid-template-columns:repeat(3,minmax(0,1fr)) minmax(0,1.6fr);gap:1px;margin:16px 0 0;background:var(--line);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.pulse>div{background:var(--surface);padding:12px 16px}
.pulse dt{font-size:12px;color:var(--muted)}
.pulse dd{margin:2px 0 0;font-size:22px;font-weight:800;letter-spacing:-.02em}
.pulse .gauge{height:4px;border-radius:2px;background:var(--sunk);margin-top:6px;overflow:hidden}
.pulse .gauge i{display:block;height:100%;background:var(--ink)}
.pulse .regime dd{font-size:15px;font-weight:700;letter-spacing:0}
.pulse .regime p{margin:2px 0 0;font-size:12.5px;color:var(--muted)}
@media(max-width:820px){.pulse{grid-template-columns:repeat(3,1fr)}.pulse .regime{grid-column:1/-1}}

/* 탭 */
.tabs{display:flex;gap:4px;margin:32px 0 0;border-bottom:1px solid var(--line);overflow-x:auto}
.tab{font-weight:800;font-size:16px;padding:12px 16px 10px;border:0;background:none;color:var(--muted);border-bottom:3px solid transparent;cursor:pointer;margin-bottom:-1px;white-space:nowrap;display:flex;gap:8px;align-items:center}
.tab[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--tc)}
.tab .rec{font-size:11px;font-weight:700;padding:1px 7px;border-radius:999px;background:var(--tc);color:var(--surface)}
.tabdesc{margin:12px 0 0;color:var(--muted);font-size:13.5px}

h2{font-size:19px;font-weight:800;margin:32px 0 4px;letter-spacing:-.015em}
h3.sec{font-size:16px;font-weight:800;margin:26px 0 10px}
.sub{color:var(--muted);font-size:13px;margin:0 0 14px}

.tag{display:inline-block;font-size:12px;font-weight:700;padding:2px 9px;border-radius:999px;border:1.5px solid currentColor;white-space:nowrap}
.s1{color:var(--s1)} .s2{color:var(--s2)} .s3{color:var(--s3)}
.lbl{display:inline-flex;gap:6px;align-items:center;font-size:12px;font-weight:800;color:var(--muted);margin:16px 0 6px}

.pick{display:grid;grid-template-columns:320px minmax(0,1fr);background:var(--surface);border:1px solid var(--line);border-radius:16px;overflow:hidden;margin-bottom:16px}
.pick .info{padding:20px;background:var(--sunk);border-right:1px solid var(--line);position:relative}
.pick .big{position:absolute;right:16px;top:6px;font-size:84px;font-weight:900;line-height:1;letter-spacing:-.06em;color:var(--line)}
.pick .right{padding:12px 8px 14px 8px;min-width:0;display:flex;flex-direction:column;gap:10px}
@media(max-width:880px){.pick{grid-template-columns:1fr}.pick .info{border-right:0;border-bottom:1px solid var(--line)}}
.name{font-size:22px;font-weight:800;letter-spacing:-.02em;margin:10px 0 0;position:relative}
.code{font-size:12.5px;color:var(--muted);font-weight:500}
.price{font-size:26px;font-weight:800;letter-spacing:-.02em;margin:6px 0 0}
.price span{font-size:15px;font-weight:700;margin-left:6px}
.score{display:flex;align-items:center;gap:10px;margin:12px 0 2px}
.score .bar{flex:1;height:8px;background:var(--line);border-radius:4px;overflow:hidden}
.score .bar i{display:block;height:100%;border-radius:4px;background:currentColor}
.score b{font-size:20px;font-weight:800;color:var(--ink)}
.scorebrk{font-size:12px;color:var(--muted)}
.checks,.aux{list-style:none;padding:0;margin:0;display:grid;gap:5px}
.checks li,.aux li{display:grid;grid-template-columns:14px 1fr auto;gap:8px;align-items:baseline;font-size:13px}
.aux li{grid-template-columns:1fr auto 34px}
.checks .ic{font-size:11px}
.checks .ok .ic{color:var(--ok)} .checks .mid .ic{color:var(--mid)} .checks .no .ic{color:var(--no)}
.checks .no .lb{color:var(--muted)}
.vl{font-size:12px;color:var(--muted);text-align:right}
.aux .d{font-size:12px;font-weight:800;text-align:right}
.pos{color:var(--ok)} .neg{color:var(--bad)} .zero{color:var(--faint)}

.tgbox{margin-top:14px;display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.tgbox>div{background:var(--surface);padding:8px 10px}
.tgbox dt{font-size:11.5px;color:var(--muted)}
.tgbox dd{margin:0;font-weight:800;font-size:14px}
.tgbox dd small{display:block;font-weight:600;font-size:11.5px}
.tgbox .rr{grid-column:1/-1;display:flex;justify-content:space-between;align-items:center;font-size:12.5px}
.tgbox .rr b{font-size:14px}
.kv{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0 0;padding-top:12px;border-top:1px dashed var(--line)}
.kv dt{font-size:11.5px;color:var(--muted)} .kv dd{margin:0;font-weight:700;font-size:13.5px}

/* AI */
.aibox{border:1px solid var(--line);border-left:4px solid var(--ai);border-radius:12px;padding:14px 16px;background:var(--aibg)}
.aibox .hd{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
.aibox .hd b{font-size:13.5px}
.aibox .hd small{color:var(--muted);font-size:12px}
.verdict{font-weight:800;font-size:13px;padding:3px 10px;border-radius:999px;color:var(--surface)}
.v-ok{background:var(--ok)} .v-mid{background:var(--mid)} .v-bad{background:var(--bad)}
.spin{width:16px;height:16px;border-radius:50%;border:2.5px solid var(--line);border-top-color:var(--ai);animation:sp 0.9s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
@media (prefers-reduced-motion: reduce){.spin{animation-duration:3s}}
.aibox p{margin:8px 0 0;font-size:13.5px}
.aibox ul{margin:6px 0 0;padding-left:18px;font-size:13px;display:grid;gap:2px}
.aibox .risk{color:var(--muted)}
.news{border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-top:10px}
.right .news{margin-top:0}
.news ul{list-style:none;margin:0;padding:0;display:grid;gap:5px;font-size:13px}
.news li{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.news a{color:var(--ink);text-decoration:none;flex:1 1 200px}
.news a:hover{text-decoration:underline}
.news .dt{color:var(--muted);font-size:12px;min-width:38px}
.acts2{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
a.btn{text-decoration:none;display:inline-block}
.linkbtn{background:none;border:0;color:var(--muted);text-decoration:underline;cursor:pointer;font-size:12.5px;padding:4px}
.aibox ul.issues{list-style:none;padding:0}
.aibox ul.issues li{display:flex;gap:8px;align-items:baseline}
.aibox .dt{color:var(--muted);font-size:12px;min-width:38px}
.en{font-size:10.5px;font-weight:800;padding:0 5px;border-radius:4px;background:var(--ai);color:#fff;margin-right:4px;white-space:nowrap}
.aibox .src{margin-top:8px;display:flex;flex-wrap:wrap;gap:6px 12px;font-size:12px}
.aibox .src a{color:var(--ai)}
.agree{font-size:12px;font-weight:700;margin-top:8px}
.aibtn{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
.aibtn p{margin:0;font-size:12.5px;color:var(--muted)}
.btn.ai{background:var(--ai);color:#fff}

.watch{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
@media(max-width:920px){.watch{grid-template-columns:1fr}}
.w{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px;display:flex;flex-direction:column;gap:2px}
.w .top{display:flex;justify-content:space-between;align-items:center}
.w .rk{font-weight:800;font-size:13px;color:var(--muted)}
.w .name{font-size:18px;margin-top:6px}
.w .price{font-size:20px}
.w .mini{margin:8px -4px 0}
.w .tgbox{margin-top:10px}
.w .btn.ai{margin-top:12px;padding:8px 12px;font-size:13px}

.tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:14px}
table{border-collapse:collapse;width:100%;min-width:860px;font-size:13.5px}
th{font-size:12px;color:var(--muted);font-weight:600;text-align:right;padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
tr:last-child td{border-bottom:0}
th:nth-child(-n+2),td:nth-child(-n+2){text-align:left}
td.nm b{font-weight:700} td.nm span{color:var(--muted);font-size:12px;margin-left:6px}
td.rk{color:var(--muted);font-weight:700;width:44px}
.empty{padding:20px;border:1px dashed var(--line);border-radius:12px;color:var(--muted);font-size:13.5px}

/* 내 종목 점검 */
.mine{margin-top:44px;padding-top:8px;border-top:2px solid var(--ink)}
.form{display:grid;grid-template-columns:minmax(0,2fr) minmax(0,1fr) minmax(0,1fr) auto;gap:10px;align-items:end;background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px}
.form label{display:grid;gap:4px;font-size:12.5px;color:var(--muted);font-weight:600}
.form input{font:inherit;font-size:15px;padding:10px 12px;border:1.5px solid var(--line);border-radius:10px;background:var(--sunk);color:var(--ink);width:100%}
.form .err{grid-column:1/-1;color:var(--bad);font-size:13px;margin:0}
@media(max-width:760px){.form{grid-template-columns:1fr 1fr}.form label:first-child{grid-column:1/-1}.form .btn{grid-column:1/-1}}
.hold{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:18px 20px;margin-top:14px;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.2fr);gap:20px}
@media(max-width:880px){.hold{grid-template-columns:1fr}}
.hold .hdr{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}
.hold .rm{background:none;border:0;color:var(--muted);cursor:pointer;font-size:12.5px;text-decoration:underline}
.pos-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px 14px;margin-top:12px}
.pos-grid dt{font-size:11.5px;color:var(--muted)} .pos-grid dd{margin:0;font-weight:800;font-size:15px}
.hv{margin-top:14px;padding:12px 14px;border-radius:12px;background:var(--sunk)}
.hv .lab{font-size:12px;color:var(--muted);font-weight:700}
.hv .val{font-size:20px;font-weight:900;letter-spacing:-.02em;margin-top:2px}
.hv .val.ok{color:var(--ok)} .hv .val.mid{color:var(--mid)} .hv .val.no{color:var(--bad)}
.hold ul.why{margin:10px 0 0;padding-left:18px;font-size:13px;display:grid;gap:3px}
.lvtable{width:100%;min-width:0;font-size:13px;margin-top:8px}
.lvtable td,.lvtable th{padding:7px 8px}
.plan{margin:10px 0 0;padding:10px 12px;border-radius:10px;border:1px dashed var(--line);font-size:13px}
.plan b{display:block;margin-bottom:2px}

footer{margin-top:44px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:12.5px}
footer p{margin:0 0 6px}

dialog{border:0;border-radius:16px;padding:0;max-width:560px;width:calc(100% - 32px);background:var(--surface);color:var(--ink)}
dialog::backdrop{background:rgba(0,0,0,.45)}
dialog .in{padding:20px}
dialog h4{margin:0 0 6px;font-size:16px}
dialog ol{margin:6px 0 12px;padding-left:20px;font-size:13.5px;color:var(--muted)}
dialog textarea{width:100%;height:180px;font:inherit;font-size:12.5px;padding:10px;border-radius:10px;border:1px solid var(--line);background:var(--sunk);color:var(--ink);resize:vertical}
dialog .acts{display:flex;gap:8px;justify-content:flex-end;margin-top:12px;flex-wrap:wrap}
dialog a.btn{text-decoration:none;display:inline-block}

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
.chart .tl{stroke:var(--up);stroke-width:1.2;stroke-dasharray:6 4}
.chart .sl{stroke:var(--down);stroke-width:1.2;stroke-dasharray:6 4}
.chart .tlx{fill:var(--up);font-size:11px;font-weight:700}
.chart .slx{fill:var(--down);font-size:11px;font-weight:700}
.lvbar .ax{stroke:var(--line);stroke-width:2}
.lvbar text{font-size:11px;font-weight:700}
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

  <details class="how" id="how" open>
    <summary>이 페이지가 종목을 고르는 방식 <span>처음 보시면 꼭 읽어주세요</span></summary>
    <div class="body" id="howbody"></div>
  </details>

  <dl class="pulse" id="pulse"></dl>

  <div class="tabs" role="tablist" id="tabs"></div>
  <p class="tabdesc" id="tabdesc"></p>
  <div id="panel" role="tabpanel"></div>

  <section class="mine" id="mine">
    <h2>내 종목 점검</h2>
    <p class="sub">보유 중인 종목과 수량, 평균 매입가를 넣으면 차트 기준으로 보유·매도 전략과 손절가·목표가를 알려드립니다. 입력한 내용은 이 기기(브라우저)에만 저장되고 다른 사람에게 보이지 않습니다.</p>
    <div class="form">
      <label>종목명 또는 코드<input id="fStock" list="stocklist" placeholder="예: 삼성전자 또는 005930" autocomplete="off"></label>
      <label>보유 수량 (주)<input id="fQty" type="number" inputmode="numeric" min="1" placeholder="예: 100"></label>
      <label>평균 매입가 (원)<input id="fAvg" type="number" inputmode="numeric" min="1" placeholder="예: 71000"></label>
      <button class="btn" id="fAdd">점검하고 저장</button>
      <p class="err" id="fErr" hidden></p>
    </div>
    <datalist id="stocklist"></datalist>
    <div id="holds"></div>
  </section>

  <footer>
    <p>📊 차트 분석은 전 거래일 종가까지의 시세로 계산한 공식 결과이고, 🤖 AI 분석은 Claude가 뉴스·공시를 검색해 낸 의견입니다. 둘 다 참고 자료이며 목표가 도달이나 수익을 보장하지 않습니다.</p>
    <p>관리종목·투자경고 여부와 최신 공시는 매수 전 HTS에서 한 번 더 확인하세요. 투자 판단과 책임은 본인에게 있습니다.</p>
  </footer>
</div>

<dialog id="dlg"><div class="in">
  <h4>🤖 AI 분석 요청</h4>
  <ol><li>아래 문장을 <b>복사</b>합니다.</li><li><b>Claude 열기</b>를 눌러 새 대화에 붙여넣고 보냅니다.</li></ol>
  <textarea id="dlgText" readonly></textarea>
  <div class="acts">
    <button class="btn ghost" id="dlgClose">닫기</button>
    <button class="btn" id="dlgCopy">복사하기</button>
    <a class="btn ai" href="https://claude.ai/new" target="_blank" rel="noopener">Claude 열기</a>
  </div>
</div></dialog>

<script>
const D = __DATA__;
const STRAT = {1:{n:'추세 눌림목',c:'s1',v:'--s1',d:'상승 추세가 살아 있는 종목이 20일선까지 쉬어 내려온 자리입니다. 가장 무난한 전략입니다.'},
               2:{n:'거래량 돌파',c:'s2',v:'--s2',d:'좁은 박스권에서 거래량과 함께 위로 뚫고 나온 종목입니다. 수익이 크지만 실패도 빠른 공격형 전략입니다.'},
               3:{n:'과매도 반등',c:'s3',v:'--s3',d:'볼린저 하단을 찍고 거래량과 함께 반등을 시작한 종목입니다. 처음 정하신 조건에 가장 가까운 전략입니다.'}};
const NS='http://www.w3.org/2000/svg';
const f0=x=>x==null||isNaN(x)?'-':Math.round(x).toLocaleString('ko-KR');
const pctS=(x,d=1)=>x==null?'-':(x>0?'+':'')+x.toFixed(d)+'%';
const pct=(x,d=2)=>x==null?'-':`<span class="${x>0?'upc':x<0?'downc':''}">${pctS(x,d)}</span>`;
const won=x=>x==null?'-':x>=1e12?(x/1e12).toFixed(1)+'조':f0(x/1e8)+'억';
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const ckIcon=s=>s>=.99?['ok','●']:s>=.4?['mid','◐']:['no','○'];
const dcls=d=>d>0?'pos':d<0?'neg':'zero';
const dtxt=d=>d>0?'+'+d:d<0?String(d):'0';

function el(tag,a,p){const e=document.createElementNS(NS,tag);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e;}
function tx(p,x,y,s,cls,anchor){const t=el('text',{x,y,class:cls||'',...(anchor?{'text-anchor':anchor}:{})},p);t.textContent=s;return t;}
function path(p,arr,X,Y,cls){let d='',pen=false;arr.forEach((v,i)=>{if(v==null){pen=false;return;}d+=(pen?'L':'M')+X(i).toFixed(1)+','+Y(v).toFixed(1);pen=true;});return el('path',{d,class:cls},p);}
function ticks(lo,hi,n){const raw=(hi-lo)/n,mag=10**Math.floor(Math.log10(raw)),f=raw/mag;const st=(f<1.5?1:f<3?2:f<7?5:10)*mag,out=[];for(let v=Math.ceil(lo/st)*st;v<=hi;v+=st)out.push(v);return out;}
function legend(t,parts){t.textContent='';parts.forEach(([s,c])=>{const sp=document.createElementNS(NS,'tspan');if(c)sp.setAttribute('class',c);sp.textContent=s;t.appendChild(sp);});}

function drawChart(box,s){
  box.innerHTML='';
  const W=Math.max(300,box.clientWidth),small=W<560;
  const m={l:4,r:66,t:24,b:20},hP=small?240:300,hV=56,hM=50,hR=70,g=14;
  const H=m.t+hP+g+hV+g+hM+g+hR+m.b;
  const svg=el('svg',{width:W,height:H,viewBox:`0 0 ${W} ${H}`,class:'chart',role:'img','aria-label':'가격 차트'},box);
  const n=s.c.length,pw=W-m.l-m.r,step=pw/n,cw=Math.max(1,Math.min(9,step*.64));
  const X=i=>m.l+step*(i+.5);
  const pT=m.t,pB=pT+hP,vT=pB+g,vB=vT+hV,mT=vB+g,mB=mT+hM,rT=mB+g,rB=rT+hR;
  let lo=Infinity,hi=-Infinity;
  for(let i=0;i<n;i++){lo=Math.min(lo,s.l[i],s.bbl[i]??Infinity);hi=Math.max(hi,s.h[i],s.bbu[i]??-Infinity);}
  if(s.stop)lo=Math.min(lo,s.stop); if(s.t2)hi=Math.max(hi,s.t2);
  const pad=(hi-lo)*.05;lo-=pad;hi+=pad;
  const YP=p=>pB-(p-lo)/(hi-lo)*hP;
  for(const t of ticks(lo,hi,small?4:5)){const y=YP(t);el('line',{x1:m.l,x2:m.l+pw,y1:y,y2:y,class:'grid'},svg);tx(svg,m.l+pw+6,y+4,f0(t),'axis');}
  // 매물대
  const bins=28,bh=(hi-lo)/bins,vp=new Array(bins).fill(0);
  for(let i=0;i<n;i++){const a=Math.max(0,Math.floor((s.l[i]-lo)/bh)),b=Math.min(bins-1,Math.floor((s.h[i]-lo)/bh)),k=b-a+1;for(let j=a;j<=b;j++)vp[j]+=s.v[i]/k;}
  const vmx=Math.max(...vp),poc=vp.indexOf(vmx),vpw=pw*.24;
  vp.forEach((x,j)=>{const w=x/vmx*vpw,y1=YP(lo+(j+1)*bh),y2=YP(lo+j*bh);el('rect',{x:m.l+pw-w,y:y1+.5,width:w,height:Math.max(.5,y2-y1-1),class:j===poc?'poc':'vp'},svg);});
  if(!small)tx(svg,m.l+pw-vpw-4,YP(lo+(poc+.5)*bh)+4,'매물대 최대','poctx','end');
  // 볼린저 · 60일선
  const up=[],dn=[];
  s.bbu.forEach((v,i)=>{if(v!=null&&s.bbl[i]!=null){up.push(X(i).toFixed(1)+','+YP(v).toFixed(1));dn.unshift(X(i).toFixed(1)+','+YP(s.bbl[i]).toFixed(1));}});
  if(up.length)el('path',{d:'M'+up.join('L')+'L'+dn.join('L')+'Z',class:'band'},svg);
  path(svg,s.bbu,X,YP,'bb');path(svg,s.bbl,X,YP,'bb');path(svg,s.bbm,X,YP,'bbm');path(svg,s.ma60,X,YP,'ma');
  // 목표가 · 손절가
  let prevY=null;
  [['t2','목표2','tl'],['t1','목표1','tl'],['stop','손절','sl']].forEach(([k,lab,cls])=>{
    if(!s[k])return;const y=YP(s[k]);
    el('line',{x1:m.l,x2:m.l+pw,y1:y,y2:y,class:cls},svg);
    const ty=(prevY!=null&&y-4-prevY<12)?y+13:y-4; prevY=ty;
    tx(svg,m.l+4,ty,`${lab} ${f0(s[k])}`,cls+'x');
  });
  // 캔들
  for(let i=0;i<n;i++){const cls=s.c[i]>=s.o[i]?'up':'down',x=X(i);
    el('line',{x1:x,x2:x,y1:YP(s.h[i]),y2:YP(s.l[i]),class:cls,'stroke-width':1},svg);
    const y1=YP(Math.max(s.o[i],s.c[i])),y2=YP(Math.min(s.o[i],s.c[i]));
    el('rect',{x:x-cw/2,y:y1,width:cw,height:Math.max(1,y2-y1),class:cls,'stroke-width':0},svg);}
  const ly=YP(s.c[n-1]);el('rect',{x:m.l+pw+1,y:ly-9,width:m.r-2,height:18,rx:4,class:'tagbg'},svg);tx(svg,m.l+pw+6,ly+4,f0(s.c[n-1]),'tagtx');
  // 거래량
  const vMax=Math.max(...s.v)*1.08,YV=v=>vB-v/vMax*hV;
  el('line',{x1:m.l,x2:m.l+pw,y1:vB,y2:vB,class:'grid'},svg);
  for(let i=0;i<n;i++)el('rect',{x:X(i)-cw/2,y:YV(s.v[i]),width:cw,height:vB-YV(s.v[i]),class:s.c[i]>=s.o[i]?'up':'down','stroke-width':0,opacity:.75},svg);
  path(svg,s.vma,X,YV,'vma');
  // MACD 히스토그램
  const hm=Math.max(...s.hist.map(x=>Math.abs(x??0)))||1,mid=(mT+mB)/2,YM=v=>mid-v/hm*(hM/2);
  el('line',{x1:m.l,x2:m.l+pw,y1:mid,y2:mid,class:'grid'},svg);
  s.hist.forEach((v,i)=>{if(v==null)return;const y=YM(v);el('rect',{x:X(i)-cw/2,y:Math.min(y,mid),width:cw,height:Math.max(.5,Math.abs(y-mid)),class:v>=0?'up':'down','stroke-width':0,opacity:(i>0&&s.hist[i-1]!=null&&Math.abs(v)<Math.abs(s.hist[i-1]))?.45:.85},svg);});
  // RSI
  const YR=r=>rB-r/100*hR;
  el('rect',{x:m.l,y:YR(30),width:pw,height:YR(0)-YR(30),class:'zone'},svg);
  [30,70].forEach(v=>{el('line',{x1:m.l,x2:m.l+pw,y1:YR(v),y2:YR(v),class:'lvl'},svg);tx(svg,m.l+pw+6,YR(v)+4,v,'axis');});
  el('line',{x1:m.l,x2:m.l+pw,y1:rB,y2:rB,class:'grid'},svg);
  path(svg,s.rsi,X,YR,'rsi');
  const every=Math.ceil(n/(small?4:6));
  for(let i=every;i<n;i+=every)tx(svg,X(i),H-4,s.d[i].slice(5).replace('-','.'),'axis','middle');
  const L1=tx(svg,m.l+2,14,'','lg'),L2=tx(svg,m.l+2,vT+11,'','lg'),L3=tx(svg,m.l+2,mT+11,'','lg'),L4=tx(svg,m.l+2,rT+11,'','lg');
  function show(i){
    const prev=i>0?s.c[i-1]:s.o[i],ch=(s.c[i]/prev-1)*100,chp=[pctS(ch,2),ch>=0?'k up':'k down'];
    legend(L1,small?[[s.d[i].slice(5).replace('-','.')+'  ','k'],['종 ',''],[f0(s.c[i])+' ','k'],chp]:
      [[s.d[i]+'  ','k'],['시 ',''],[f0(s.o[i])+'  ','k'],['고 ',''],[f0(s.h[i])+'  ','k'],['저 ',''],[f0(s.l[i])+'  ','k'],['종 ',''],[f0(s.c[i])+' ','k'],chp,['   BB ',''],[`${f0(s.bbu[i])} / ${f0(s.bbl[i])}`,'k']]);
    const vr=s.vma[i]?s.v[i]/s.vma[i]:null;
    legend(L2,[['거래량 ',''],[f0(s.v[i]),'k'],...(vr?[[`  (평균의 ${vr.toFixed(1)}배)`,'']]:[])]);
    legend(L3,[['MACD 히스토그램 ',''],[s.hist[i]==null?'-':s.hist[i].toFixed(1),'k']]);
    legend(L4,[['RSI(14) ',''],[s.rsi[i]==null?'-':s.rsi[i].toFixed(1),'k']]);
  }
  show(n-1);
  const vl=el('line',{y1:pT,y2:rB,class:'cross',visibility:'hidden'},svg);
  const ov=el('rect',{x:m.l,y:pT,width:pw,height:rB-pT,fill:'transparent'},svg);
  ov.addEventListener('pointermove',e=>{const r=svg.getBoundingClientRect(),x=(e.clientX-r.left)*W/r.width;const i=Math.max(0,Math.min(n-1,Math.floor((x-m.l)/step)));vl.setAttribute('x1',X(i));vl.setAttribute('x2',X(i));vl.setAttribute('visibility','visible');show(i);});
  ov.addEventListener('pointerleave',()=>{vl.setAttribute('visibility','hidden');show(n-1);});
}

function drawMini(box,s){
  box.innerHTML='';
  const k=60,W=Math.max(240,box.clientWidth),H=96,m=6;
  const c=s.c.slice(-k),u=s.bbu.slice(-k),b=s.bbl.slice(-k),n=c.length;
  const lo=Math.min(...c,...b.filter(x=>x!=null),s.stop||Infinity),hi=Math.max(...c,...u.filter(x=>x!=null),s.t1||-Infinity);
  const X=i=>m+(W-2*m)*i/(n-1),Y=v=>H-m-(v-lo)/(hi-lo)*(H-2*m);
  const svg=el('svg',{width:W,height:H,viewBox:`0 0 ${W} ${H}`,class:'chart',role:'img','aria-label':'최근 60일 종가와 볼린저밴드'},box);
  const up=[],dn=[];u.forEach((v,i)=>{if(v!=null&&b[i]!=null){up.push(X(i)+','+Y(v));dn.unshift(X(i)+','+Y(b[i]));}});
  if(up.length)el('path',{d:'M'+up.join('L')+'L'+dn.join('L')+'Z',class:'band'},svg);
  path(svg,u,X,Y,'bb');path(svg,b,X,Y,'bb');
  if(s.t1)el('line',{x1:m,x2:W-m,y1:Y(s.t1),y2:Y(s.t1),class:'tl'},svg);
  if(s.stop)el('line',{x1:m,x2:W-m,y1:Y(s.stop),y2:Y(s.stop),class:'sl'},svg);
  path(svg,c,X,Y,'cl');
  el('circle',{cx:X(n-1),cy:Y(c[n-1]),r:3.5,class:c[n-1]>=c[n-2]?'up':'down'},svg);
}

/* ───── AI 요청 대화상자 ───── */
const dlg=document.getElementById('dlg');
function askAI(text){document.getElementById('dlgText').value=text;dlg.showModal();}
document.getElementById('dlgClose').onclick=()=>dlg.close();
document.getElementById('dlgCopy').onclick=async e=>{const t=document.getElementById('dlgText');try{await navigator.clipboard.writeText(t.value);}catch(_){t.select();document.execCommand('copy');}e.target.textContent='복사됨';setTimeout(()=>e.target.textContent='복사하기',1500);};

/* ───── AI 분석 (중계 서버가 있으면 바로 표시, 없으면 복사 방식) ───── */
const AI_URL=(D.worker||'').replace(/\/$/,'');
const PASSKEY='family-pass';
const JSON_PICK='\n\n답변은 다른 말 없이 아래 JSON 하나로만 출력해줘:\n{"verdict":"매수|관망|주의","summary":"한 문장 결론","issues":[{"date":"MM/DD","text":"한국어 한 줄 요약","lang":"ko 또는 en"}],"reasons":["근거1","근거2","근거3"],"risks":["리스크1"],"sources":[{"title":"기사 제목","url":"https://...","lang":"ko 또는 en"}]}';
const JSON_HOLD=JSON_PICK.replace('매수|관망|주의','보유|일부 매도|전량 매도');
const TONE={'매수':'ok','보유':'ok','관망':'mid','일부 매도':'mid','주의':'bad','전량 매도':'bad'};
function getPass(force){let p=null;try{p=localStorage.getItem(PASSKEY);}catch(_){}
  if(!p||force){p=window.prompt(force?'비밀번호가 맞지 않습니다. 가족 비밀번호를 다시 입력하세요.':'가족 비밀번호를 입력하세요. (한 번 입력하면 이 기기에 저장됩니다)');if(!p)return null;try{localStorage.setItem(PASSKEY,p);}catch(_){}}
  return p;}
async function relay(path,opts={},retry=true){
  const pass=getPass();if(!pass)throw new Error('비밀번호를 입력해야 분석할 수 있습니다.');
  const r=await fetch(AI_URL+path,{...opts,headers:{'content-type':'application/json','x-family-key':pass}});
  if(r.status===401&&retry){try{localStorage.removeItem(PASSKEY);}catch(_){}if(!getPass(true))throw new Error('비밀번호가 필요합니다.');return relay(path,opts,false);}
  const j=await r.json().catch(()=>({error:'중계 서버 응답을 읽지 못했습니다.'}));
  if(!r.ok)throw new Error(j.error||('오류 '+r.status));
  return j;
}
const loadAI=k=>{try{return JSON.parse(localStorage.getItem(k));}catch(_){return null;}};
const saveAI=(k,v)=>{try{localStorage.setItem(k,JSON.stringify(v));}catch(_){}};
try{Object.keys(localStorage).forEach(k=>{if(k.startsWith('ai:')&&!k.startsWith('ai:'+D.asof))localStorage.removeItem(k);});}catch(_){}
function parseAI(text){
  const m=String(text||'').match(/\{[\s\S]*\}/);if(!m)throw new Error('AI 답변 형식을 읽지 못했습니다. 다시 시도해 주세요.');
  const d=JSON.parse(m[0]),arr=x=>Array.isArray(x)?x:[];
  return {verdict:String(d.verdict||'관망'),summary:String(d.summary||''),
    issues:arr(d.issues).slice(0,6).map(i=>({date:String(i.date||''),text:String(i.text||''),lang:i.lang==='en'?'en':'ko'})),
    reasons:arr(d.reasons).slice(0,3).map(String),risks:arr(d.risks).slice(0,2).map(String),
    sources:arr(d.sources).filter(x=>x&&/^https?:/.test(x.url||'')).slice(0,5).map(x=>({title:String(x.title||'기사'),url:x.url,lang:x.lang==='en'?'en':'ko'})),
    at:new Date().toLocaleString('ko-KR',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'})};
}
const NEWS=D.news||{};
function newsHTML(code,max=6){
  const n=(NEWS[code]||[]).slice(0,max);
  if(!n.length)return '';
  return `<div class="news"><div class="lbl" style="margin:0 0 4px">📰 최근 한 달 뉴스 <span class="vl">(제목만 모은 것 · AI 판단 아님)</span></div><ul>${n.map(x=>
    `<li><span class="dt tn">${esc(x.d)}</span>${x.en?'<span class="en">해외</span>':''}<a href="${esc(x.u)}" target="_blank" rel="noopener">${esc(x.t)}</a>${x.s?` <span class="vl">${esc(x.s)}</span>`:''}</li>`).join('')}</ul></div>`;
}
// 요청 문장을 Claude 주소에 실어 보냄 (주소가 너무 길면 뉴스·근거를 줄임)
function claudeURL(prompt,code){
  const lines=(NEWS[code]||[]).slice(0,6).map(x=>`- ${x.d} ${x.en?'[해외] ':''}${x.t.slice(0,70)}`);
  const build=()=>prompt+(lines.length?`\n\n[참고: 오늘 아침 모은 뉴스 제목]\n${lines.join('\n')}`:'');
  let p=build();
  while(encodeURIComponent(p).length>7000&&lines.length){lines.pop();p=build();}
  while(encodeURIComponent(p).length>7000)p=p.slice(0,Math.floor(p.length*.9));
  return 'https://claude.ai/new?q='+encodeURIComponent(p);
}
const SLOTS=[];
function aiSlot(o){SLOTS.push(o);const i=SLOTS.length-1;return `<div class="aislot" data-slot="${i}">${slotInner(i)}</div>`;}
function slotInner(i){
  const s=SLOTS[i],a=s.pre||loadAI(s.key);
  if(a)return aiResultHTML(a,s,i);
  if(AI_URL)return `<div class="aibox aibtn"><p><b style="color:var(--ink)">🤖 AI 분석</b><br>버튼을 누르면 Claude가 최근 한 달의 국내·해외(영문) 기사와 공시를 검색해서 이 자리에 바로 분석 결과를 보여드립니다. 30초~1분 정도 걸립니다.</p><button class="btn ai" data-run="${i}">🤖 AI 분석 요청</button></div>`;
  return `<div class="aibox aibtn"><p><b style="color:var(--ink)">🤖 AI 분석</b><br>누르면 Claude가 <b>분석 요청이 입력된 채로</b> 열립니다. <b>전송(Enter)만</b> 누르세요. 최근 한 달 국내·해외 기사를 검색해서 분석해 드립니다.</p>
    <span class="acts2"><a class="btn ai" href="${claudeURL(s.prompt,s.code)}" target="_blank" rel="noopener">🤖 Claude에서 분석하기</a><button class="linkbtn" data-p="${i}">요청 문장 복사</button></span></div>`;
}
function aiResultHTML(a,s,i){
  const tone=TONE[a.verdict]||'mid';
  let agree;
  if(s.kind==='hold')agree=tone===s.tone?'✅ 차트 점검과 AI 의견이 같은 방향입니다.':'⚠️ 차트 점검과 AI 의견이 다릅니다. 아래 근거를 꼭 읽어보세요.';
  else agree=tone==='ok'?(s.score>=70?'✅ 차트와 AI 의견이 같은 방향입니다.':'차트 점수는 보통인데 AI는 긍정적입니다.'):
    tone==='bad'?'⚠️ 차트는 좋지만 AI가 뉴스상 위험 요인을 찾았습니다. 매수 전 꼭 확인하세요.':'AI는 뉴스상 뚜렷한 방향이 없다고 봤습니다.';
  return `<div class="aibox"><div class="hd"><b>🤖 AI 분석 결론</b><span class="verdict v-${tone}">${esc(a.verdict)}</span></div>
    <p>${esc(a.summary)}</p>
    ${(a.issues||[]).length?`<div class="lbl" style="margin:10px 0 2px">최근 한 달 이슈</div><ul class="issues">${a.issues.map(x=>`<li><span class="dt tn">${esc(x.date)}</span>${x.lang==='en'?'<span class="en">해외</span>':''}${esc(x.text)}</li>`).join('')}</ul>`:''}
    <div class="lbl" style="margin:10px 0 2px">판단 근거</div>
    <ul>${(a.reasons||[]).map(r=>`<li>${esc(r)}</li>`).join('')}${(a.risks||[]).map(r=>`<li class="risk">리스크: ${esc(r)}</li>`).join('')}</ul>
    <div class="src">${(a.sources||[]).map(x=>`<a href="${esc(x.url)}" target="_blank" rel="noopener">${x.lang==='en'?'<span class="en">EN</span>':''}${esc(x.title)}</a>`).join('')}</div>
    <div class="agree">${agree}</div>
    <div class="hd" style="margin-top:8px"><small>${esc(a.at)} 분석 · 뉴스 검색 기반 의견</small>
      <span style="display:flex;gap:6px">${AI_URL?`<button class="btn ghost" data-run="${i}" data-force="1">다시 분석</button>`:''}<a class="btn ghost" href="${claudeURL(s.prompt,s.code)}" target="_blank" rel="noopener">Claude에서 더 묻기</a></span></div></div>`;
}
async function runAI(i,force){
  const s=SLOTS[i],slot=document.querySelector(`[data-slot="${i}"]`);if(!slot)return;
  slot.innerHTML='<div class="aibox"><div class="hd"><b>🤖 AI 분석 중…</b><span class="spin"></span></div><p>국내 뉴스·공시와 해외 영문 기사를 검색하고 있습니다. 30초~1분 정도 걸립니다. 이 화면을 닫지 말고 기다려 주세요.</p></div>';
  try{
    const prompt=s.prompt+(force?`\n(재분석 요청: ${new Date().toISOString()})`:'')+(s.kind==='hold'?JSON_HOLD:JSON_PICK);
    const j=await relay('/analyze',{method:'POST',body:JSON.stringify({prompt})});
    const a=parseAI(j.text);s.pre=null;saveAI(s.key,a);slot.innerHTML=aiResultHTML(a,s,i);
  }catch(e){
    slot.innerHTML=`<div class="aibox aibtn"><p><b style="color:var(--bad)">분석하지 못했습니다</b><br>${esc(e.message)}</p><span style="display:flex;gap:6px"><button class="btn ai" data-run="${i}">다시 시도</button><button class="btn ghost" data-p="${i}">복사해서 Claude에서 보기</button></span></div>`;
  }
}
document.addEventListener('click',e=>{
  const r=e.target.closest('[data-run]');if(r){runAI(+r.dataset.run,!!r.dataset.force);return;}
  const b=e.target.closest('[data-p]');if(b)askAI(SLOTS[+b.dataset.p].prompt);
});

/* ───── 추천 방식 설명 ───── */
function renderHow(){
  const c=D.cfg;
  const rule=k=>`<div class="rule"><h5 class="${STRAT[k].c}">${STRAT[k].n}</h5>
    <ol>${D.rules[k].map(([l,w])=>`<li>${esc(l)} <span>${w}점</span></li>`).join('')}</ol>
    <div class="tg">1차 목표: ${D.targets[k][0]}<br>2차 목표: ${D.targets[k][1]}<br>손절: ${D.targets[k][2]}</div></div>`;
  document.getElementById('howbody').innerHTML=`
    <div class="kinds">
      <div class="kind"><b>📊 차트 분석 결론</b>정해진 공식으로 계산한 결과입니다. 사람의 판단이나 AI의 해석은 들어가지 않습니다. 전략·점수·목표가·손절가가 여기에 해당합니다.</div>
      <div class="kind ai"><b>🤖 AI 분석 결론</b>🤖 버튼을 누르면 Claude가 요청이 입력된 채로 열리고, 전송만 누르면 최근 한 달 국내 뉴스·공시와 해외 영문 기사를 검색해 의견(매수 / 관망 / 주의)을 냅니다. 카드의 📰 뉴스 목록은 매일 아침 제목만 모은 것으로 AI 판단이 아닙니다.</div>
    </div>
    <div><h4>1단계 · 기본 필터</h4><p>코스피·코스닥 중 거래대금 ${f0(c.min_amount/1e8)}억 원 이상, 시가총액 ${f0(c.min_mcap/1e8)}억 원 이상, 주가 ${f0(c.min_price)}원 이상, 상장 6개월 이상인 종목만 봅니다. 우선주·ETF·ETN·스팩은 빼고, 실적을 확인한 결과 적자(EPS 마이너스)인 기업도 추천에서 뺍니다.</p></div>
    <div><h4>2단계 · 전략별 기본 점수 (100점 만점)</h4><p style="margin-bottom:8px">세 전략마다 조건 5개를 얼마나 충족하는지로 점수를 매깁니다. 조건을 완전히 채우면 만점, 근접하면 부분 점수입니다.</p>
      <div class="rules">${rule(3)}${rule(1)}${rule(2)}</div></div>
    <div><h4>3단계 · 보조 지표로 가감점</h4>
      <div class="auxrules">
        <div><b>손익비</b> (1차 목표까지 수익 ÷ 손절까지 손실): 2 이상 +8, 1.5 이상 +4, 1 미만 −12</div>
        <div><b>수급</b> (최근 5일 외국인·기관 순매수): 둘 다 매수 +6, 합계 매수 +3, 둘 다 매도 −6</div>
        <div><b>MACD 히스토그램</b>: 반등 전략은 하락 둔화·0선 돌파 +5, 다른 전략은 상승 확대 +3, 꺾이면 감점</div>
        <div><b>OBV</b>: 가격은 저점을 낮추는데 거래량 누적은 올라가는 강세 다이버전스 +3~6</div>
        <div><b>상대강도</b>: 최근 20일 수익률이 시장 평균보다 좋으면 +2~4, 크게 나쁘면 −3~4</div>
        <div><b>매물대</b>: 목표가 아래에 거래가 많이 쌓인 가격대가 있으면 목표가를 그 아래로 낮춥니다</div>
      </div></div>
    <div><h4>4단계 · 순위</h4><p>최종 점수 = 기본 점수 + 가감점 (0~100점). 전략별로 1~3위는 매수 후보, 4~6위는 지켜볼 종목, 7~16위는 목록으로 보여줍니다. 데이터는 네이버 금융 기준이며 평일 아침마다 전 거래일 종가로 다시 계산합니다.</p></div>`;
}

/* ───── 시장 · 탭 ───── */
function renderPulse(){
  const B=D.breadth;
  const regime=B.above20>=60?['상승 우위','추세 눌림목·거래량 돌파가 잘 맞는 장입니다.']:B.above20>=45?['중립','추세 눌림목 위주로, 종목 수를 줄이세요.']:['약세','과매도 반등 위주로, 비중을 줄이세요.'];
  const g=(t,v)=>`<div><dt>${t}</dt><dd class="tn">${v==null?'-':v.toFixed(0)+'%'}</dd><div class="gauge"><i style="width:${v||0}%"></i></div></div>`;
  document.getElementById('pulse').innerHTML=g('20일선 위 종목',B.above20)+g('60일선 위 종목',B.above60)+g('상승 마감 종목',B.rising)+
    `<div class="regime"><dt>시장 분위기</dt><dd>${regime[0]}</dd><p>${regime[1]} 아래에서 <b>${STRAT[D.default_tab].n}</b> 탭을 먼저 보여드립니다.</p></div>`;
}
let current=D.default_tab;
function renderTabs(){
  document.getElementById('tabs').innerHTML=D.tabs.map(t=>`<button class="tab" role="tab" style="--tc:var(${STRAT[t.k].v})" aria-selected="${t.k===current}" data-k="${t.k}">${t.name}${t.k===D.default_tab?'<span class="rec">오늘 추천</span>':''}</button>`).join('');
  document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{current=+b.dataset.k;renderTabs();renderPanel();});
}

function scoreHTML(x){
  return `<div class="score ${STRAT[x.strat].c}"><div class="bar"><i style="width:${x.score}%"></i></div><b class="tn">${x.score}</b></div>
    <div class="scorebrk tn">기본 ${x.base}점 ${x.adj>=0?'+':'−'} 보조 ${Math.abs(x.adj)}점</div>`;
}
function checksHTML(list){return '<ul class="checks">'+list.map(k=>{const[c,i]=ckIcon(k.s);return `<li class="${c}"><span class="ic">${i}</span><span class="lb">${esc(k.label)}</span><span class="vl tn">${esc(k.v)}</span></li>`;}).join('')+'</ul>';}
function auxHTML(list){return '<ul class="aux">'+list.map(a=>`<li><span>${esc(a.label)}</span><span class="vl tn">${esc(a.v)}</span><span class="d ${dcls(a.d)} tn">${dtxt(a.d)}</span></li>`).join('')+'</ul>';}
function tgHTML(x){const t=x.tg;return `<dl class="tgbox tn">
  <div><dt>1차 목표</dt><dd class="upc">${f0(t.t1)}<small>${pctS(t.t1p)}</small></dd></div>
  <div><dt>2차 목표</dt><dd class="upc">${f0(t.t2)}<small>${pctS(t.t2p)}</small></dd></div>
  <div><dt>손절 · ${esc(t.stop_label)}</dt><dd class="downc">${f0(t.stop)}<small>${pctS(t.sp)}</small></dd></div>
  <div class="rr"><span>손익비 <b>1 : ${t.rr==null?'-':t.rr.toFixed(1)}</b></span><span class="vl">${esc(t.note||'')}</span></div></dl>`;}
function aiHTML(x){
  return aiSlot({prompt:x.prompt,code:x.code,kind:'pick',key:`ai:${D.asof}:${x.code}:p${x.strat}`,score:x.score,pre:x.ai||null});
}
function renderPanel(){
  const tab=D.tabs.find(t=>t.k===current),st=tab.stocks;
  document.getElementById('tabdesc').textContent=STRAT[current].d+(tab.loss?` (적자 기업 ${tab.loss}개는 제외했습니다.)`:'');
  const picks=st.slice(0,3),watch=st.slice(3,6),rest=st.slice(6);
  let h='';
  h+='<h3 class="sec">매수 후보 1–3위</h3>';
  h+=picks.length?picks.map((x,i)=>`
    <article class="pick">
      <div class="info">
        <span class="big tn" aria-hidden="true">${x.rank}</span>
        <span class="tag ${STRAT[x.strat].c}">${STRAT[x.strat].n}</span>
        <div class="name">${esc(x.name)} <span class="code tn">${x.code} ${x.market}</span></div>
        <div class="price tn">${f0(x.close)}<span>${pct(x.chg)}</span></div>
        <div class="lbl">📊 차트 분석 결론</div>
        ${scoreHTML(x)}
        ${tgHTML(x)}
        <div class="lbl">기본 조건</div>${checksHTML(x.checks)}
        <div class="lbl">보조 지표 가감점</div>${auxHTML(x.aux)}
        <dl class="kv tn">
          <div><dt>거래대금</dt><dd>${won(x.amount)}</dd></div>
          <div><dt>시가총액</dt><dd>${won(x.mcap)}</dd></div>
          <div><dt>거래량</dt><dd>${x.vr==null?'-':x.vr.toFixed(1)+'배'}</dd></div>
        </dl>
      </div>
      <div class="right"><div class="chartbox" data-i="${i}"></div>${newsHTML(x.code)}${aiHTML(x)}</div>
    </article>`).join(''):'<div class="empty">오늘은 이 전략에 맞는 종목이 없습니다.</div>';
  if(watch.length){
    h+='<h3 class="sec">지켜볼 종목 4–6위</h3><div class="watch">'+watch.map((x,i)=>`
      <article class="w">
        <div class="top"><span class="rk">${x.rank}위</span><span class="tag ${STRAT[x.strat].c}">점수 ${x.score}</span></div>
        <div class="name">${esc(x.name)} <span class="code tn">${x.code}</span></div>
        <div class="price tn">${f0(x.close)}<span>${pct(x.chg)}</span></div>
        <div class="mini" data-i="${i+3}"></div>
        ${tgHTML(x)}
        <div class="lbl">빠진 조건</div>
        ${checksHTML(x.checks.filter(k=>k.s<.99).length?x.checks.filter(k=>k.s<.99):x.checks.slice(0,2))}
        ${newsHTML(x.code,3)}
        <div style="margin-top:12px">${aiHTML(x)}</div>
      </article>`).join('')+'</div>';
  }
  if(rest.length){
    h+='<h3 class="sec">그 외 후보 7–16위</h3><div class="tablewrap"><table><thead><tr><th>순위</th><th>종목</th><th>종가</th><th>등락률</th><th>1차 목표</th><th>손절</th><th>손익비</th><th>수급(5일)</th><th>RSI</th><th>점수</th></tr></thead><tbody>'+
      rest.map(x=>{const fl=x.aux.find(a=>a.label.startsWith('수급'));return `<tr class="tn"><td class="rk">${x.rank}</td><td class="nm"><b>${esc(x.name)}</b><span>${x.code} ${x.market}</span></td>
      <td>${f0(x.close)}</td><td>${pct(x.chg)}</td><td class="upc">${f0(x.tg.t1)} <small>${pctS(x.tg.t1p)}</small></td><td class="downc">${f0(x.tg.stop)}</td>
      <td>${x.tg.rr==null?'-':'1:'+x.tg.rr.toFixed(1)}</td><td class="${fl?dcls(fl.d):''}">${fl&&fl.d>0?'매수 우위':fl&&fl.d<0?'매도 우위':'중립'}</td><td>${x.rsi??'-'}</td><td><b>${x.score}</b></td></tr>`;}).join('')+'</tbody></table></div>';
  }
  document.getElementById('panel').innerHTML=h;
  drawAll();
}
function drawAll(){
  const st=D.tabs.find(t=>t.k===current).stocks;
  document.querySelectorAll('#panel .chartbox').forEach(b=>drawChart(b,st[+b.dataset.i].series));
  document.querySelectorAll('#panel .mini').forEach(b=>drawMini(b,st[+b.dataset.i].series));
}

/* ───── 내 종목 점검 ───── */
const KEY='my-holdings-v1';
const loadH=()=>{try{return JSON.parse(localStorage.getItem(KEY))||[];}catch(_){return [];}};
const saveH=h=>{try{localStorage.setItem(KEY,JSON.stringify(h));}catch(_){}};
const NAMES=Object.assign({},D.names||{});Object.entries(D.all).forEach(([code,a])=>{if(!NAMES[code])NAMES[code]=a[0];});
const byName={};Object.entries(NAMES).forEach(([code,n])=>{byName[n]=code;});
const SNAP={},PENDING={};
const snapOf=code=>D.all[code]||SNAP[code]||null;
function vpLevels(h,l,v,c0,bins=28){
  if(h.length<20)return [null,null];const lo=Math.min(...l),hi=Math.max(...h),bh=(hi-lo)/bins;if(!(bh>0))return [null,null];
  const vp=new Array(bins).fill(0);for(let i=0;i<h.length;i++){const a=Math.min(bins-1,Math.max(0,Math.floor((l[i]-lo)/bh))),b=Math.min(bins-1,Math.max(0,Math.floor((h[i]-lo)/bh)));for(let j=a;j<=b;j++)vp[j]+=v[i]/(b-a+1);}
  const sorted=[...vp].sort((x,y)=>x-y),q75=sorted[Math.floor(bins*.75)],med=sorted[Math.floor(bins/2)],th=Math.max(q75,1.2*med);
  let res=null,sup=null;for(let j=0;j<bins;j++){const el_=lo+j*bh,eh=lo+(j+1)*bh;if(vp[j]>=th&&res==null&&el_>c0*1.005)res=el_;if(vp[j]>=th&&eh<c0*.995)sup=eh;}
  return [res,sup];
}
function computeSnap(j,name){
  const {h,l,c,v}=j,n=c.length,L=n-1;
  const mean=(s,e)=>{let t=0;for(let i=s;i<e;i++)t+=c[i];return t/(e-s);};
  const ma=(p,k)=>p-k+1>=0?mean(p-k+1,p+1):null;
  const ma20=ma(L,20),ma60=ma(L,60),ma60_5=ma(L-5,60);
  let bbu=null,bbl=null;if(ma20!=null){let q=0;for(let i=n-20;i<n;i++)q+=(c[i]-ma20)**2;const sd=Math.sqrt(q/20);bbu=ma20+2*sd;bbl=ma20-2*sd;}
  let au=null,ad=null,atr=null;
  for(let i=1;i<n;i++){const d=c[i]-c[i-1],u=Math.max(d,0),dn=Math.max(-d,0);au=au==null?u:au+(u-au)/14;ad=ad==null?dn:ad+(dn-ad)/14;
    const tr=Math.max(h[i]-l[i],Math.abs(h[i]-c[i-1]),Math.abs(l[i]-c[i-1]));atr=atr==null?tr:atr+(tr-atr)/14;}
  const rsi=n>14?(ad===0?100:100-100/(1+au/ad)):null;
  const ema=(arr,sp)=>{const a=2/(sp+1),o=[];arr.forEach((x,i)=>o.push(i?o[i-1]+a*(x-o[i-1]):x));return o;};
  const e12=ema(c,12),e26=ema(c,26),mc=c.map((_,i)=>e12[i]-e26[i]),sg=ema(mc,9),hs=mc.map((x,i)=>x-sg[i]);
  const vp=vpLevels(h.slice(-100),l.slice(-100),v.slice(-100),c[L]);
  return [name,'',c[L],n>1?(c[L]/c[L-1]-1)*100:null,ma20,ma60,(ma60!=null&&ma60_5!=null&&ma60>ma60_5)?1:-1,bbu,bbl,rsi,atr,hs[L],hs[L-1],
    Math.max(...h.slice(-20)),Math.min(...l.slice(-20)),n>61?Math.max(...h.slice(-61,-1)):null,vp[0],vp[1],'',null];
}
async function fetchSnap(code){
  if(PENDING[code])return;PENDING[code]=1;
  try{const j=await relay('/chart?code='+code);SNAP[code]=computeSnap(j,NAMES[code]||code);}
  catch(e){SNAP[code]=null;PENDING[code]='err:'+e.message;}
  renderHolds();
}
function findCode(q){q=q.trim();if(!q)return null;if(NAMES[q])return q;const m=q.match(/\((\w{6})\)/);if(m&&NAMES[m[1]])return m[1];
  if(/^[0-9A-Z]{6}$/.test(q)&&AI_URL)return q;
  if(byName[q])return byName[q];const k=Object.keys(byName).find(n=>n.replace(/\s/g,'')===q.replace(/\s/g,''));return k?byName[k]:null;}

function analyzeHolding(code,qty,avg){
  const [name,mk,c,chg,ma20,ma60,slope,bbu,bbl,rsi,atr0,h0,h1,hi20,lo20,hi60,res,sup,pick,rs]=snapOf(code);
  const atr=atr0||c*.03,pl=(c/avg-1)*100,plAmt=(c-avg)*qty;
  const pb=(bbu&&bbl&&bbu>bbl)?(c-bbl)/(bbu-bbl):null;
  const up=ma20&&ma60&&c>ma20&&ma20>ma60&&slope>0, down=ma60&&c<ma60&&slope<0;
  const macdUp=h0!=null&&h1!=null&&h0>h1;
  let stop=up?Math.max((ma20||c)-atr,lo20||0):(sup?sup-.5*atr:(lo20||c)-.5*atr), stopWhy=up?'20일선 − 1×ATR':'아래 매물대·20일 저점';
  if(!(stop<c*.995)){stop=c-1.5*atr;stopWhy='현재가 − 1.5×ATR';}
  const cand=[bbu,hi20,res].filter(x=>x&&x>c*1.01);
  let t1=cand.length?Math.min(...cand):c+1.5*atr; let t2=Math.max(hi60||0,c+2.5*atr); if(t2<=t1*1.01)t2=t1+atr;
  const why=[];
  if(ma20&&ma60)why.push(up?'주가가 20일선·60일선 위에 있고 60일선이 올라가는 상승 추세입니다.':down?'주가가 60일선 아래에 있고 60일선이 내려가는 하락 추세입니다.':`주가가 20일선 ${c>ma20?'위':'아래'}, 60일선 ${c>ma60?'위':'아래'}에 있는 혼조 구간입니다.`);
  if(rsi!=null)why.push(`RSI ${rsi.toFixed(0)}: ${rsi>=70?'단기 과열 구간입니다.':rsi<=30?'과매도 구간입니다.':rsi<=40?'약한 편입니다.':'보통 수준입니다.'}`);
  if(pb!=null)why.push(`볼린저밴드 ${pb>=1?'상단을 넘었습니다 (단기 과열 신호).':pb<=0?'하단 아래에 있습니다.':pb>=.8?'상단 근처입니다.':pb<=.2?'하단 근처입니다.':'중간 부근입니다.'}`);
  if(h0!=null)why.push(`MACD 히스토그램 ${h0>=0?(macdUp?'상승 확대':'상승 둔화'):(macdUp?'하락 둔화 (반등 준비)':'하락 확대')}입니다.`);
  if(rs!=null)why.push(`최근 20일 시장 대비 ${rs>=0?'강했습니다':'약했습니다'} (${rs>0?'+':''}${rs.toFixed(1)}%p).`);
  if(res)why.push(`${f0(res)}원 부근에 위쪽 매물대(저항)가 있습니다.`);
  if(pick)why.push(`오늘 매수 후보 목록에 있습니다: ${pick}.`);
  let v,tone,plan;
  if(down&&h0!=null&&h0<0&&!macdUp){v=pl<0?'손절 검토':'이익 보호 매도';tone='no';plan=pl<0?`하락 추세가 이어지고 있습니다. 반등 시 ${f0(Math.min(ma20||t1,t1))}원 부근에서 비중을 줄이고, ${f0(stop)}원을 깨면 정리하는 것을 검토하세요.`:'수익이 남아 있을 때 일부(1/2)를 정리하고, 나머지는 손절가 이탈 시 정리하는 방법을 검토하세요.';}
  else if(((rsi!=null&&rsi>=70)||(pb!=null&&pb>=1))&&pl>=10){v='일부 익절 (1/3)';tone='mid';plan=`단기 과열 신호가 있습니다. 1/3 정도 수익을 확정하고, 나머지는 ${f0(stop)}원 손절가를 올려 잡은 채 보유하는 방법을 검토하세요.`;}
  else if(up){v='보유 (추세 유지)';tone='ok';plan=`1차 목표 ${f0(t1)}원 도달 시 1/3, 2차 목표 ${f0(t2)}원 도달 시 1/3을 나눠 매도하고, 나머지는 20일선을 깨기 전까지 보유하는 방법을 검토하세요.`;}
  else if(rsi!=null&&rsi<=35&&macdUp){v='보유 (반등 대기)';tone='ok';plan=`과매도에서 하락이 둔화되고 있습니다. ${f0(stop)}원을 지키는 동안은 반등을 기다리고, ${f0(t1)}원 부근에서는 비중 조절을 검토하세요.`;}
  else if(ma60&&c<ma60&&pl<=-10){v='손절 검토';tone='no';plan=`손실이 크고 60일선 아래에 있습니다. 추세가 돌아서기 전에는 추가 매수(물타기)를 피하고, ${f0(stop)}원 이탈 시 정리를 검토하세요.`;}
  else{v='보유 · 관망';tone='mid';plan=`뚜렷한 방향이 없는 구간입니다. ${f0(stop)}원 위에서는 보유하고, ${f0(t1)}원 부근 저항을 넘는지 지켜보세요.`;}
  const tips=[];
  if(up&&rsi!=null&&rsi>=40&&rsi<=55&&ma20&&Math.abs(c/ma20-1)<.03)tips.push('20일선 눌림 구간이라 분할 추가매수를 고려해볼 수 있는 자리입니다.');
  if(pl<=-15&&!up)tips.push('손실 구간에서의 물타기는 추세가 돌아선 것(20일선 회복 등)을 확인한 뒤에 하는 것이 안전합니다.');
  return {name,mk,c,chg,pl,plAmt,val:c*qty,v,tone,plan,why,tips,stop,stopWhy,t1,t2,qty,avg,code};
}
function lvBar(r){
  const pts=[['손절',r.stop,'var(--down)'],['평단',r.avg,'var(--muted)'],['현재',r.c,'var(--ink)'],['목표1',r.t1,'var(--up)'],['목표2',r.t2,'var(--up)']];
  const lo=Math.min(...pts.map(p=>p[1])),hi=Math.max(...pts.map(p=>p[1])),W=600,X=v=>20+(v-lo)/((hi-lo)||1)*(W-40);
  const sorted=[...pts].sort((a,b)=>a[1]-b[1]);
  return `<svg class="lvbar" viewBox="0 0 ${W} 80" width="100%" role="img" aria-label="가격 위치"><line class="ax" x1="20" x2="${W-20}" y1="42" y2="42"/>`+
    sorted.map(([l,v,col],i)=>{const x=X(v),y=i%2?72:20,an=x<90?'start':x>W-90?'end':'middle';
      return `<circle cx="${x}" cy="42" r="${l==='현재'?7:5}" fill="${col}"/><text x="${x}" y="${y}" text-anchor="${an}" fill="${col}">${l} ${f0(v)}</text>`;}).join('')+'</svg>';
}
function holdPrompt(r){
  return `한국 주식 ${r.name}(${r.code}, ${r.mk})을 ${f0(r.qty)}주, 평균 ${f0(r.avg)}원에 보유하고 있어.\n`+
    `- 기준일 ${D.asof} 종가 ${f0(r.c)}원, 평가손익 ${pctS(r.pl)} (${f0(r.plAmt)}원)\n`+
    `- 차트 점검 결과: ${r.v}. 손절 참고 ${f0(r.stop)}원, 1차 목표 ${f0(r.t1)}원, 2차 목표 ${f0(r.t2)}원\n`+
    `- 차트 근거: ${r.why.slice(0,3).join(' ')}\n\n`+
    `최근 한 달 동안의 국내 뉴스·공시와 Reuters, Bloomberg, CNBC 같은 해외 영문 기사(영문은 한국어로 요약), 업황, 외국인·기관 수급을 웹에서 검색해서 `+
    `최근 한 달 주요 이슈를 날짜순으로 정리하고, 차트 점검 결과와 같은 방향인지 확인해줘. `+
    `보유 / 일부 매도 / 전량 매도 중 어떤 전략이 좋을지 근거 3가지, 주요 리스크, 참고 기사 링크와 함께 알려줘.`;
}
function renderHolds(){
  const list=loadH(),box=document.getElementById('holds');
  if(!list.length){box.innerHTML='<div class="empty" style="margin-top:14px">아직 저장된 종목이 없습니다. 위에 종목과 수량, 평균 매입가를 넣고 <b>점검하고 저장</b>을 눌러 보세요.</div>';return;}
  box.innerHTML=list.map((h,idx)=>{
    if(!snapOf(h.code)){
      const nm=esc(NAMES[h.code]||h.code),st=PENDING[h.code];
      if(AI_URL&&!st)setTimeout(()=>fetchSnap(h.code),0);
      const msg=!AI_URL?'오늘 아침 시세 수집에서 빠진 종목이라 차트 점검을 할 수 없습니다. 내일 아침 갱신 때 다시 나타나는지 확인해 주세요. 아래 AI 분석은 지금도 쓸 수 있습니다.':
        (typeof st==='string'&&st.startsWith('err:'))?'시세를 불러오지 못했습니다: '+esc(st.slice(4)):'시세를 불러오는 중입니다…';
      const bp=`한국 주식 ${NAMES[h.code]||h.code}(${h.code})을 ${f0(h.qty)}주, 평균 ${f0(h.avg)}원에 보유하고 있어. `+
        `현재 주가와 최근 차트 흐름(20일선·60일선, RSI, 거래량), 최근 한 달 국내 뉴스·공시와 Reuters·Bloomberg 같은 해외 영문 기사(영문은 한국어로 요약)를 웹에서 검색해서 `+
        `최근 한 달 주요 이슈를 날짜순으로 정리하고, 보유 / 일부 매도 / 전량 매도 중 어떤 전략이 좋을지 손절가·목표가, 근거 3가지, 리스크, 참고 기사 링크와 함께 알려줘.`;
      return `<div class="hold"><div><div class="hdr"><div><b>${nm}</b> <span class="code">${esc(h.code)}</span> · ${f0(h.qty)}주 · 평단 ${f0(h.avg)}</div><button class="rm" data-rm="${idx}">삭제</button></div>
        <p class="vl" style="text-align:left;margin:8px 0 0">${msg}</p></div><div>${aiSlot({prompt:bp,code:h.code,kind:'hold',key:`ai:${D.asof}:${h.code}:hx`,tone:'mid'})}</div></div>`;
    }
    const r=analyzeHolding(h.code,h.qty,h.avg);
    const row=(l,p,why)=>`<tr><td>${l}<br><small class="vl" style="text-align:left">${why}</small></td><td>${f0(p)}</td><td>${pctS((p/r.c-1)*100)}</td><td class="${p>=r.avg?'upc':'downc'}">${(p-r.avg)*r.qty>=0?'+':''}${f0((p-r.avg)*r.qty)}원</td></tr>`;
    return `<article class="hold">
      <div>
        <div class="hdr"><div><div class="name" style="margin:0">${esc(r.name)} <span class="code tn">${r.code} ${r.mk}</span></div>
          <div class="price tn">${f0(r.c)}<span>${pct(r.chg)}</span></div></div><button class="rm" data-rm="${idx}">삭제</button></div>
        <dl class="pos-grid tn">
          <div><dt>보유</dt><dd>${f0(r.qty)}주 · 평단 ${f0(r.avg)}</dd></div>
          <div><dt>평가금액</dt><dd>${f0(r.val)}원</dd></div>
          <div><dt>평가손익</dt><dd class="${r.pl>=0?'upc':'downc'}">${r.plAmt>=0?'+':''}${f0(r.plAmt)}원</dd></div>
          <div><dt>수익률</dt><dd class="${r.pl>=0?'upc':'downc'}">${pctS(r.pl,2)}</dd></div>
        </dl>
        <div class="hv"><div class="lab">📊 차트 기준 전략</div><div class="val ${r.tone}">${r.v}</div></div>
        <div class="plan"><b>실행 계획 (예시)</b>${r.plan}${r.tips.map(t=>`<br>· ${t}`).join('')}</div>
        <div style="margin-top:12px">${aiSlot({prompt:holdPrompt(r),code:r.code,kind:'hold',key:`ai:${D.asof}:${r.code}:h${r.qty}:${r.avg}`,tone:r.tone==='no'?'bad':r.tone})}</div>
      </div>
      <div>
        ${lvBar(r)}
        <table class="lvtable tn"><thead><tr><th>가격</th><th>원</th><th>현재가 대비</th><th>그때 손익</th></tr></thead><tbody>
          ${row('2차 목표',r.t2,'')}${row('1차 목표',r.t1,'')}${row('손절 참고',r.stop,r.stopWhy)}</tbody></table>
        <div class="lbl">판단 근거</div><ul class="why">${r.why.map(w=>`<li>${esc(w)}</li>`).join('')}</ul>
        ${newsHTML(r.code,4)}
      </div></article>`;
  }).join('');
  box.querySelectorAll('[data-rm]').forEach(b=>b.onclick=()=>{const l=loadH();l.splice(+b.dataset.rm,1);saveH(l);renderHolds();});
}
function initHolds(){
  const dl=document.getElementById('stocklist');
  dl.innerHTML=Object.entries(NAMES).map(([code,n])=>`<option value="${esc(n)} (${code})"></option>`).join('');
  const err=document.getElementById('fErr');
  document.getElementById('fAdd').onclick=()=>{
    const code=findCode(document.getElementById('fStock').value),qty=+document.getElementById('fQty').value,avg=+document.getElementById('fAvg').value;
    const bad=!code?'종목을 찾지 못했습니다. 목록에서 골라 주시거나 6자리 종목코드로 입력해 주세요 (ETF·ETN은 지원하지 않습니다).':!(qty>0)?'보유 수량을 입력해 주세요.':!(avg>0)?'평균 매입가를 입력해 주세요.':'';
    err.hidden=!bad;err.textContent=bad;if(bad)return;
    const l=loadH().filter(h=>h.code!==code);l.unshift({code,qty,avg});saveH(l);renderHolds();
    ['fStock','fQty','fAvg'].forEach(id=>document.getElementById(id).value='');
  };
  renderHolds();
}

/* ───── 시작 ───── */
document.getElementById('meta').textContent=`${D.asof} 종가 기준 · 분석 ${D.universe.toLocaleString()}종목 · ${D.generated} 생성 · AI 분석 ${D.ai_mode==='auto'?'자동':'버튼 방식'}`;
document.getElementById('demo').hidden=!D.demo;
document.getElementById('refresh').onclick=()=>location.reload();
try{if(localStorage.getItem('how-closed')==='1')document.getElementById('how').open=false;}catch(_){}
document.getElementById('how').addEventListener('toggle',e=>{try{localStorage.setItem('how-closed',e.target.open?'0':'1');}catch(_){}});
renderHow();renderPulse();renderTabs();renderPanel();initHolds();
let rt;addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(drawAll,150);});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()

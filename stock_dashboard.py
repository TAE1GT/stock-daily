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
    days=500,              # 수집 거래일 수 (약 2년: 52주 지표 · 1년 반 과거 성적 검증용)
    min_amount=30e8,       # 최소 거래대금 30억
    min_mcap=1000e8,       # 최소 시가총액 1,000억
    min_price=2000,        # 최소 주가
    min_history=120,       # 상장 후 최소 거래일 (약 6개월)
    chart_days=100,        # 차트 표시 기간
    top_n=16,              # 1~3 / 4~6 / 7~16
    cand_n=25,             # 전략별 2차 검토(수급·실적 조회) 후보 수
    cost=0.25,             # 왕복 거래비용 % (수수료+세금) — 성적 계산 때 차감
    vb_filter=dict(max_day_change=15.0, min_close_to_high=0.95, min_vol_ratio=3.0),   # 거래량 돌파 '그림자 필터' (기록만, 아직 미적용)
    min_samples=30, min_days=15,   # 실시간 기록 진단: 이보다 적으면 '관찰 중'

)

STATUS = {}   # 데이터 출처별 수집 결과 (상세 설명 페이지에 표시)


def stat(key, ok, detail=""):
    STATUS[key] = dict(ok=(None if ok is None else bool(ok)), detail=str(detail))


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
            stat("종목 목록 (네이버)", True, f"{fn.__doc__.strip()} · {len(df)}종목")
            return df
    stat("종목 목록 (네이버)", False, "모든 경로 실패")
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
        stat("종목 시세 (네이버 차트)", len(data) > 0, f"{len(data)}종목 성공 · {len(failed)}종목 실패")
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
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.offsets.BDay(1), periods=days)
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


def evaluate(P, I, market, light=False):
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

    # 피보나치: 최근 120일 고점·저점 사이 되돌림 위치
    hw = P["h"].iloc[-120:].to_numpy(float)
    lw = P["l"].iloc[-120:].to_numpy(float)
    ihi = np.where(np.isnan(hw), -np.inf, hw).argmax(0)
    ilo = np.where(np.isnan(lw), np.inf, lw).argmin(0)
    fhi, flo = S(np.nanmax(hw, 0)), S(np.nanmin(lw, 0))
    fup = S(ihi > ilo) > 0                                   # 저점 → 고점 순서면 상승 파동
    rng_ = fhi - flo
    big = (rng_ / flo) >= 0.15                                # 변동폭 15% 이상일 때만 의미 있음
    depth = (fhi - c) / rng_                                  # 상승 파동에서 되돌린 비율
    bounce = (c - flo) / rng_                                 # 하락 파동에서 반등한 비율

    def fib_txt(t):
        if not big[t]:
            return "변동폭 작음"
        return f"상승폭의 {depth[t] * 100:.0f}% 되돌림" if fup[t] else f"하락폭의 {bounce[t] * 100:.0f}% 반등"

    fib1 = W(~big, 0, W(~fup, -2, W((depth >= 0.5) & (depth <= 0.65), 5,
             W((depth >= 0.33) & (depth < 0.5), 3, W(depth > 0.8, -3, 0)))))
    fib2 = W(big & fup & (depth <= 0.05), 2, 0)
    fib3 = W(big & fup & (depth >= 0.58) & (depth <= 0.82), 5,
             W(big & ~fup & (bounce >= 0.1) & (bounce <= 0.382), 2, 0))

    # 세력 패턴 (최근 20일): 큰손의 매집·이탈 흔적으로 자주 쓰이는 캔들·거래량 모양
    C_, O_, H_, L_, V_ = (P[k].iloc[-20:].to_numpy(float) for k in "cohlv")
    VM = I["vma20"].iloc[-20:].to_numpy(float)
    CH = I["chg"].iloc[-20:].to_numpy(float)
    cols_ = np.arange(C_.shape[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        VR = V_ / VM
        bigb = (CH >= 8) & (VR >= 3)
        bigb[-2:] = False
        lastb = np.where(bigb.any(0), 19 - np.argmax(bigb[::-1], 0), -1)
        jj = np.clip(lastb, 0, 19)
        midb = (O_[jj, cols_] + C_[jj, cols_]) / 2
        v3 = np.nanmean(V_[-3:], 0)
        pa = (lastb >= 0) & (C_[-1] >= midb) & (v3 < V_[jj, cols_] * 0.4)          # 장대양봉 후 거래량 줄며 눌림
        pa_break = (lastb >= 0) & (C_[-1] < midb * 0.97)                             # 장대양봉 중간값 이탈
        lo60_ = P["l"].iloc[-60:].min().to_numpy(float)
        hi60_ = P["h"].iloc[-60:].max().to_numpy(float)
        lowzone = C_[-5:] <= lo60_ + (hi60_ - lo60_) * 0.4
        pc = ((VR[-5:] >= 3) & (np.abs(C_[-5:] - O_[-5:]) <= (H_[-5:] - L_[-5:]) * 0.35) & lowzone).any(0)   # 저점권 매집봉
        prng = np.nanmax(H_, 0) / np.nanmin(L_, 0) - 1
        obv20 = (I["obv"].iloc[-1] - I["obv"].iloc[-20]).to_numpy(float)
        pb_ = (prng <= 0.12) & (obv20 > 0) & (np.nanmean(V_[-10:], 0) > np.nanmean(V_[:10], 0) * 1.2)   # 조용한 매집
        pd_ = ((VR[-5:] >= 3) & (CH[-5:] <= -7)).any(0) | pa_break                  # 이탈 의심
    PA, PB, PC, PD = (S(x) > 0 for x in (pa, pb_, pc, pd_))
    pat = S(np.clip(np.where(pa, 5, 0) + np.where(pb_, 4, 0) + np.where(pc, 4, 0) + np.where(pd_, -6, 0), -6, 9))

    hi250 = P["h"].iloc[-250:].max()
    near52 = c / hi250
    atrp = I["atr"].iloc[-1] / c * 100
    w52 = W(near52 >= 0.9, 3, 0)
    w52_txt = lambda t: f"52주 고가의 {near52[t] * 100:.0f}%"
    volp = W(atrp > 7, -3, 0)
    volp_txt = lambda t: f"하루 평균 변동 {atrp[t]:.1f}%"

    def pat_txt(t):
        names = [n for n, f in (("장대양봉 후 눌림", PA), ("조용한 매집", PB), ("저점 매집봉", PC), ("이탈 의심", PD)) if f[t]]
        return ", ".join(names) if names else "없음"

    aux = {
        1: [("MACD 히스토그램", W(rising, 3, -2), macd_txt),
            ("OBV", W(div, 3, 0) + W(obv_up, 2, -2), obv_txt),
            ("상대강도 (20일)", rs_trend, rs_txt),
            ("피보나치", fib1, fib_txt),
            ("세력 패턴", pat, pat_txt),
            ("52주 고가 근접", w52, w52_txt),
            ("변동성", volp, volp_txt)],
        2: [("MACD 히스토그램", W(rising & (h0 > 0), 3, W(h0 < 0, -3, 0)), macd_txt),
            ("OBV", W(obv_up, 2, -2), obv_txt),
            ("상대강도 (20일)", rs_trend, rs_txt),
            ("피보나치", fib2, fib_txt),
            ("세력 패턴", pat, pat_txt),
            ("52주 고가 근접", w52, w52_txt),
            ("변동성", volp, volp_txt)],
        3: [("MACD 히스토그램", W(cross, 5, W((h0 < 0) & rising2, 5, W(rising, 2, -3))), macd_txt),
            ("OBV", W(div, 6, W(obv_up, 2, 0)), obv_txt),
            ("상대강도 (20일)", W(rs > -5, 3, W(rs < -15, -3, 0)), rs_txt),
            ("피보나치", fib3, fib_txt),
            ("세력 패턴", pat, pat_txt),
            ("변동성", volp, volp_txt)],
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
                   hi20=hi20, lo20=P["l"].iloc[-20:].min(), hi60=hi60,
                   fhi=fhi, flo=flo, fup=fup, fbig=big,
                   pat=(None if light else pd.Series([pat_txt(t) for t in idx], index=idx)))
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


def fib_levels(t, m):
    """차트에 그릴 피보나치 되돌림 가격 (38.2 / 50 / 61.8%)"""
    if not bool(m["fbig"][t]):
        return None
    hi, lo, up = float(m["fhi"][t]), float(m["flo"][t]), bool(m["fup"][t])
    return [[f"{r * 100:g}%", num(hi - (hi - lo) * r if up else lo + (hi - lo) * r)] for r in (0.382, 0.5, 0.618)]


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
def flow_sum(days):
    """최근 5일 외국인·기관 순매수 합계 + 둘 다 순매수한 연속 일수 (최근일부터)"""
    f = sum(d[0] for d in days[:5])
    i = sum(d[1] for d in days[:5])
    st = 0
    for a, b in days:
        if a > 0 and b > 0:
            st += 1
        else:
            break
    return f, i, st


def fetch_flow(code):
    try:
        data = json.loads(http_get(f"https://m.stock.naver.com/api/stock/{code}/trend?pageSize=10", retries=2))
        items = data if isinstance(data, list) else next((x for x in data.values() if isinstance(x, list)), [])
        days = []
        for it in items[:10]:
            fk = next((k for k in it if re.search(r"foreign.*pure", k, re.I)), None)
            ok = next((k for k in it if re.search(r"organ.*pure", k, re.I)), None)
            if fk is None or ok is None:
                break
            days.append((np.nan_to_num(_num(it[fk])), np.nan_to_num(_num(it[ok]))))
        if days:
            return flow_sum(days)
    except Exception:
        pass
    try:
        html = http_get(f"https://finance.naver.com/item/frgn.naver?code={code}", retries=2)
        days = []
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
            cells = [_strip(x) for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
            if len(cells) >= 7 and re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", cells[0]):
                days.append((np.nan_to_num(_num(cells[6])), np.nan_to_num(_num(cells[5]))))
                if len(days) == 10:
                    break
        if days:
            return flow_sum(days)
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
            e["f"], e["i"], e["st"] = flows[t]
        if epss[t]:
            e["eps"], e["per"] = epss[t]
        out[t] = e
    nf = sum(1 for t in codes if flows[t])
    ne = sum(1 for t in codes if epss[t])
    print(f"  수급 {nf}/{len(codes)}, 실적 {ne}/{len(codes)} 확인", flush=True)
    stat("외국인·기관 수급 (네이버)", nf > 0, f"{nf}/{len(codes)}종목")
    stat("실적 EPS·PER (네이버)", ne > 0, f"{ne}/{len(codes)}종목")
    return out, dict(flow=nf, eps=ne, total=len(codes))


def demo_extras(codes):
    rng = np.random.default_rng(7)
    out = {}
    for t in codes:
        eps = rng.normal(1500, 2000)
        out[t] = dict(f=rng.normal(0, 2e5), i=rng.normal(0, 1.5e5), st=int(rng.integers(0, 5)), eps=eps,
                      per=(rng.uniform(5, 40) if eps > 0 else None))
    return out, dict(flow=len(codes), eps=len(codes), total=len(codes))


# ───────────────────────────── 시장 위험 신호 ─────────────────────────────
def fetch_index_naver(sym, count):
    xml = http_get(f"https://fchart.stock.naver.com/sise.nhn?symbol={sym}&timeframe=day&count={count}&requestType=0", retries=2)
    recs = re.findall(r'data="(\d{8})\|([\d.]+)\|([\d.]+)\|([\d.]+)\|([\d.]+)\|([\d.]+)"', xml)
    if len(recs) < 70:
        return None
    df = pd.DataFrame(recs, columns=["d", "o", "h", "l", "c", "v"])
    df.index = pd.to_datetime(df.pop("d"), format="%Y%m%d")
    return df.astype(float)[["c", "v"]]


def fetch_yahoo(sym):
    from urllib.parse import quote
    j = json.loads(http_get(f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(sym)}?range=6y&interval=1d", retries=2))
    r = j["chart"]["result"][0]
    s = pd.Series(r["indicators"]["quote"][0]["close"],
                  index=pd.to_datetime([datetime.utcfromtimestamp(t).date() for t in r["timestamp"]]), dtype=float).dropna()
    return s[~s.index.duplicated(keep="last")] if len(s) > 40 else None


def market_risk(P, I, uni, demo):
    """시장 방향 신호: 하락 위험 점수 + 상승 신호 점수 (0~100). 선행 = 미리 켜지는 신호, 진행/확인 = 이미 움직이는 중"""
    dates = P["c"].index
    c = P["c"][uni]
    idx, src = {}, "지수"
    if not demo:
        for name in ("KOSPI", "KOSDAQ"):
            try:
                df = fetch_index_naver(name, len(dates) + 10)
                if df is not None:
                    idx[name] = df.reindex(dates).ffill()
            except Exception:
                pass
    if not idx:
        r = c.pct_change(fill_method=None).median(axis=1).fillna(0)
        idx["시장 평균"] = pd.DataFrame({"c": 1000 * (1 + r).cumprod(), "v": P["v"][uni].sum(axis=1)})
        src = "전 종목 평균(지수 대체)"

    def isig(df):
        cc, vv = df["c"], df["v"]
        ma20, ma60 = cc.rolling(20).mean(), cc.rolling(60).mean()
        ret = cc.pct_change() * 100
        dist = ((ret <= -0.2) & (vv > vv.shift(1))).astype(int).rolling(25).sum()
        volr = ret.rolling(10).std() / ret.rolling(60).std()
        hi60 = cc.rolling(60).max()
        # 바닥 확인일(Follow-through day): 최근 저점 이후 4일째 이후에 거래량 늘며 +1.5% 이상 상승
        ftd = pd.Series(False, index=cc.index)
        vals, rets, vols = cc.to_numpy(), ret.to_numpy(), vv.to_numpy()
        for i in range(25, len(cc)):
            w = vals[i - 15:i + 1]
            if np.isnan(w).any():
                continue
            lo = i - 15 + int(np.argmin(w))
            for j in range(lo + 4, i + 1):
                if rets[j] >= 1.5 and vols[j] > vols[j - 1] and vals[i] > vals[lo]:
                    ftd.iloc[i] = True
                    break
        return pd.DataFrame(dict(c=cc, ret=ret, below20=cc < ma20, below60=cc < ma60, dead=ma20 < ma60,
                                 cross_up=(cc > ma20) & (cc.shift(5) < ma20.shift(5)), golden=(ma20 > ma60) & (ma20.shift(5) <= ma60.shift(5)),
                                 dist=dist, volr=volr, dd=(cc / hi60 - 1) * 100, ftd=ftd))

    sig = {k: isig(v) for k, v in idx.items()}
    above20 = (c > I["ma20"][uni]).mean(axis=1) * 100
    newlow = (c <= c.rolling(60).min()).mean(axis=1) * 100
    newhigh = (c >= c.rolling(60).max()).mean(axis=1) * 100

    ext = load_external(demo, dates)
    E = {}
    for key, sr in ext.items():
        al = sr.reindex(sr.index.union(dates)).ffill().reindex(dates)
        E[key] = dict(v=al, chg5=(al / al.shift(5) - 1) * 100, chg20=(al / al.shift(20) - 1) * 100,
                      below20=al < al.rolling(20).mean())

    def at(i):
        val = lambda s_, col: float(s_[col].iloc[i]) if s_[col].iloc[i] == s_[col].iloc[i] else 0.0
        worst = lambda col, agg=max: agg(val(s_, col) for s_ in sig.values())
        anyb = lambda col: any(bool(s_[col].iloc[i]) for s_ in sig.values())
        a20 = float(above20.iloc[i])
        a20_5 = float(above20.iloc[i - 5]) if i >= 5 else a20
        a20_min10 = float(above20.iloc[max(0, i - 10):i + 1].min())
        near_high = any(val(s_, "dd") > -3 for s_ in sig.values())
        dist, volr, dd = worst("dist"), worst("volr"), worst("dd", min)
        nl, nh = float(newlow.iloc[i]), float(newhigh.iloc[i])
        dn, up = [], []

        def add(lst, kind, label, on, pts, v=""):
            lst.append(dict(kind=kind, label=label, on=bool(on), pts=int(pts if on else 0), val=v))

        # ── 하락 쪽 ──
        add(dn, "선행", "분산일 누적 (25일 중 지수가 거래량 늘며 하락한 날)", dist >= 4, 15 if dist >= 5 else 8, f"{dist:.0f}일")
        add(dn, "선행", "지수는 고점 근처인데 오르는 종목은 줄어듦", near_high and a20 < 50, 10, f"20일선 위 종목 {a20:.0f}%")
        add(dn, "선행", "변동성 확대 (최근 10일 흔들림이 평소보다 큼)", volr >= 1.2, 10 if volr >= 1.5 else 5, f"평소의 {volr:.1f}배")
        if "vix" in E:
            v, v5 = float(E["vix"]["v"].iloc[i]), float(np.nan_to_num(E["vix"]["chg5"].iloc[i]))
            add(dn, "선행", "미국 공포지수(VIX) 상승", v >= 20 or v5 >= 25, 10 if v >= 25 else 5, f"{v:.1f} (5일 {v5:+.0f}%)")
        if "krw" in E:
            k20 = float(np.nan_to_num(E["krw"]["chg20"].iloc[i]))
            add(dn, "선행", "원·달러 환율 급등 (외국인 매도 압력)", k20 >= 2.5, 5, f"20일 {k20:+.1f}%")
        sb = bool(E["sox"]["below20"].iloc[i]) if "sox" in E else False
        nb = bool(E["nasdaq"]["below20"].iloc[i]) if "nasdaq" in E else False
        if E.get("sox") or E.get("nasdaq"):
            add(dn, "선행", "미국 기술주·반도체 지수 20일선 아래", sb or nb, 5 if sb and nb else 3,
                ", ".join(x for x, f in (("반도체", sb), ("나스닥", nb)) if f) or "모두 위")
        mk_ = [k for k in E if k not in ("vix", "krw", "nasdaq", "sox")]
        effs = [macro_effect(k, float(E[k]["v"].iloc[i]), float(np.nan_to_num(E[k]["chg5"].iloc[i])),
                             float(np.nan_to_num(E[k]["chg20"].iloc[i]))) for k in mk_]
        nbad, ngood = sum(e < 0 for e in effs), sum(e > 0 for e in effs)
        if mk_:
            add(dn, "선행", "글로벌 지표 다수가 주식에 불리 (달러·금리·유가·금·신용·해외 증시 등)", nbad >= 4, 10 if nbad >= 6 else 5, f"{nbad}/{len(mk_)}개")
        if "oil" in E:
            o20 = float(np.nan_to_num(E["oil"]["chg20"].iloc[i]))
            add(dn, "선행", "국제유가 급등 (물가·금리 부담)", o20 >= 15, 5, f"20일 {o20:+.1f}%")
        if "hyg" in E:
            h20 = float(np.nan_to_num(E["hyg"]["chg20"].iloc[i]))
            add(dn, "선행", "신용위험 확대 (하이일드 채권 약세)", h20 <= -2 and bool(E["hyg"]["below20"].iloc[i]), 5, f"20일 {h20:+.1f}%")
        add(dn, "진행", "지수 20일선 아래", anyb("below20"), 10)
        add(dn, "진행", "지수 60일선 아래", anyb("below60"), 10)
        add(dn, "진행", "20일선이 60일선 아래 (데드크로스)", anyb("dead"), 5)
        add(dn, "진행", "오르는 종목이 빠르게 줄어듦 (5일간 20%p 이상)", a20_5 - a20 >= 20, 10, f"{a20_5:.0f}% → {a20:.0f}%")
        add(dn, "진행", "20일선 위 종목 30% 미만", a20 < 30, 10, f"{a20:.0f}%")
        add(dn, "진행", "60일 신저가 종목 급증", nl >= 8, 10 if nl >= 15 else 5, f"{nl:.1f}%")
        add(dn, "진행", "지수 60일 고점 대비 큰 하락", dd <= -5, 10 if dd <= -10 else 5, f"{dd:.1f}%")
        # ── 상승 쪽 ──
        add(up, "선행", "오르는 종목 급증 (10일 내 30% 미만 → 60% 이상)", a20_min10 < 30 and a20 >= 60, 20, f"{a20_min10:.0f}% → {a20:.0f}%")
        add(up, "선행", "바닥 확인일 (저점 후 거래량 늘며 +1.5% 이상 상승)", anyb("ftd"), 15)
        add(up, "선행", "신고가 종목이 신저가 종목보다 많음", nh > nl and nh >= 3, 10, f"신고가 {nh:.1f}% · 신저가 {nl:.1f}%")
        if "vix" in E:
            v, v5 = float(E["vix"]["v"].iloc[i]), float(np.nan_to_num(E["vix"]["chg5"].iloc[i]))
            add(up, "선행", "공포지수(VIX) 진정", v5 <= -15 or v < 16, 10, f"{v:.1f} (5일 {v5:+.0f}%)")
        if "krw" in E:
            k20 = float(np.nan_to_num(E["krw"]["chg20"].iloc[i]))
            add(up, "선행", "원·달러 환율 안정 (외국인 매수 우호)", k20 <= -2, 5, f"20일 {k20:+.1f}%")
        if E.get("sox") or E.get("nasdaq"):
            add(up, "선행", "미국 기술주·반도체 지수 20일선 위", not sb and not nb, 5)
        if mk_:
            add(up, "선행", "글로벌 지표 다수가 주식에 유리", ngood >= 4, 10 if ngood >= 6 else 5, f"{ngood}/{len(mk_)}개")
        if "copper" in E:
            cu = float(np.nan_to_num(E["copper"]["chg20"].iloc[i]))
            add(up, "선행", "구리 가격 상승 (경기 회복 기대)", cu >= 5, 5, f"20일 {cu:+.1f}%")
        add(up, "확인", "지수가 20일선 위 (최근 회복이면 가점)", not anyb("below20"), 10 if anyb("cross_up") else 5)
        add(up, "확인", "20일선이 60일선 위 (최근 골든크로스면 가점)", not anyb("dead"), 10 if anyb("golden") else 5)
        add(up, "확인", "분산일 적음 (25일 중 2일 이하)", dist <= 2, 5, f"{dist:.0f}일")
        add(up, "확인", "20일선 위 종목 60% 이상", a20 >= 60, 10, f"{a20:.0f}%")
        return min(100, sum(x["pts"] for x in dn)), min(100, sum(x["pts"] for x in up)), dn, up

    n = len(dates)
    hist = []
    for i in range(max(65, n - 30), n):
        a, b_, _, _ = at(i)
        hist.append(dict(d=dates[i].strftime("%m/%d"), s=a, u=b_))
    # 과거 검증: 각 날짜의 하락 위험 단계별로 이후 코스피 5·20일 등락
    first = next(iter(sig.values()))["c"]
    lv_rows = []
    for i in range(65, n - 5):
        a_, u_, _, _ = at(i)
        f5 = (first.iloc[i + 5] / first.iloc[i] - 1) * 100
        f20 = (first.iloc[i + 20] / first.iloc[i] - 1) * 100 if i + 20 < n else np.nan
        lv_rows.append(dict(lv=("경계 이상" if a_ >= 50 else "주의" if a_ >= 25 else "안정"), up=("개선 이상" if u_ >= 35 else "약함 이하"), f5=f5, f20=f20))
    valid = []
    if lv_rows:
        vdf = pd.DataFrame(lv_rows)
        for col, labs in (("lv", ("안정", "주의", "경계 이상")), ("up", ("개선 이상", "약함 이하"))):
            for lab in labs:
                g = vdf[vdf[col] == lab]
                if len(g) >= 5:
                    valid.append(dict(kind=("하락 위험" if col == "lv" else "상승 신호"), lab=lab, n=len(g), f5=round(g.f5.mean(), 2),
                                      up5=round((g.f5 > 0).mean() * 100), f20=(round(g.f20.mean(), 2) if g.f20.notna().sum() >= 5 else None)))
    score, upscore, dn, up = at(n - 1)
    level = "위험" if score >= 75 else "경계" if score >= 50 else "주의" if score >= 25 else "안정"
    ulevel = "강함" if upscore >= 60 else "개선" if upscore >= 35 else "약함" if upscore >= 15 else "없음"
    idxinfo = {k: dict(c=num(s_["c"].iloc[-1], 2), chg=num(s_["ret"].iloc[-1], 2), dd=num(s_["dd"].iloc[-1], 1)) for k, s_ in sig.items()}
    stat("코스피·코스닥 지수 (네이버)", src == "지수", ", ".join(sig) if src == "지수" else "실패 → 전 종목 평균으로 대체")
    print(f"시장 신호: 하락 위험 {score}({level}) · 상승 신호 {upscore}({ulevel}) · 해외 지표 {', '.join(E) or '없음'}", flush=True)
    return dict(score=score, level=level, up=upscore, ulevel=ulevel,
                prev5=hist[-6]["s"] if len(hist) >= 6 else None, uprev5=hist[-6]["u"] if len(hist) >= 6 else None,
                hist=hist, dn=dn, upl=up, idx=idxinfo, src=src, ext=sorted(E), valid=valid,
                vperiod=f"{dates[65]:%Y-%m-%d}~{dates[n - 6]:%Y-%m-%d}" if n > 71 else "")


# ───────────────────────────── 5일 전망 (확률 모델) ─────────────────────────────
SECTOR_ETF = [("반도체", "091160"), ("2차전지", "305720"), ("자동차", "091180"), ("바이오", "244580"),
              ("헬스케어", "266420"), ("IT", "266370"), ("소프트웨어", "157490"), ("은행", "091170"),
              ("증권", "102970"), ("보험", "140700"), ("건설", "117700"), ("철강", "117680"),
              ("에너지화학", "117460"), ("조선·중공업", "139230"), ("운송", "140710"), ("화장품", "228790"),
              ("미디어·엔터", "228810"), ("방산", "449450"), ("필수소비재", "266410"), ("경기소비재", "266390")]
EXT_CACHE = {}


# 키: (이름, 야후 기호, 오르면 주식에 좋은가, 20일 변화 기준%)
MACRO = {
    "vix": ("미국 공포지수(VIX)", "^VIX", False, None),
    "krw": ("원·달러 환율", "KRW=X", False, 2.0),
    "dxy": ("달러인덱스", "DX-Y.NYB", False, 2.0),
    "us10y": ("미국 10년 국채금리", "^TNX", False, 7.0),
    "oil": ("국제유가(WTI)", "CL=F", False, 12.0),
    "gold": ("금", "GC=F", False, 5.0),
    "copper": ("구리 (경기 선행)", "HG=F", True, 5.0),
    "hyg": ("하이일드 채권 (신용위험)", "HYG", True, 2.0),
    "spx": ("미국 S&P 500", "^GSPC", True, 4.0),
    "nasdaq": ("미국 나스닥", "^IXIC", True, 5.0),
    "sox": ("필라델피아 반도체", "^SOX", True, 7.0),
    "nikkei": ("일본 닛케이", "^N225", True, 5.0),
    "china": ("중국 상해종합", "000001.SS", True, 5.0),
    "taiwan": ("대만 가권", "^TWII", True, 5.0),
    "btc": ("비트코인 (위험 선호)", "BTC-USD", True, 15.0),
    "irx": ("미국 3개월 금리", "^IRX", False, 10.0),
}


def macro_effect(key, v, chg5, chg20):
    """+1 주식에 유리, -1 불리, 0 중립"""
    if key == "vix":
        return -1 if (v >= 25 or chg5 >= 20) else 1 if (v < 16 or chg5 <= -15) else 0
    name, sym, good_up, thr = MACRO[key]
    if chg20 != chg20 or abs(chg20) < thr:
        return 0
    return (1 if chg20 > 0 else -1) * (1 if good_up else -1)


def load_external(demo, dates):
    """해외 지표 (야후 파이낸스, 약 6년치): VIX·환율·달러·금리·유가·금·구리·신용·해외 증시·비트코인"""
    if "data" in EXT_CACHE:
        return EXT_CACHE["data"]
    ext = {}
    if not demo:
        from concurrent.futures import ThreadPoolExecutor

        def get(k):
            try:
                return k, fetch_yahoo(MACRO[k][1])
            except Exception:
                return k, None
        with ThreadPoolExecutor(max_workers=5) as ex:
            for k, sr in ex.map(get, list(MACRO)):
                if sr is not None:
                    ext[k] = sr
    else:
        rng = np.random.default_rng(5)
        dd = pd.bdate_range(end=dates[-1], periods=1500)
        base = dict(vix=18, krw=1300, dxy=100, us10y=4, oil=75, gold=2000, copper=4, hyg=77, spx=4500, nasdaq=14000,
                    sox=4000, nikkei=33000, china=3100, taiwan=17000, btc=40000, irx=4.5)
        for k in MACRO:
            vol = 0.03 if k in ("vix", "btc") else 0.012
            ext[k] = pd.Series(base[k] * np.exp(np.cumsum(rng.normal(0.0002, vol, len(dd)))), index=dd)
    EXT_CACHE["data"] = ext
    stat("해외 지표 (야후 파이낸스)", len(ext) > 0, f"{len(ext)}/{len(MACRO)}개" + (f" · 실패: {', '.join(MACRO[k][0] for k in MACRO if k not in ext)}" if len(ext) < len(MACRO) else ""))
    return ext


def macro_panel(ext):
    out = []
    for k, (name, sym, good_up, thr) in MACRO.items():
        sr = ext.get(k)
        if sr is None or len(sr) < 25:
            continue
        v = float(sr.iloc[-1])
        c5 = (v / float(sr.iloc[-6]) - 1) * 100
        c20 = (v / float(sr.iloc[-21]) - 1) * 100
        eff = macro_effect(k, v, c5, c20)
        out.append(dict(key=k, name=name, v=num(v, 2), c5=num(c5, 1), c20=num(c20, 1), eff=eff,
                        d=sr.index[-1].strftime("%m/%d"), spark=[num(x, 3) for x in sr.iloc[-60:]]))
    return out


def fetch_series(code, count):
    xml = http_get(f"https://fchart.stock.naver.com/sise.nhn?symbol={code}&timeframe=day&count={count}&requestType=0", retries=2)
    recs = re.findall(r'data="(\d{8})\|([\d.]+)\|([\d.]+)\|([\d.]+)\|([\d.]+)\|([\d.]+)"', xml)
    nm = re.search(r'name="([^"]*)"', xml)
    if len(recs) < 200:
        return None, None
    df = pd.DataFrame(recs, columns=["d", "o", "h", "l", "c", "v"])
    df.index = pd.to_datetime(df.pop("d"), format="%Y%m%d")
    return (nm.group(1) if nm else code), df.astype(float)["c"]


# ───────────────────────────── 일정 (이벤트 달력) ─────────────────────────────
# 날짜 확인: 2026-10 검색 기준. '예상'은 주최 측 공식 발표 전 관례 날짜.
# 새 일정은 저장소의 events.json 에 [{"date":"2027-05-20","name":"행사명","sectors":["반도체"]}] 형식으로 추가하면 함께 반영됩니다.
BIO = ["바이오", "헬스케어"]
TECH = ["반도체", "IT", "소프트웨어"]
EVENT_SERIES = {
    "ASCO (미국임상종양학회)": (BIO, ["2021-06-04", "2022-06-03", "2023-06-02", "2024-05-31", "2025-05-30", "2026-05-29", "2027-06-04"]),
    "AACR (미국암연구학회)": (BIO, ["2021-04-10", "2022-04-08", "2023-04-14", "2024-04-05", "2025-04-25", "2026-04-17", "2027-04-02"]),
    "ESMO (유럽종양학회)": (BIO, ["2021-09-16", "2022-09-09", "2023-10-20", "2024-09-13", "2025-10-17", "2026-10-23"]),
    "ASH (미국혈액학회)": (BIO, ["2021-12-11", "2022-12-10", "2023-12-09", "2024-12-07", "2025-12-06", "2026-12-12"]),
    "JP모건 헬스케어 콘퍼런스": (BIO, ["2021-01-11", "2022-01-10", "2023-01-09", "2024-01-08", "2025-01-13", "2026-01-12", "2027-01-11"]),
    "CES": (TECH + ["자동차"], ["2022-01-05", "2023-01-05", "2024-01-09", "2025-01-07", "2026-01-06", "2027-01-06"]),
    "엔비디아 GTC": (TECH, ["2022-03-21", "2023-03-20", "2024-03-18", "2025-03-17", "2026-03-16", "2027-03-15~"]),
    "컴퓨텍스": (TECH, ["2022-05-24", "2023-05-30", "2024-06-04", "2025-05-20", "2026-06-02"]),
    "삼성 갤럭시 언팩": (["반도체", "IT"], ["2021-01-14", "2021-08-11", "2022-02-09", "2022-08-10", "2023-02-01", "2023-07-26",
                                      "2024-01-17", "2024-07-10", "2025-01-22", "2025-07-09"]),
    "애플 신제품 발표": (["IT", "반도체"], ["2021-09-14", "2022-09-07", "2023-09-12", "2024-09-09", "2025-09-09"]),
    "미국 FOMC 금리 결정": (["*"], ["2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
                                "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
                                "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
                                "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
                                "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
                                "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
                                "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09", "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08"]),
    "한국은행 금통위": (["*", "은행", "증권", "보험"], ["2026-01-15", "2026-02-26", "2026-04-10", "2026-05-28", "2026-07-16", "2026-08-27",
                                                "2026-10-22", "2026-11-26"]),
}
EVENT_NOTE = {"엔비디아 GTC": "2027년은 예상 날짜", "한국은행 금통위": "2027년 일정은 발표되면 추가"}


def _bday_on_or_after(d):
    d = pd.Timestamp(d)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def _second_thursday(y, m):
    d = pd.Timestamp(y, m, 1)
    first_thu = d + pd.Timedelta(days=(3 - d.weekday()) % 7)
    return first_thu + pd.Timedelta(days=7)


def build_events(start_year=2021, end_year=2027):
    """[(날짜, 이름, [섹터], 예상여부)] — 검색한 일정 + 규칙으로 계산한 반복 일정 + events.json"""
    ev = []
    for name, (secs, ds) in EVENT_SERIES.items():
        for d in ds:
            ev.append((pd.Timestamp(d.rstrip("~")), name, secs, d.endswith("~")))
    for y in range(start_year, end_year + 1):
        for m in (3, 6, 9, 12):
            ev.append((_second_thursday(y, m), "선물옵션 동시만기·지수 정기변경" if m in (6, 12) else "선물옵션 동시만기", ["*"], False))
        for m in (1, 4, 7, 10):
            ev.append((_bday_on_or_after(pd.Timestamp(y, m, 7)), "실적 시즌 시작 (삼성전자 잠정실적 전후)", ["*"], False))
        for m in range(1, 13):
            ev.append((_bday_on_or_after(pd.Timestamp(y, m, 1)), "월간 수출입 동향 (반도체 수출)", ["반도체", "자동차"], False))
        last = pd.Timestamp(y, 12, 30)
        while last.weekday() >= 5:
            last -= pd.Timedelta(days=1)
        exd = last - pd.Timedelta(days=1)
        while exd.weekday() >= 5:
            exd -= pd.Timedelta(days=1)
        ev.append((exd, "연말 배당락", ["*", "은행", "보험", "증권"], False))
    path = os.path.join(BASE, "events.json")
    if os.path.exists(path):
        try:
            for x in json.load(open(path, encoding="utf-8")):
                ev.append((pd.Timestamp(x["date"]), str(x["name"]), list(x.get("sectors", ["*"])), bool(x.get("tentative", False))))
        except Exception as ex:
            print(f"  events.json 읽기 실패: {ex}", flush=True)
    return sorted(ev, key=lambda e: e[0])


def event_flags(index, events, sector):
    """각 날짜마다: 앞으로 5거래일 안에 섹터 일정 / 시장 전체 일정이 있는지, 최근 5거래일 안에 섹터 일정이 있었는지, 실적 시즌인지"""
    idx = pd.DatetimeIndex(index)
    pos = {d: i for i, d in enumerate(idx)}
    n = len(idx)
    sec_next, mkt_next, sec_prev = np.zeros(n), np.zeros(n), np.zeros(n)
    for d, name, secs, _ in events:
        if sector in secs:
            arr_n, arr_p = sec_next, sec_prev
        elif "*" in secs:
            arr_n, arr_p = mkt_next, None
        else:
            continue
        j = idx.searchsorted(d)
        if j >= n:
            continue
        arr_n[max(0, j - 5):j] = 1
        if arr_p is not None:
            arr_p[j:min(n, j + 5)] = 1
    earn = np.array([(d.month in (1, 4, 7, 10)) and 7 <= d.day <= 31 for d in idx], dtype=float)
    return pd.DataFrame(dict(ev_sec_next=sec_next, ev_mkt_next=mkt_next, ev_sec_prev=sec_prev, earn_season=earn), index=idx)


def event_study(series_map, events):
    """일정 종류 × 관련 섹터마다 과거 '일정 전 5일', '일정 후 5일' 평균 등락과 상승 비율"""
    out = {}
    for name in sorted({e[1] for e in events}):
        occ = [e for e in events if e[1] == name]
        secs = occ[0][2]
        target = next((s for s in secs if s != "*" and s in series_map), "KOSPI" if "*" in secs else None)
        c = series_map.get(target)
        if c is None:
            continue
        pre, post = [], []
        for d, _, _, _ in occ:
            j = c.index.searchsorted(d)
            if j < 6 or j + 4 >= len(c):
                continue
            pre.append((c.iloc[j - 1] / c.iloc[j - 6] - 1) * 100)
            post.append((c.iloc[j + 4] / c.iloc[j - 1] - 1) * 100)
        if len(pre) >= 3:
            out[name] = dict(target=target, n=len(pre), pre=round(float(np.mean(pre)), 2), pre_up=round(float(np.mean(np.array(pre) > 0)) * 100),
                             post=round(float(np.mean(post)), 2), post_up=round(float(np.mean(np.array(post) > 0)) * 100))
    return out


def outlook_features(c, ext):
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    r = c.pct_change()
    d = c.diff()
    au = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    ad = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    X = pd.DataFrame(dict(
        r1=r * 100, r5=(c / c.shift(5) - 1) * 100, r20=(c / c.shift(20) - 1) * 100,
        dist20=(c / ma20 - 1) * 100, dist60=(c / ma60 - 1) * 100, slope20=(ma20 / ma20.shift(5) - 1) * 100,
        rsi=100 - 100 / (1 + au / ad), volr=r.rolling(10).std() / r.rolling(60).std(),
        dd60=(c / c.rolling(60).max() - 1) * 100), index=c.index)
    for key, sr in ext.items():
        al = sr.reindex(sr.index.union(c.index)).ffill().reindex(c.index)
        if key == "vix":
            X["vix"] = al
        X[key + "5"] = (al / al.shift(5) - 1) * 100
        X[key + "20"] = (al / al.shift(20) - 1) * 100
    if "us10y" in ext and "irx" in ext:
        al = lambda sr: sr.reindex(sr.index.union(c.index)).ffill().reindex(c.index)
        X["curve"] = al(ext["us10y"]) - al(ext["irx"])                 # 장단기 금리차 (마이너스면 경기 둔화 경고)
    pos = pd.Series(np.arange(len(c)), index=c.index)
    ym = c.index.to_period("M")
    X["tom"] = (((pos - pos.groupby(ym).transform("min")) <= 2) | ((pos.groupby(ym).transform("max") - pos) <= 1)).astype(float)
    fwd = (c.shift(-5) / c - 1) * 100
    return X, fwd


def fit_logit(X, y, l2=2.0, iters=1500, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) / len(y)
        g[1:] += l2 * w[1:] / len(y)
        w -= lr * g
    return mu, sd, w


def logit_pred(model, X):
    mu, sd, w = model
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    return 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))


def market_outlook(demo, dates):
    """지수·섹터 ETF의 과거 데이터로 '5일 뒤 오를 확률'을 추정 (로지스틱 회귀, 시점 분리 검증)"""
    ext = load_external(demo, dates)
    series = []
    if demo:
        rng = np.random.default_rng(11)
        dd = pd.bdate_range(end=dates[-1], periods=1300)
        for nm, code in [("KOSPI", "KOSPI"), ("KOSDAQ", "KOSDAQ")] + SECTOR_ETF:
            series.append((nm, code, f"{nm} (예시)", pd.Series(1000 * np.exp(np.cumsum(rng.normal(0.0003, 0.013, len(dd)))), index=dd)))
    else:
        from concurrent.futures import ThreadPoolExecutor
        jobs = [("KOSPI", "KOSPI"), ("KOSDAQ", "KOSDAQ")] + SECTOR_ETF
        with ThreadPoolExecutor(max_workers=6) as ex:
            res = list(ex.map(lambda j: (j, *safe_fetch(j[1], 1300)), jobs))
        for (nm, code), etfname, c in res:
            if c is not None:
                series.append((nm, code, etfname, c))
    if len(series) < 2:
        print("  5일 전망: 지수·섹터 데이터를 받지 못해 건너뜀", flush=True)
        return None

    events = build_events(2021, 2027)
    rows = []
    for nm, code, etfname, c in series:
        X, fwd = outlook_features(c, ext)
        X = X.join(event_flags(c.index, events, nm))
        rows.append((nm, code, etfname, c, X, fwd))
    kos = next((r for r in rows if r[0] == "KOSPI"), None)
    if kos is not None:
        for r in rows:
            r[4]["rs20"] = r[4]["r20"] - kos[4]["r20"].reindex(r[4].index)   # 코스피 대비 20일 강도
    cols = sorted(set.intersection(*[set(r[4].columns) for r in rows]))
    train = pd.concat([r[4][cols].assign(fwd=r[5]) for r in rows])
    train = train.replace([np.inf, -np.inf], np.nan).dropna()
    train = train.sort_index()
    Xa, ya, fa = train[cols].to_numpy(float), (train["fwd"] > 0).to_numpy(float), train["fwd"].to_numpy(float)
    # 모델 고르기: 앞 60% 학습 → 다음 20%로 두 모델 비교 → 마지막 20%로 최종 성적 (시간 순서 유지, 사이에 10일 간격)
    ix = train.index
    c1, c2, gap = ix[int(len(ix) * 0.6)], ix[int(len(ix) * 0.8)], pd.Timedelta(days=10)
    tr, va, te = ix < c1, (ix >= c1 + gap) & (ix < c2), ix >= c2 + gap
    cands = {"로지스틱 회귀": (fit_logit, logit_pred)}
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier

        def fit_hgb(X_, y_):
            return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                                  min_samples_leaf=200, random_state=0).fit(X_, y_)
        cands["그래디언트 부스팅"] = (fit_hgb, lambda m_, X_: m_.predict_proba(X_)[:, 1])
    except Exception:
        pass
    acc = lambda p_, y_: float(((p_ > 0.5) == (y_ > 0.5)).mean())
    va_score = {k_: acc(pr_(ft_(Xa[tr], ya[tr]), Xa[va]), ya[va]) for k_, (ft_, pr_) in cands.items()}
    best = max(va_score, key=va_score.get)
    fit_, pred_ = cands[best]
    p_te = pred_(fit_(Xa[tr | va], ya[tr | va]), Xa[te])
    hit = acc(p_te, ya[te])
    base = float(max(ya[te].mean(), 1 - ya[te].mean()))
    conf = (p_te > 0.6) | (p_te < 0.4)
    hit_conf = float(((p_te[conf] > 0.5) == (ya[te][conf] > 0.5)).mean()) if conf.sum() >= 30 else None
    calib = []
    for lo_, hi_, lab in ((0, .45, "45% 미만"), (.45, .55, "45~55%"), (.55, 1.01, "55% 이상")):
        mk_ = (p_te >= lo_) & (p_te < hi_)
        if mk_.sum() >= 20:
            calib.append(dict(lab=lab, n=int(mk_.sum()), up=round(float(ya[te][mk_].mean()) * 100, 1), ret=round(float(fa[te][mk_].mean()), 2)))
    model_final = fit_(Xa, ya)
    model = fit_logit(Xa, ya)                       # 해석용 (어떤 지표를 무겁게 봤는지)
    predict = lambda X_: pred_(model_final, X_)
    FK = dict(r1="전일 등락", r5="5일 등락", r20="20일 등락", dist20="20일선과의 거리", dist60="60일선과의 거리",
              slope20="20일선 기울기", rsi="RSI", volr="변동성 확대", dd60="고점 대비 하락폭",
              rs20="코스피 대비 20일 강도", curve="미국 장단기 금리차", tom="월말·월초",
              ev_sec_next="섹터 일정 5일 이내", ev_mkt_next="시장 일정(FOMC·만기 등) 5일 이내", ev_sec_prev="섹터 일정 직후", earn_season="실적 시즌")
    def fk(c_):
        if c_ in FK:
            return FK[c_]
        for k_ in sorted(MACRO, key=len, reverse=True):
            if c_.startswith(k_):
                rest = c_[len(k_):]
                return MACRO[k_][0] + (" 수준" if rest == "" else f" {rest}일 변화")
        return c_
    wts = model[2][1:]
    top = [dict(name=fk(cols[j]), dir=("높을수록 상승 쪽" if wts[j] > 0 else "높을수록 하락 쪽")) for j in np.argsort(-np.abs(wts))[:5]]
    p_all = predict(Xa)
    bins = np.clip((p_all * 10).astype(int), 0, 9)
    exp_ret = {b: float(np.mean(fa[bins == b])) for b in range(10) if (bins == b).sum() >= 30}

    out = []
    for nm, code, etfname, c, X, fwd in rows:
        x = X[cols].replace([np.inf, -np.inf], np.nan).iloc[[-1]]
        if x.isna().any(axis=None):
            continue
        p = float(predict(x.to_numpy(float))[0])
        b = min(9, int(p * 10))
        er = exp_ret.get(b, 0.0)
        sig5 = float(c.pct_change().iloc[-20:].std() * np.sqrt(5) * 100)
        ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
        last, m20, m60, m20_5 = c.iloc[-1], ma20.iloc[-1], ma60.iloc[-1], ma20.iloc[-6]
        trend = "상승" if last > m20 > m60 and m20 > m20_5 else "하락" if last < m20 < m60 and m20 < m20_5 else "횡보"
        out.append(dict(name=nm, code=code, etf=etfname, p=round(p * 100), er=round(er, 2), band=round(sig5, 2),
                        trend=trend, r5=num((last / c.iloc[-6] - 1) * 100, 1), r20=num((last / c.iloc[-21] - 1) * 100, 1),
                        last=num(last, 2), spark=[num(x_, 2) for x_ in c.iloc[-60:]]))
    smap = {nm: c for nm, code, etfname, c, X, fwd in rows}
    SERIES.update(smap)
    study = event_study(smap, events)
    today = pd.Timestamp(datetime.now().date())
    upcoming = []
    for d, name, secs, tent in events:
        dd_ = (d - today).days
        if 0 <= dd_ <= 45:
            st = study.get(name)
            upcoming.append(dict(date=d.strftime("%m/%d"), dow="월화수목금토일"[d.weekday()], dday=dd_, name=name,
                                 secs=(["시장 전체"] if "*" in secs else []) + [x for x in secs if x != "*"], tent=tent, note=EVENT_NOTE.get(name, ""), st=st))
    for o in out:
        nxt = next((u for u in upcoming if o["name"] in u["secs"] and u["dday"] <= 20), None)
        o["next_ev"] = f"{nxt['name']} D-{nxt['dday']}" if nxt else ""
    kospi_r20 = next((o["r20"] for o in out if o["name"] == "KOSPI"), None)
    for o in out:
        o["rs"] = num(o["r20"] - kospi_r20, 1) if (kospi_r20 is not None and o["r20"] is not None) else None
    idxs = [o for o in out if o["name"] in ("KOSPI", "KOSDAQ")]
    secs = sorted([o for o in out if o["name"] not in ("KOSPI", "KOSDAQ")], key=lambda o: -o["p"])
    stat("섹터 대표 ETF·지수 장기 시세 (네이버)", len(series) > 2, f"{len(series)}개")
    print(f"  5일 전망: {best} · 섹터 {len(secs)}개 · 검증 적중률 {hit * 100:.1f}% (기본 {base * 100:.1f}%)", flush=True)
    return dict(idx=idxs, sec=secs, hit=round(hit * 100, 1), base=round(base * 100, 1),
                hit_conf=(round(hit_conf * 100, 1) if hit_conf is not None else None), n=int(len(train)),
                since=train.index[0].strftime("%Y-%m"), feats=len(cols), top=top, events=upcoming,
                model=best, va={k_: round(v_ * 100, 1) for k_, v_ in va_score.items()}, calib=calib,
                test_from=ix[te][0].strftime("%Y-%m") if te.any() else "")


def safe_fetch(code, count):
    try:
        return fetch_series(code, count)
    except Exception:
        return None, None


# ───────────────────────────── 유튜브 채널 · 급등주 해부 ─────────────────────────────
YT_CHANNEL = dict(name="한균수의 주식사용설명서", id="UCadSWH0pDXxEatvLHEHCWlg",
                  url="https://www.youtube.com/@%ED%95%9C%EA%B7%A0%EC%88%98%EC%9D%98%EC%A3%BC%EC%8B%9D%EC%82%AC%EC%9A%A9%EC%84%A4%EB%AA%85")
NAME_ALIAS = {"삼전": "삼성전자", "하닉": "SK하이닉스", "하이닉스": "SK하이닉스", "현차": "현대차", "엘전": "LG전자", "엔솔": "LG에너지솔루션",
              "삼바": "삼성바이오로직스", "한화에어로": "한화에어로스페이스", "두산에너빌": "두산에너빌리티", "포스코홀딩스": "POSCO홀딩스",
              "카뱅": "카카오뱅크", "에코비엠": "에코프로비엠", "현대로템": "현대로템", "HD현중": "HD현대중공업"}
NAME_STOP = {"대상", "한국", "미래", "동방", "선진", "대원", "국보", "진도", "광명", "세방", "부국", "서울", "신라", "조선", "성안", "상신",
             "대성", "동성", "한일", "태양", "우진", "일진", "대교", "남성", "사조", "다우", "효성", "이월", "체시스", "유진", "삼일", "현대"}


def fetch_youtube(n=15):
    import html as _h
    xml = http_get(f"https://www.youtube.com/feeds/videos.xml?channel_id={YT_CHANNEL['id']}", retries=2)
    out = []
    for e in re.findall(r"<entry>(.*?)</entry>", xml, re.S)[:n]:
        g = lambda pat: (re.search(pat, e, re.S) or [None, ""])[1]
        try:
            pub = datetime.fromisoformat(g(r"<published>(.*?)</published>").strip()).astimezone()
        except Exception:
            continue
        out.append(dict(vid=g(r"<yt:videoId>(.*?)</yt:videoId>").strip(), title=_h.unescape(g(r"<title>(.*?)</title>")).strip(),
                        desc=_h.unescape(g(r"<media:description>(.*?)</media:description>")), pub=pub,
                        views=int(g(r'<media:statistics views="(\d+)"') or 0)))
    return out


def demo_videos(names):
    nm = [v for k, v in names.items() if k.endswith("0")][:12]
    now = datetime.now().astimezone()
    return [dict(vid="dQw4w9WgXcQ", title=f"[공개방송] {nm[i]}·{nm[i + 1]} 지금 사도 될까? 시장 점검", desc=f"00:00 시황\n05:10 {nm[i]}\n12:30 {nm[i + 1]}",
                 pub=now - timedelta(days=i * 2 + 1), views=10000 + i * 1000) for i in range(0, 10, 2)]


def find_mentions(text, name2code):
    """제목·설명에서 종목명 찾기 (긴 이름 먼저, 짧은 이름은 앞뒤가 글자가 아닐 때만)"""
    found, used = [], []
    for alias, real in NAME_ALIAS.items():
        if alias in text and real in name2code:
            found.append(name2code[real])
    for nm in sorted(name2code, key=len, reverse=True):
        if len(nm) < 2 or nm in NAME_STOP:
            continue
        j = text.find(nm)
        while j >= 0:
            a, b = j, j + len(nm)
            if not any(x <= a < y or x < b <= y for x, y in used):
                if len(nm) <= 2:
                    pre, post = text[a - 1:a] or " ", text[b:b + 1] or " "
                    if re.match(r"[가-힣A-Za-z0-9]", pre) or re.match(r"[가-힣A-Za-z0-9]", post):
                        j = text.find(nm, j + 1)
                        continue
                used.append((a, b))
                found.append(name2code[nm])
            j = text.find(nm, j + 1)
    return list(dict.fromkeys(found))[:12]


def perf_since(P, t, pub, market):
    """영상 공개 후 첫 장 시가에 샀다면: 지금까지 · 5거래일 뒤 수익 (비용 차감, 같은 시장 중간값과 비교)"""
    dates = P["c"].index
    d0 = pd.Timestamp(pub.date())
    j = dates.searchsorted(d0, side="left" if pub.hour < 9 else "right")
    n = len(dates)
    if j >= n or not (P["o"].iloc[j][t] > 0):
        return None
    e = float(P["o"].iloc[j][t])
    liq = (P["a"].iloc[j] >= CFG["min_amount"]) & (P["o"].iloc[j] > 0)
    mk = pd.Series([market.get(x, "") for x in P["c"].columns], index=P["c"].columns)

    def ret(k):
        r = (float(P["c"].iloc[k][t]) / e - 1) * 100 - CFG["cost"]
        b = ((P["c"].iloc[k][liq] / P["o"].iloc[j][liq] - 1) * 100).groupby(mk[liq]).median().get(market.get(t, ""), 0.0)
        return round(r, 2), round(r - float(b), 2)
    now, now_ex = ret(n - 1)
    out = dict(entry=num(e), from_=dates[j].strftime("%m/%d"), days=int(n - j), now=now, now_ex=now_ex)
    if j + 4 < n:
        out["r5"], out["ex5"] = ret(j + 4)
    return out


def chart_read(t, ev):
    """차트만 보고 내린 짧은 판정 (유튜브 언급·급등주용)"""
    m = ev["m"]
    c, ma20, ma60, ma60_5 = (float(m[k][t]) for k in ("c", "ma20", "ma60", "ma60_5"))
    rsi, pb = float(m["rsi"][t]), float(m["pb"][t])
    trend = "상승 추세" if c > ma20 > ma60 and ma60 > ma60_5 else "하락 추세" if c < ma60 and ma60 < ma60_5 else "혼조"
    hot = rsi >= 70 or pb >= 1
    pat = str(m["pat"][t]) if m.get("pat") is not None else ""
    if trend == "하락 추세" or "이탈" in pat:
        op = ("주의", "bad")
    elif hot:
        op = ("추격 주의 · 눌림 기다리기", "mid")
    elif trend == "상승 추세":
        op = ("우리 방식도 긍정적", "ok")
    else:
        op = ("관망", "mid")
    return dict(trend=trend, hot=bool(hot), rsi=num(rsi, 1), pat=pat, op=op[0], tone=op[1])


def gainer_stats(P, I):
    """과거 2년: 하루 +25% 이상(상한가급) / +15~25% 급등 다음 날·5일 뒤 평균"""
    c, o, a = P["c"].to_numpy(float), P["o"].to_numpy(float), P["a"].to_numpy(float)
    ch = I["chg"].to_numpy(float)
    out = {}
    for lab, lo, hi in (("상한가급 (+25% 이상)", 25, 999), ("급등 (+15~25%)", 15, 25)):
        with np.errstate(invalid="ignore"):
            mask = (ch[:-5] >= lo) & (ch[:-5] < hi) & (a[:-5] >= 10e8)
        ii, jj = np.nonzero(mask)
        if len(ii) < 20:
            continue
        gap = (o[ii + 1, jj] / c[ii, jj] - 1) * 100
        d1 = (c[ii + 1, jj] / c[ii, jj] - 1) * 100
        d5 = (c[ii + 5, jj] / c[ii, jj] - 1) * 100
        ok = np.isfinite(gap) & np.isfinite(d1) & np.isfinite(d5)
        out[lab] = dict(n=int(ok.sum()), gap=round(float(np.mean(gap[ok])), 2), d1=round(float(np.mean(d1[ok])), 2),
                        up1=round(float(np.mean(d1[ok] > 0)) * 100), d5=round(float(np.mean(d5[ok])), 2), up5=round(float(np.mean(d5[ok] > 0)) * 100))
    return out


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


# ───────────────────────────── 오늘의 브리핑 ─────────────────────────────
def gnews(query, lang="ko", days=1, n=6, must=None):
    from urllib.parse import quote
    hl = "hl=ko&gl=KR&ceid=KR:ko" if lang == "ko" else "hl=en-US&gl=US&ceid=US:en"
    try:
        items = _rss(f"https://news.google.com/rss/search?q={quote(query + f' when:{days}d')}&{hl}", days=days + 1)
    except Exception:
        return []
    if must:
        items = [x for x in items if any(k in x["t"] for k in must)]
    items.sort(key=lambda x: x["dt"], reverse=True)
    out, seen = [], set()
    for x in items:
        k = re.sub(r"\W", "", x["t"])[:18]
        if k in seen:
            continue
        seen.add(k)
        out.append(dict(t=x["t"][:110], u=x["u"], s=x["s"][:24], d=x["dt"].strftime("%m/%d %H:%M"), en=int(lang != "ko")))
        if len(out) >= n:
            break
    return out


def dart_list(key, days, match, pages=5):
    """OpenDART 공시 목록 (거래소 공시) 중 제목에 match 가 들어간 것"""
    now = datetime.now()
    out = []
    for page in range(1, pages + 1):
        url = (f"https://opendart.fss.or.kr/api/list.json?crtfc_key={key}&bgn_de={(now - timedelta(days=days)):%Y%m%d}"
               f"&end_de={now:%Y%m%d}&pblntf_ty=I&page_no={page}&page_count=100")
        try:
            j = json.loads(http_get(url, retries=2))
        except Exception:
            break
        for x in j.get("list", []):
            nm = re.sub(r"\s+", " ", x.get("report_nm", "")).strip()
            if all(m in nm for m in match):
                out.append(dict(t=f"{x['corp_name']} · {nm}", u=f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={x['rcept_no']}",
                                s="DART 공시", d=f"{x['rcept_dt'][4:6]}/{x['rcept_dt'][6:]}", en=0))
        if page >= int(j.get("total_page", 1) or 1):
            break
    return out


def build_briefing(risk, outlook, macro, tabs, demo):
    now = datetime.now()
    ext = EXT_CACHE.get("data", {})

    def d1(k):
        sr = ext.get(k)
        if sr is None or len(sr) < 2:
            return None, None, None
        return float(sr.iloc[-1]), (float(sr.iloc[-1]) / float(sr.iloc[-2]) - 1) * 100, sr.index[-1].strftime("%m/%d")

    lines = []
    us = [(nm, d1(k)) for nm, k in (("S&P500", "spx"), ("나스닥", "nasdaq"), ("반도체", "sox"))]
    us = [(nm, v) for nm, v in us if v[1] is not None]
    if us:
        lines.append(dict(k="밤사이 미국", t=" · ".join(f"{nm} {v[1]:+.2f}%" for nm, v in us), tone=1 if sum(v[1] for _, v in us) > 0 else -1))
    fx = [(nm, d1(k)) for nm, k in (("원·달러", "krw"), ("WTI", "oil"), ("금", "gold"), ("VIX", "vix"))]
    fx = [(nm, v) for nm, v in fx if v[1] is not None]
    if fx:
        lines.append(dict(k="환율·원자재", t=" · ".join(f"{nm} {v[0]:,.2f} ({v[1]:+.1f}%)" if nm != "원·달러" else f"{nm} {v[0]:,.1f}원 ({v[1]:+.2f}%)" for nm, v in fx), tone=0))
    if risk:
        lines.append(dict(k="시장 신호등", t=f"하락 위험 {risk['score']}점({risk['level']}) · 상승 신호 {risk['up']}점({risk['ulevel']})",
                          tone=-1 if risk["score"] >= 50 else 1 if risk["up"] >= 35 else 0))
    if outlook and outlook.get("sec"):
        sec = outlook["sec"]
        lines.append(dict(k="5일 전망", t=f"오를 확률 높은 섹터 {', '.join(o['name'] for o in sec[:2])} · 낮은 섹터 {', '.join(o['name'] for o in sec[-2:])}", tone=0))
    evs = (outlook or {}).get("events", [])
    today_ev = [e for e in evs if e["dday"] == 0]
    week_ev = [e for e in evs if 0 <= e["dday"] <= 7]
    lines.append(dict(k="오늘 일정", t=(", ".join(e["name"] for e in today_ev) if today_ev else "큰 일정 없음")
                      + (f" · 이번 주 {len(week_ev)}건" if week_ev else ""), tone=0))
    if macro:
        g = sum(1 for x in macro if x["eff"] > 0)
        b = sum(1 for x in macro if x["eff"] < 0)
        lines.append(dict(k="글로벌 지표", t=f"주식에 유리 {g}개 · 불리 {b}개", tone=1 if g > b else -1 if b > g else 0))

    if demo:
        fake = lambda pre, n: [dict(t=f"{pre} 예시 기사 제목 {i + 1}", u="https://example.com", s="예시", d=now.strftime("%m/%d 07:0") + str(i), en=0) for i in range(n)]
        earnings, kr_news, gl_news = fake("실적 발표", 4), fake("국내 증시", 5), [dict(t=f"Wall Street example headline {i + 1}", u="https://example.com", s="Example", d=now.strftime("%m/%d"), en=1) for i in range(4)]
        sector_issues = [dict(sector="반도체", items=fake("반도체", 2)), dict(sector="바이오", items=fake("바이오", 2))]
        dart_used = False
    else:
        print("브리핑 뉴스 모으는 중…", flush=True)
        key = os.environ.get("DART_API_KEY", "").strip()
        dart_used = bool(key)
        earnings = []
        if key:
            earnings = dart_list(key, 1, ["잠정", "실적"])[:12] + dart_list(key, 10, ["기업설명회"])[:8]
        earnings += gnews("실적 발표 OR 잠정실적 OR 어닝", "ko", 1, 8, must=["실적", "어닝", "영업이익"])
        kr_news = gnews("코스피 OR 코스닥 증시 전망", "ko", 1, 6)
        gl_news = gnews("stock market Wall Street", "en", 1, 5)
        sector_issues = []
        if outlook and outlook.get("sec"):
            hot = sorted(outlook["sec"], key=lambda o: -abs(o.get("r5") or 0))
            pick = [o for o in hot if abs(o.get("r5") or 0) >= 3][:4]
            pick += [o for o in outlook["sec"] if o.get("next_ev") and o not in pick and "D-" in o["next_ev"]
                     and int(o["next_ev"].split("D-")[-1]) <= 3][:2]
            for o in pick:
                its = gnews(f"{o['name']} 관련주 주가", "ko", 2, 2)
                if its:
                    sector_issues.append(dict(sector=o["name"], r5=o.get("r5"), ev=o.get("next_ev", ""), items=its))
        print(f"  실적 {len(earnings)} · 국내 {len(kr_news)} · 해외 {len(gl_news)} · 섹터 {len(sector_issues)}", flush=True)
        stat("브리핑 뉴스 (구글 뉴스)", len(kr_news) + len(gl_news) > 0, f"실적 {len(earnings)} · 국내 {len(kr_news)} · 해외 {len(gl_news)}")
        stat("DART 공시 (선택)", True if key else None, "키 등록됨" if key else "키 미등록 (선택 사항)")
    top_picks = [dict(tab=t["name"], name=t["stocks"][0]["name"], code=t["stocks"][0]["code"], score=t["stocks"][0]["score"])
                 for t in tabs if t["stocks"]]
    quick = [dict(n=nm, chg=round(v[1], 2)) for nm, v in us] + \
            [dict(n=nm, chg=round(v[1], 2)) for nm, v in fx if nm in ("원·달러", "WTI")]
    return dict(date=now.strftime("%m월 %d일"), dow="월화수목금토일"[now.weekday()], lines=lines, week=week_ev, quick=quick,
                earnings=earnings, kr=kr_news, gl=gl_news, sectors=sector_issues, picks=top_picks, dart=dart_used)


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
    stat("종목 뉴스 (구글 뉴스·야후)", nk + ne > 0, f"국내 {nk} · 해외 {ne} / {len(stocks)}종목")
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


# ───────────────────────────── 예측 기록 · 다음 날 비교 (프로그램 점검) ─────────────────────────────
HIST_DIR = os.path.join(BASE, "history")
SERIES = {}     # 5일 전망에 쓴 지수·섹터 시세 (점검 때 실제 결과 확인용)


def make_snapshot(asof, tabs, outlook, risk, regime=None):
    picks = [dict(k=tab["k"], rank=e["rank"], code=e["code"], name=e["name"], market=e.get("market", ""), close=e["close"], score=e["score"],
                  t1=e["tg"]["t1"], stop=e["tg"]["stop"], feat=e.get("feat", {})) for tab in tabs for e in tab["stocks"][:6]]
    ol = (outlook or {})
    outs = [dict(name=o["name"], p=o["p"], er=o["er"], band=o["band"]) for o in ol.get("idx", []) + ol.get("sec", [])]
    return dict(asof=asof, made=datetime.now().strftime("%Y-%m-%d %H:%M"), picks=picks, outlook=outs, regime=regime,
                risk=(dict(score=risk["score"], up=risk["up"], level=risk["level"]) if risk else None))


def save_snapshot(snap):
    """그날 데이터 기준일 파일 하나 (같은 날 여러 번 돌면 마지막 장전 갱신으로 덮어씀). 150개 넘으면 오래된 것 삭제"""
    os.makedirs(HIST_DIR, exist_ok=True)
    with open(os.path.join(HIST_DIR, f"{snap['asof']}.json"), "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False)
    files = sorted(x for x in os.listdir(HIST_DIR) if x.endswith(".json"))
    for x in files[:-150]:
        os.remove(os.path.join(HIST_DIR, x))


def load_history(exclude_asof):
    if not os.path.isdir(HIST_DIR):
        return []
    out = []
    for x in sorted(x for x in os.listdir(HIST_DIR) if x.endswith(".json"))[-60:]:
        try:
            s = json.load(open(os.path.join(HIST_DIR, x), encoding="utf-8"))
            if s.get("asof") != exclude_asof:
                out.append(s)
        except Exception:
            pass
    return out


def demo_history(P, I, market, k_days=12):
    """미리보기용: 과거 시점의 추천을 다시 계산해 가상의 기록을 만듦"""
    n, rng, out = len(P["c"].index), np.random.default_rng(9), []
    for off in range(k_days, 0, -1):
        t = n - 1 - off
        Pt = {k: v.iloc[:t + 1] for k, v in P.items()}
        It = {k: v.iloc[:t + 1] for k, v in I.items()}
        ev = evaluate(Pt, It, market, light=True)
        c = Pt["c"].iloc[-1]
        liq = c.index[(Pt["a"].iloc[-1] >= CFG["min_amount"]) & c.notna()]
        picks = []
        for k in (1, 2, 3):
            sc = ev["scores"][f"s{k}"][liq] + sum(d[liq] for _, d, _ in ev["aux"][k])
            t1s, _, sts = ev["tg"][k]
            for r, tk in enumerate(sc.sort_values(ascending=False).head(6).index, 1):
                c0, atr = float(c[tk]), float(ev["m"]["atr"][tk])
                t1 = float(t1s[tk]) if float(t1s[tk]) > c0 * 1.01 else c0 + 1.5 * atr
                st = float(sts[tk]) if float(sts[tk]) < c0 * 0.995 else c0 - 1.5 * atr
                picks.append(dict(k=k, rank=r, code=tk, name=f"종목{tk}", close=num(c0), score=num(sc[tk]), t1=num(t1), stop=num(st)))
        outs = [dict(name=nm, p=int(rng.integers(38, 66)), er=round(float(rng.normal(0, 0.5)), 2), band=round(float(rng.uniform(2, 4)), 2)) for nm in SERIES]
        out.append(dict(asof=P["c"].index[t].strftime("%Y-%m-%d"), made="(미리보기)", picks=picks, outlook=outs,
                        risk=dict(score=int(rng.integers(20, 70)), up=int(rng.integers(5, 40)), level="주의")))
    return out


def review(hist, P, market=None):
    dates = P["c"].index
    pos = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(dates)}
    n = len(dates)
    o, c, h, l, a = P["o"], P["c"], P["h"], P["l"], P["a"]
    recs, daylog, oev, rev = [], [], [], []
    for s in hist:
        i = pos.get(s["asof"])
        if i is None or i + 1 >= n:
            continue
        liq = (a.iloc[i] >= CFG["min_amount"]) & (o.iloc[i + 1] > 0)
        mkS = pd.Series([(market or {}).get(x, "") for x in c.columns], index=c.columns)
        bench = {}
        for hh in H_LIST:
            if i + hh < n:
                rr = (c.iloc[i + hh][liq] / o.iloc[i + 1][liq] - 1) * 100
                bench[hh] = rr.groupby(mkS[liq]).median().to_dict()
                bench[hh]["_all"] = float(rr.median())
        b1 = bench[1]["_all"]
        mat = i + 5 < n
        day = []
        for pk in s.get("picks", []):
            t = pk["code"]
            if t not in c.columns:
                continue
            e = float(o.iloc[i + 1][t])
            if not (e > 0) or not (c.iloc[i + 1][t] > 0):
                continue
            j_now = min(i + 5, n - 1)
            mk_t = pk.get("market") or mkS.get(t, "")
            rec_ = dict(asof=s["asof"], date=s["asof"], k=pk["k"], rank=pk["rank"], code=t, name=pk["name"], entry=num(e),
                        regime=s.get("regime"), vb_pass=(pk.get("feat") or {}).get("vb_pass"))
            for hh in H_LIST:
                if hh in bench and c.iloc[i + hh][t] > 0:
                    g = (float(c.iloc[i + hh][t]) / e - 1) * 100 - CFG["cost"]
                    rec_[f"r{hh}"] = round(g, 2)
                    rec_[f"ex{hh}"] = round(g - bench[hh].get(mk_t, bench[hh]["_all"]), 2)
                else:
                    rec_[f"r{hh}"] = rec_[f"ex{hh}"] = None
            res = "진행"
            for j in range(i + 1, min(i + 11, n)):
                if pk.get("stop") and l.iloc[j][t] <= pk["stop"]:
                    res = "손절"
                    break
                if pk.get("t1") and h.iloc[j][t] >= pk["t1"]:
                    res = "목표"
                    break
            rec_.update(now=round((float(c.iloc[j_now][t]) / e - 1) * 100 - CFG["cost"], 2), held=j_now - i, out=res,
                        t1=pk.get("t1"), stop=pk.get("stop"))
            day.append(rec_)
        recs += day
        top = [x for x in day if x["rank"] <= 3]
        daylog.append(dict(asof=s["asof"], day=dates[i + 1].strftime("%m/%d"), n=len(top),
                           avg1=(round(float(np.mean([x["r1"] for x in top])), 2) if top else None),
                           win1=(round(float(np.mean([x["r1"] > 0 for x in top])) * 100) if top else None), b1=round(b1, 2)))
        for ob in s.get("outlook", []):
            sr = SERIES.get(ob["name"])
            if sr is None:
                continue
            k0 = sr.index.searchsorted(pd.Timestamp(s["asof"]), side="right") - 1
            if k0 < 0 or k0 + 5 >= len(sr):
                continue
            act = (float(sr.iloc[k0 + 5]) / float(sr.iloc[k0]) - 1) * 100
            oev.append(dict(asof=s["asof"], name=ob["name"], p=ob["p"], er=ob["er"], band=ob["band"], act=round(act, 2),
                            hit=bool((ob["p"] >= 50) == (act > 0)), inband=bool(abs(act - ob["er"]) <= ob["band"])))
        if s.get("risk") and "KOSPI" in SERIES:
            sr = SERIES["KOSPI"]
            k0 = sr.index.searchsorted(pd.Timestamp(s["asof"]), side="right") - 1
            if 0 <= k0 and k0 + 5 < len(sr):
                rev.append(dict(score=s["risk"]["score"], act=(float(sr.iloc[k0 + 5]) / float(sr.iloc[k0]) - 1) * 100))

    strat, regime_live = {}, {}
    for k in (1, 2, 3):
        r = [x for x in recs if x["k"] == k and x["rank"] <= 3]
        if not r:
            continue
        rr = [dict(x, **{f"r{hh}": (np.nan if x[f"r{hh}"] is None else x[f"r{hh}"]) for hh in H_LIST},
                   **{f"ex{hh}": (np.nan if x[f"ex{hh}"] is None else x[f"ex{hh}"]) for hh in H_LIST}) for x in r]
        sm = summarize_rows(rr)
        sm["watch"] = sm["n"] < CFG["min_samples"] or sm["days"] < CFG["min_days"]
        strat[str(k)] = sm
        regime_live[str(k)] = {g: summarize_rows([x for x in rr if x.get("regime") == g]) for g in ("강세", "약세")}
    vbl = [x for x in recs if x["k"] == 2 and x["rank"] <= 3 and x.get("vb_pass") is not None]
    vb_live = None
    if vbl:
        cv = lambda L: [dict(x, **{f"r{hh}": (np.nan if x[f"r{hh}"] is None else x[f"r{hh}"]) for hh in H_LIST},
                             **{f"ex{hh}": (np.nan if x[f"ex{hh}"] is None else x[f"ex{hh}"]) for hh in H_LIST}) for x in L]
        vb_live = dict(pass_=summarize_rows(cv([x for x in vbl if x["vb_pass"]])), fail=summarize_rows(cv([x for x in vbl if not x["vb_pass"]])))
    ol = None
    if oev:
        hi = [x for x in oev if x["p"] >= 55]
        lo = [x for x in oev if x["p"] < 45]
        ol = dict(n=len(oev), hit=round(float(np.mean([x["hit"] for x in oev])) * 100, 1), inband=round(float(np.mean([x["inband"] for x in oev])) * 100),
                  up_hi=(round(float(np.mean([x["act"] > 0 for x in hi])) * 100) if len(hi) >= 5 else None), n_hi=len(hi),
                  up_lo=(round(float(np.mean([x["act"] > 0 for x in lo])) * 100) if len(lo) >= 5 else None), n_lo=len(lo))
    rk = None
    if rev:
        hiR = [x["act"] for x in rev if x["score"] >= 50]
        loR = [x["act"] for x in rev if x["score"] < 25]
        rk = dict(n=len(rev), hi=(round(float(np.mean(hiR)), 2) if len(hiR) >= 3 else None), n_hi=len(hiR),
                  lo=(round(float(np.mean(loR)), 2) if len(loR) >= 3 else None), n_lo=len(loR))
    last = hist[-1]["asof"] if hist else None
    latest = [x for x in recs if x["asof"] == (daylog[-1]["asof"] if daylog else None)]
    latest_ol = [x for x in oev if x["asof"] == (oev[-1]["asof"] if oev else None)]
    return dict(n_snap=len(hist), first=(hist[0]["asof"] if hist else None), last=last, daylog=daylog[-20:], strat=strat,
                regime=regime_live, vb=vb_live, cost=CFG["cost"],
                ol=ol, rk=rk, latest=latest, latest_ol=latest_ol, recent=recs[-60:])


FIX = {
    3: "과매도 반등: 기본 조건에 '시장 신호등 경계 미만' 또는 '지수 20일선 위'를 필수로 넣고, 거래량 기준을 2배→3배로 올리는 수정",
    1: "추세 눌림목: '60일선 우상향'을 필수 조건으로 바꾸고, 조정 중 거래량 기준을 평균의 1.0배→0.8배로 강화하는 수정",
    2: "거래량 돌파: 거래량 기준 2.5배→3배, 윗꼬리 기준 −3%→−2%로 강화하고 신호등 '경계' 이상일 땐 돌파 전략을 쉬는 수정",
}


def diagnose_bt(bt, add):
    """1년 반 과거 검증(표본 수백 건) 기반 진단 — 지금 바로 믿을 수 있는 쪽"""
    if not bt:
        return
    for k_, v in bt["strat"].items():
        k, nm, h3 = int(k_), STRATS[int(k_)], v.get("h3") or {}
        if not h3:
            continue
        txt = f"{v['n']}건·{v['days']}거래일 · 3일 시장 대비 {h3['ex']:+.2f}%p (비용 차감) · t={h3['t']}"
        if v["verdict"] == "역효과":
            add("bad", f"{nm}: 과거 1년 반 검증에서 시장보다 못함", txt, FIX[k])
        elif v["verdict"] == "효과 있음":
            add("ok", f"{nm}: 과거 검증에서 시장보다 꾸준히 나음", txt, "현재 설정 유지")
        else:
            add("info", f"{nm}: 과거 검증에서 뚜렷한 우위 없음", txt, "점수 상위 종목만 추천하거나, 약한 시장에서 이 전략을 쉬는 방법 검토")
        rg = bt["regime"].get(k_, {})
        w, s_ = (rg.get("약세") or {}).get("h3"), (rg.get("강세") or {}).get("h3")
        if w and s_ and w["n"] >= 30 and w["ex"] < -0.5 and s_["ex"] > 0:
            add("warn", f"{nm}: 약세장에서만 성적이 나쁨", f"강세장 {s_['ex']:+.2f}%p ({s_['n']}건) · 약세장 {w['ex']:+.2f}%p ({w['n']}건)",
                "시장 날씨가 비·폭풍일 때 이 전략을 홈 TOP 3과 추천에서 빼는 수정")
        if v.get("stp") is not None and v["stp"] >= 50 and v["stp"] > v.get("tgt", 0):
            add("warn", f"{nm}: 손절이 목표보다 먼저 닿는 경우가 많음", f"10일 안에 손절 {v['stp']}% · 목표 {v['tgt']}%",
                "손절가를 ATR 1.5배→2배로 넓히거나, 손익비 2 이상인 종목만 추천하는 수정")
    g = bt.get("vb_grid") or []
    base = next((x for x in g if x["vmin"] == 2.0 and x["cmax"] == 30.0), None)
    good = [x for x in g if x["n"] >= 60 and (x["t"] or 0) >= 2 and (x["ex3_a"] or -1) > 0 and (x["ex3_b"] or -1) > 0]
    if base and good:
        best = max(good, key=lambda x: x["ex3"])
        if best["ex3"] - base["ex3"] >= 0.3:
            add("warn", "거래량 돌파: 기준값을 바꾸면 과거 성적이 좋아짐",
                f"지금(거래량 2배·상승 제한 없음) 3일 시장 대비 {base['ex3']:+.2f}%p → 거래량 {best['vmin']:g}배 이상·당일 상승 {best['cmax']:g}% 이하만 고르면 {best['ex3']:+.2f}%p (t={best['t']}, 앞·뒤 기간 모두 플러스)",
                f"거래량 돌파 후보에서 거래량 {best['vmin']:g}배 미만, 당일 +{best['cmax']:g}% 초과 종목을 빼는 수정")
    sh = bt.get("vb_shadow") or {}
    pa, fa = (sh.get("pass_") or {}).get("h3"), (sh.get("fail") or {}).get("h3")
    if pa and fa and pa["n"] >= 30 and fa["n"] >= 30 and pa["ex"] - fa["ex"] >= 0.5:
        add("info", "거래량 돌파 그림자 필터가 효과 있어 보임", f"필터 통과 {pa['ex']:+.2f}%p ({pa['n']}건) vs 미통과 {fa['ex']:+.2f}%p ({fa['n']}건)",
            "그림자 필터(당일 +15% 이하·종가가 고가의 95% 이상·거래량 3배 이상)를 실제 조건으로 바꾸는 수정")


def diagnose(rv, status, bt=None):
    tips = []
    add = lambda sev, title, detail, fix="": tips.append(dict(sev=sev, title=title, detail=detail, fix=fix))
    for k_, v in status.items():
        if v["ok"] is False:
            add("bad", f"데이터 수집 실패: {k_}", v["detail"], "이 출처를 다른 무료 출처로 바꾸거나 재시도 횟수를 늘리는 수정")
    diagnose_bt(bt, add)
    if not rv or rv["n_snap"] == 0:
        add("info", "실시간 기록 시작", "오늘부터 아침 예측을 저장합니다. 내일 아침부터 하루 결과, 5거래일 뒤부터 5일 결과를 비교합니다.")
        order = {"bad": 0, "warn": 1, "info": 2, "ok": 3}
        return sorted(tips, key=lambda x: order[x["sev"]])
    watch = [f"{STRATS[int(k_)]} {v['n']}건·{v['days']}일" for k_, v in rv["strat"].items() if v.get("watch")]
    if watch:
        add("info", "실시간 기록은 아직 관찰 중", " · ".join(watch) + f" (판단 기준: 전략별 {CFG['min_samples']}건·{CFG['min_days']}거래일 이상)",
            "그 전까지는 위의 '과거 1년 반 검증' 결과를 기준으로 판단하세요")
    for k_, v in rv["strat"].items():
        if v.get("watch"):
            continue
        k, nm = int(k_), STRATS[int(k_)]
        for hh in (3, 1):
            x = v.get(f"h{hh}")
            if not x or x["t"] is None:
                continue
            txt = f"{x['n']}건 · {hh}일 평균 {x['avg']:+.2f}% (비용 차감) · 시장 대비 {x['ex']:+.2f}%p · t={x['t']}"
            if x["t"] <= -2:
                add("bad", f"{nm}: 실제 기록에서 시장보다 꾸준히 못함", txt, FIX[k])
            elif x["t"] >= 2:
                add("ok", f"{nm}: 실제 기록에서도 시장보다 나음", txt, "현재 설정 유지")
            break
    ol = rv.get("ol")
    if ol and ol["n"] >= 30:
        if ol["hit"] < 50:
            add("bad", "5일 전망 방향이 자주 틀림", f"{ol['n']}건 중 적중 {ol['hit']}%",
                "입력 지표가 너무 많아 과거에 끼워 맞춰졌을 수 있습니다: 지표를 중요도 상위 15개로 줄이고 학습 기간을 최근 3년으로 줄이는 수정")
        elif ol["hit"] >= 55:
            add("ok", "5일 전망 방향 적중 양호", f"{ol['n']}건 중 적중 {ol['hit']}%", "현재 설정 유지")
        if ol["inband"] < 55:
            add("warn", "실제 움직임이 예상 흔들림 폭을 자주 벗어남", f"범위 안 {ol['inband']}% (기대치 약 68%)",
                "흔들림 폭을 과소평가했습니다: 최근 20일→10일 변동성으로 계산하거나 1.2배로 넓히는 수정")
        elif ol["inband"] > 85:
            add("info", "예상 흔들림 폭이 넉넉함", f"범위 안 {ol['inband']}% (기대치 약 68%)", "흔들림 폭을 0.9배로 좁혀도 되는지 검토")
        if ol["up_hi"] is not None and ol["up_lo"] is not None and ol["up_hi"] <= ol["up_lo"]:
            add("warn", "확률이 거꾸로 작동", f"'55% 이상' 실제 상승 {ol['up_hi']}% ≤ '45% 미만' {ol['up_lo']}%",
                "확률 숫자를 숨기고 '방향 불분명'으로만 표시하거나, 모델을 로지스틱 회귀로 고정하는 수정")
    rk = rv.get("rk")
    if rk and rk["hi"] is not None and rk["lo"] is not None:
        if rk["hi"] > rk["lo"]:
            add("warn", "시장 신호등이 반대로 작동", f"위험 50점 이상 다음 5일 {rk['hi']:+.2f}% > 25점 미만 {rk['lo']:+.2f}%",
                "분산일·VIX 점수를 줄이고 지수 이동평균·신저가 비율 점수를 늘리는 수정")
        else:
            add("ok", "시장 신호등이 방향대로 작동", f"위험 50점 이상 다음 5일 {rk['hi']:+.2f}% < 25점 미만 {rk['lo']:+.2f}%", "현재 설정 유지")
    order = {"bad": 0, "warn": 1, "info": 2, "ok": 3}
    return sorted(tips, key=lambda x: order[x["sev"]])


# ───────────────────────────── 과거 성적 검증 (백테스트, 약 1년 반 · 매 거래일) ─────────────────────────────
H_LIST = (1, 3, 5)


def _tstat(vals):
    v = np.asarray([x for x in vals if x == x], float)
    if len(v) < 5 or v.std(ddof=1) == 0:
        return None
    return float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))


def summarize_rows(rows, key_day="date"):
    """보유기간별: 건수·평균(비용 차감)·중앙값·승률·시장 대비·최대 손실·t값(날짜별 평균 기준)"""
    if not rows:
        return None
    a = pd.DataFrame(rows)
    out = dict(n=len(a), days=int(a[key_day].nunique()))
    for h in H_LIST:
        r, e = a[f"r{h}"].dropna(), a[f"ex{h}"].dropna()
        if not len(r):
            continue
        daily = a.dropna(subset=[f"ex{h}"]).groupby(key_day)[f"ex{h}"].mean()
        out[f"h{h}"] = dict(n=int(len(r)), avg=round(float(r.mean()), 2), med=round(float(r.median()), 2),
                            win=round(float((r > 0).mean() * 100)), ex=round(float(e.mean()), 2),
                            worst=round(float(r.min()), 1), t=(round(_tstat(daily.values), 2) if _tstat(daily.values) is not None else None))
    if "out" in a:
        done = a[a["out"] != "진행"]
        if len(done):
            out["tgt"] = round(float((done["out"] == "목표").mean() * 100))
            out["stp"] = round(float((done["out"] == "손절").mean() * 100))
    return out


def verdict(s, h=3):
    """t값 2 이상 + 앞·뒤 기간 모두 플러스 → 효과 있음"""
    x = (s or {}).get(f"h{h}")
    if not x or x["t"] is None:
        return "판단 보류"
    if x["t"] >= 2:
        return "효과 있음"
    if x["t"] <= -2:
        return "역효과"
    return "효과 불분명"


def backtest(P, I, market, mcap, names, asof=None, demo=False):
    """과거 매 거래일마다 '그날 이 프로그램이 골랐을 전략별 1~3위'를 다시 계산해 이후 성적을 확인.
    진입 = 다음 날 시가 · 수익 = 1/3/5거래일 뒤 종가 · 왕복 비용 차감 · 시장 대비 = 같은 시장(코스피/코스닥) 종목 중간값과 비교.
    한계: 지금 상장된 종목만으로 계산(그 사이 상장폐지된 종목 빠짐 → 결과가 실제보다 조금 좋게 나올 수 있음), 과거 수급·실적 자료 없음."""
    cache = os.path.join(HIST_DIR, "backtest_cache.json")
    if not demo and asof and os.path.exists(cache):
        try:
            cj = json.load(open(cache, encoding="utf-8"))
            if cj.get("asof") == asof and cj.get("ver") == 2:
                print("  과거 성적: 오늘 계산해 둔 결과 재사용", flush=True)
                stat("과거 성적 검증", True, f"{cj['res']['n_days']}거래일 · {cj['res']['start']}~{cj['res']['end']} (재사용)")
                return cj["res"]
        except Exception:
            pass
    dates = P["c"].index
    n = len(dates)
    cost = CFG["cost"]
    mk = pd.Series([market.get(x, "") for x in P["c"].columns], index=P["c"].columns)
    okname = pd.Series([x.endswith("0") and (x in market) and "스팩" not in str(names.get(x, "")) for x in P["c"].columns], index=P["c"].columns)
    mc_ok = mcap.reindex(P["c"].columns).fillna(np.inf) >= CFG["min_mcap"]
    VB = CFG["vb_filter"]
    rows, grid = [], []
    start = CFG["min_history"] + 5
    if demo:
        start = max(start, n - 80)
    for t in range(start, n - 1):
        Pt = {k: v.iloc[:t + 1] for k, v in P.items()}
        It = {k: v.iloc[:t + 1] for k, v in I.items()}
        try:
            ev = evaluate(Pt, It, market, light=True)
        except Exception:
            continue
        c = Pt["c"].iloc[-1]
        ok = (Pt["c"].iloc[-CFG["min_history"]:].notna().sum() >= CFG["min_history"]) & (Pt["a"].iloc[-1] >= CFG["min_amount"]) \
            & (c >= CFG["min_price"]) & (Pt["v"].iloc[-1] > 0) & okname & mc_ok
        uni = c.index[ok]
        if len(uni) < 30:
            continue
        regime = "강세" if float((c[uni] > It["ma20"].iloc[-1][uni]).mean()) >= 0.5 else "약세"
        o1 = P["o"].iloc[t + 1]
        fut = {h: (P["c"].iloc[t + h] if t + h < n else None) for h in H_LIST}
        bench = {}
        for h in H_LIST:
            if fut[h] is None:
                continue
            rr = (fut[h][uni] / o1[uni] - 1) * 100
            bench[h] = rr.groupby(mk[uni]).median().to_dict()
        m_ = ev["m"]
        h_last = Pt["h"].iloc[-1]
        H10, L10 = P["h"].iloc[t + 1:t + 11], P["l"].iloc[t + 1:t + 11]
        for k in (1, 2, 3):
            sc = (ev["scores"][f"s{k}"][uni] + sum(d[uni] for _, d, _ in ev["aux"][k])).sort_values(ascending=False)
            t1s, _, sts = ev["tg"][k]

            def rec(tk, rank):
                e = float(o1[tk])
                if not (e > 0):
                    return None
                c0, atr = float(c[tk]), float(m_["atr"][tk])
                atr = atr if np.isfinite(atr) and atr > 0 else c0 * 0.03
                tgt, stp = float(t1s[tk]), float(sts[tk])
                if not (np.isfinite(stp) and stp < c0 * 0.995):
                    stp = c0 - 1.5 * atr
                if not (np.isfinite(tgt) and tgt > c0 * 1.01):
                    tgt = c0 + 1.5 * atr
                out = "진행" if t + 10 >= n else "없음"
                for hh, ll in zip(H10[tk].to_numpy(float), L10[tk].to_numpy(float)):
                    if ll <= stp:
                        out = "손절"
                        break
                    if hh >= tgt:
                        out = "목표"
                        break
                r = dict(date=dates[t].strftime("%Y-%m-%d"), k=k, rank=rank, regime=regime, out=out,
                         vr=float(m_["vr"][tk]), chg=float(m_["chg"][tk]), cth=float(c0 / h_last[tk]) if h_last[tk] > 0 else np.nan)
                for h in H_LIST:
                    if fut[h] is None or not (fut[h][tk] > 0):
                        r[f"r{h}"] = r[f"ex{h}"] = np.nan
                        continue
                    g = (float(fut[h][tk]) / e - 1) * 100
                    r[f"r{h}"] = g - cost
                    r[f"ex{h}"] = g - cost - bench[h].get(mk[tk], 0.0)
                if k == 2:
                    r["vb_pass"] = bool(r["chg"] <= VB["max_day_change"] and r["cth"] >= VB["min_close_to_high"] and r["vr"] >= VB["min_vol_ratio"])
                return r
            for rank, tk in enumerate(sc.index[:3], 1):
                x = rec(tk, rank)
                if x:
                    rows.append(x)
            if k == 2 and 3 in fut and fut[3] is not None:      # 거래량 돌파 기준값 격자 비교 (그날 후보 30개 안에서 다시 고르기)
                cand = list(sc.index[:30])
                for vmin in (2.0, 2.5, 3.0, 4.0):
                    for cmax in (10.0, 15.0, 30.0):
                        pk = [x for x in cand if m_["vr"][x] >= vmin and m_["chg"][x] <= cmax][:3]
                        for tk in pk:
                            x = rec(tk, 0)
                            if x and x["ex3"] == x["ex3"]:
                                grid.append(dict(date=x["date"], vmin=vmin, cmax=cmax, ex3=x["ex3"], r3=x["r3"]))
    if not rows:
        stat("과거 성적 검증", False, "기간 부족")
        return None
    A = pd.DataFrame(rows)
    days_ = sorted(A["date"].unique())
    mid = days_[len(days_) // 2]
    res = dict(start=days_[0], end=days_[-1], n_days=len(days_), cost=cost, strat={}, regime={}, half={})
    for k in (1, 2, 3):
        rk = [r for r in rows if r["k"] == k]
        s = summarize_rows(rk)
        s["verdict"] = verdict(s)
        res["strat"][str(k)] = s
        res["regime"][str(k)] = {g: summarize_rows([r for r in rk if r["regime"] == g]) for g in ("강세", "약세")}
        h1 = summarize_rows([r for r in rk if r["date"] < mid])
        h2 = summarize_rows([r for r in rk if r["date"] >= mid])
        res["half"][str(k)] = dict(a=h1, b=h2, first=f"{days_[0]}~{mid}", second=f"{mid}~{days_[-1]}")
        if s["verdict"] == "효과 있음" and not ((h1 or {}).get("h3", {}).get("ex", -1) > 0 and (h2 or {}).get("h3", {}).get("ex", -1) > 0):
            s["verdict"] = "효과 불분명"     # 한쪽 기간에서만 좋았다면 우연일 수 있음
    vb = [r for r in rows if r["k"] == 2]
    res["vb_shadow"] = dict(pass_=summarize_rows([r for r in vb if r.get("vb_pass")]), fail=summarize_rows([r for r in vb if not r.get("vb_pass")]),
                            rule=VB)
    if grid:
        G = pd.DataFrame(grid)
        gt = []
        for (vmin, cmax), g in G.groupby(["vmin", "cmax"]):
            daily = g.groupby("date")["ex3"].mean()
            gt.append(dict(vmin=vmin, cmax=cmax, n=len(g), ex3=round(float(g.ex3.mean()), 2), win=round(float((g.r3 > 0).mean() * 100)),
                           t=(round(_tstat(daily.values), 2) if _tstat(daily.values) is not None else None),
                           ex3_a=round(float(g[g.date < mid].ex3.mean()), 2) if (g.date < mid).any() else None,
                           ex3_b=round(float(g[g.date >= mid].ex3.mean()), 2) if (g.date >= mid).any() else None))
        res["vb_grid"] = gt
    stat("과거 성적 검증", True, f"{len(days_)}거래일 · {days_[0]}~{days_[-1]} · {len(rows)}건")
    print("  과거 성적: " + " · ".join(f"{STRATS[k]} 3일 시장 대비 {res['strat'][str(k)]['h3']['ex']:+.2f}%p (t={res['strat'][str(k)]['h3']['t']})"
                                     for k in (1, 2, 3) if res["strat"][str(k)].get("h3")), flush=True)
    if not demo and asof:
        try:
            os.makedirs(HIST_DIR, exist_ok=True)
            json.dump(dict(asof=asof, ver=2, res=res), open(cache, "w", encoding="utf-8"), ensure_ascii=False)
        except Exception:
            pass
    return res


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
            fib=fib_levels(t, m),
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
            st = x.get("st") or 0
            aux.append(dict(label="수급 (5일)", v=(f"외국인 {man(f)} · 기관 {man(i)}" + (f" · {st}일 연속 동반 매수" if st >= 2 else "")
                                                  if f is not None else "정보 없음"),
                            d=flow_delta(f, i) + (3 if st >= 3 else 0)))
            per = x.get("per")
            aux.append(dict(label="실적", v=(f"흑자 · PER {per:.1f}" if per and per == per else "흑자")
                            if x.get("eps") is not None else "확인 불가", d=(-2 if per and per == per and per > 80 else 0)))
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
            hl = float(P["h"].iloc[-1][t]) if P["h"].iloc[-1][t] > 0 else np.nan
            vbf = CFG["vb_filter"]
            e["feat"] = dict(vr=num(m["vr"][t], 2), rsi=num(m["rsi"][t], 1), chg=num(m["chg"][t], 2),
                             gap20=num((c[t] / m["ma20"][t] - 1) * 100, 2), cth=num(c[t] / hl, 3))
            if k == 2:
                e["feat"]["vb_pass"] = bool((e["feat"]["chg"] or 0) <= vbf["max_day_change"] and (e["feat"]["cth"] or 0) >= vbf["min_close_to_high"]
                                            and (e["feat"]["vr"] or 0) >= vbf["min_vol_ratio"])
            e["prompt"] = ai_prompt(e, asof)
            if rank <= 6:
                e["series"] = series(t, tg)
            stocks.append(e)
            picks_of.setdefault(t, f"{STRATS[k]} {rank}위")
        tabs.append(dict(k=k, name=STRATS[k], stocks=stocks, loss=loss))

    # ── 유튜브 언급 종목 · 어제의 급등주를 우리 방식으로 분석 ──
    def light_entry(t, k=None):
        if k is None:
            sc_ = {kk: float(ev["scores"].loc[t, f"s{kk}"]) + sum(float(d[t]) for _, d, _ in ev["aux"][kk]) for kk in (1, 2, 3)}
            k = max(sc_, key=sc_.get)
        tg = target_for(k, t, ev, vp.get(t, (None, None))[0])
        aux = [dict(label=lb, v=safe(fn, t), d=int(d[t])) for lb, d, fn in ev["aux"][k]]
        aux.append(dict(label="손익비", v=f"1 : {tg['rr']:.1f}" if np.isfinite(tg["rr"]) else "-", d=rr_delta(tg["rr"])))
        base = float(ev["scores"].loc[t, f"s{k}"])
        adj = sum(a_["d"] for a_ in aux)
        e = dict(rank=0, code=t, name=names[t], market=market.get(t, ""), strat=k, close=num(c[t]), chg=num(m["chg"][t], 2),
                 amount=num(m["a"][t]), mcap=num(mcap.get(t)), vr=num(m["vr"][t], 2), rsi=num(m["rsi"][t], 1), pb=num(m["pb"][t], 2),
                 base=num(base), adj=int(adj), score=num(max(0.0, min(100.0, base + adj))),
                 checks=[dict(label=lb, s=num(s_[t], 2), v=safe(fn, t)) for lb, _, s_, fn in ev["checks"][k]], aux=aux,
                 tg=dict(t1=num(tg["t1"]), t2=num(tg["t2"]), stop=num(tg["stop"]), rr=num(tg["rr"], 2), t1p=num(tg["t1p"], 1),
                         t2p=num(tg["t2p"], 1), sp=num(tg["sp"], 1), note=tg["note"], stop_label=tg["stop_label"]),
                 series=series(t, tg), read=chart_read(t, ev))
        e["prompt"] = ai_prompt(e, asof)
        return e

    yt = None
    try:
        vids = demo_videos(names) if demo else fetch_youtube()
        name2code = {str(v_): k_ for k_, v_ in names.items() if k_.endswith("0") and k_ in P["c"].columns}
        ystocks, vlist, perf_rows = {}, [], []
        for v_ in vids:
            codes_ = [t for t in find_mentions(v_["title"] + "\n" + v_["desc"], name2code) if c.get(t, np.nan) == c.get(t, np.nan)]
            ments = []
            for t in codes_:
                if t not in ystocks:
                    ystocks[t] = light_entry(t)
                pf = perf_since(P, t, v_["pub"], market)
                ments.append(dict(code=t, perf=pf))
                if pf:
                    perf_rows.append(pf)
            vlist.append(dict(vid=v_["vid"], title=v_["title"], date=v_["pub"].strftime("%m/%d %H:%M"), views=v_["views"],
                              desc=v_["desc"][:300], ments=ments))
        summ = None
        if perf_rows:
            summ = dict(n=len(perf_rows), now_ex=round(float(np.mean([x["now_ex"] for x in perf_rows])), 2),
                        win=round(float(np.mean([x["now"] > 0 for x in perf_rows])) * 100),
                        n5=sum(1 for x in perf_rows if "ex5" in x),
                        ex5=(round(float(np.mean([x["ex5"] for x in perf_rows if "ex5" in x])), 2) if any("ex5" in x for x in perf_rows) else None))
        yt = dict(channel=YT_CHANNEL, videos=vlist, stocks=ystocks, summ=summ)
        stat("유튜브 채널 (RSS)", bool(vlist), f"영상 {len(vlist)}개 · 언급 종목 {len(ystocks)}개")
    except Exception as ex:
        print(f"  유튜브 읽기 실패: {ex}", flush=True)
        stat("유튜브 채널 (RSS)", False, str(ex)[:80])

    gainers = None
    try:
        lastchg = m["chg"]
        gl = [t for t in lastchg[(lastchg >= 15) & (m["a"] >= 10e8)].sort_values(ascending=False).index if t.endswith("0")][:5]
        if demo and not gl:
            gl = list(lastchg.dropna().sort_values(ascending=False).index[:3])
        gstats = gainer_stats(P, I)
        gitems = []
        for t in gl:
            e = light_entry(t)
            e["news"] = [] if demo else gnews(f'"{names[t]}"', "ko", 2, 3, must=[names[t]])
            if demo:
                e["news"] = [dict(t=f"{names[t]} 예시 기사 제목", u="https://example.com", s="예시", d="10/16 14:00", en=0)]
            gitems.append(e)
        gainers = dict(items=gitems, stats=gstats)
    except Exception as ex:
        print(f"  급등주 분석 실패: {ex}", flush=True)

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
                     picks_of.get(t, ""), num(m["rs"][t], 1),
                     num(m["fhi"][t]), num(m["flo"][t]), (1 if m["fup"][t] else 0) if bool(m["fbig"][t]) else None,
                     m["pat"][t]]

    print("과거 성적 검증 중…", flush=True)
    try:
        bt = backtest(P, I, market, mcap, names, asof=asof, demo=demo)
    except Exception as ex:
        print(f"  과거 성적 검증 실패: {ex}", flush=True)
        bt = None
    risk = market_risk(P, I, uni, demo)
    try:
        outlook = market_outlook(demo, P["c"].index)
    except Exception as ex:
        print(f"  5일 전망 계산 실패: {ex}", flush=True)
        outlook = None
    macro = macro_panel(load_external(demo, P["c"].index))
    # 예측 기록 저장 + 지난 기록과 실제 결과 비교
    try:
        past = demo_history(P, I, market) if demo else load_history(asof)
        if not demo:
            save_snapshot(make_snapshot(asof, tabs, outlook, risk, regime=("강세" if (c[uni] > m["ma20"][uni]).mean() >= 0.5 else "약세")))
        rv = review(past, P, market)
    except Exception as ex:
        print(f"  예측 기록 점검 실패: {ex}", flush=True)
        rv = None
    stat("예측 기록 (저장소 history 폴더)", rv is not None, f"{rv['n_snap']}일치 기록" if rv else "실패")
    tips = diagnose(rv, STATUS, bt)
    try:
        brief = build_briefing(risk, outlook, macro, tabs, demo)
    except Exception as ex:
        print(f"  브리핑 생성 실패: {ex}", flush=True)
        brief = None
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
        default_tab=(3 if above20 < 45 or risk["score"] >= 50 else 1),
        risk=risk, outlook=outlook, macro=macro,
        bt=bt, status=STATUS, review=rv, tips=tips, brief=brief, gen_iso=datetime.now().strftime("%Y-%m-%dT%H:%M"),
        tabs=tabs, all=allstk, news=news, yt=yt, gainers=gainers,
        names={t: str(n) for t, n in getattr(namefn, "__self__", {}).items()},
        worker=os.environ.get("AI_WORKER_URL", "").strip(),
    )


def write_html(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o)).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", payload)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    with open(os.path.join(os.path.dirname(os.path.abspath(path)), "version.json"), "w", encoding="utf-8") as f:
        json.dump(dict(generated=data.get("gen_iso"), asof=data.get("asof")), f)
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
.how{margin:44px 0 0;background:var(--surface);border:1px solid var(--line);border-radius:14px}
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
.aux li{grid-template-columns:auto minmax(0,1fr) 34px}
.aux li>span:first-child{white-space:nowrap}
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
.mine{margin-top:8px}
.risk{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:18px 20px;margin-top:4px}
.rtop{display:grid;grid-template-columns:auto minmax(0,1fr) minmax(0,1fr);gap:20px;align-items:center}
@media(max-width:880px){.rtop{grid-template-columns:1fr}.rbadges .rbadge{flex:1}}
.rbadge{display:grid;justify-items:center;padding:12px 18px;border-radius:14px;color:#fff;min-width:110px}
.rbadge.ok{background:var(--ok)} .rbadge.mid{background:var(--mid)} .rbadge.bad{background:var(--bad)}
.rbadge span{font-weight:800;font-size:15px} .rbadge b{font-size:34px;font-weight:900;line-height:1.1} .rbadge small{opacity:.85}
.ract{margin:0 0 6px;font-weight:700;font-size:14.5px}
.rmain p{margin:0 0 4px}
.rspark .rg{stroke:var(--line);stroke-dasharray:3 3} .rspark .rl{fill:none;stroke:var(--ink);stroke-width:2}
.rspark .rd.ok{fill:var(--ok)} .rspark .rd.mid{fill:var(--mid)} .rspark .rd.bad{fill:var(--bad)} .rspark .rx{fill:var(--faint);font-size:11px}
.rcols{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:14px;padding-top:14px;border-top:1px dashed var(--line)}
@media(max-width:880px){.rcols{grid-template-columns:1fr}}
.rcols h4{margin:0 0 6px;font-size:14px}
.rcols ul{list-style:none;margin:0;padding:0;display:grid;gap:4px;font-size:13px}
.rcols li{display:grid;grid-template-columns:14px 1fr auto;gap:8px;align-items:baseline;color:var(--muted)}
.rcols li.on{color:var(--ink)} .rcols li.on .ic{color:var(--bad)} .rcols .ic{font-size:11px;color:var(--no)}
.rcols li b{color:var(--bad);font-size:12px}
.rwarn{color:var(--bad);font-weight:700}
.scoreboard{margin-top:16px;background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:16px 20px}
.scgrid{display:grid;grid-template-columns:2fr 1fr 1fr;gap:12px;margin-top:12px}
@media(max-width:980px){.scgrid{grid-template-columns:1fr 1fr}.sc.wide{grid-column:1/-1}}
@media(max-width:620px){.scgrid{grid-template-columns:1fr}}
.sc{border:1px solid var(--line);border-radius:12px;padding:12px 14px;background:var(--sunk)}
.sc h4{margin:0 0 8px;font-size:14px}
.sctbl{min-width:520px;font-size:13px} .tablewrap.flat{border:0;background:transparent}
.sctbl td,.sctbl th{padding:6px 8px}
.big2{font-size:30px;font-weight:900;letter-spacing:-.02em}
.verdict2{margin:6px 0 0;font-weight:800;font-size:13px}
.scl{list-style:none;margin:0;padding:0;display:grid;gap:4px;font-size:13px}
.scl li{display:grid;grid-template-columns:64px auto 1fr;gap:8px;align-items:baseline}
.scl li b{font-size:15px}
a.q{text-decoration:none}
.hmore{display:block;margin-top:10px;padding-top:10px;border-top:1px solid var(--line);font-weight:700;color:var(--ai);text-decoration:none;font-size:13.5px}
#guideview{font-size:14px;line-height:1.7}
.gtoc{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 18px}
.gtoc a{font-size:12.5px;font-weight:700;padding:4px 10px;border-radius:999px;border:1px solid var(--line);background:var(--surface);color:var(--ink);text-decoration:none}
.gsec{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px 20px;margin-bottom:14px;scroll-margin-top:12px}
.gsec h3{margin:0 0 10px;font-size:17px}
.gsec h5{margin:14px 0 6px;font-size:14px}
.gsec p{margin:0 0 10px}
.gsec .ht{width:100%;border-collapse:collapse;font-size:12.5px;margin:0 0 10px}
.gsec .ht th,.gsec .ht td{text-align:left;white-space:normal;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
.gsec .kinds,.gsec .rules,.gsec .auxrules{margin-bottom:10px}
.revlink{margin-top:44px;background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--ink);border-radius:16px;padding:16px 20px}
.rlrow{display:grid;grid-template-columns:auto 1fr;gap:20px;align-items:center;margin-top:10px}
@media(max-width:760px){.rlrow{grid-template-columns:1fr}}
.rlrow>div{display:grid;gap:2px}.rlrow>div b{font-size:24px;font-weight:900}
.tips{list-style:none;margin:0;padding:0;display:grid;gap:8px}
.tips li{border:1px solid var(--line);border-left:4px solid var(--no);border-radius:10px;padding:10px 12px;background:var(--surface)}
.tips li.bad{border-left-color:var(--bad)} .tips li.warn{border-left-color:var(--mid)} .tips li.ok{border-left-color:var(--ok)} .tips li.info{border-left-color:var(--down)}
.tips .th{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap} .tips .th span{font-size:12px;font-weight:800;color:var(--muted)}
.tips .fix{margin-top:4px;font-size:13px;font-weight:600}
.tips.mini li{padding:6px 10px;display:flex;gap:8px;font-size:13.5px}
#reviewview{font-size:14px;line-height:1.6}
.revtbl{min-width:640px;font-size:13px} .revtbl td,.revtbl th{padding:7px 10px}
.revkpi{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:12px}
@media(max-width:820px){.revkpi{grid-template-columns:1fr 1fr}}
.revkpi>div{border:1px solid var(--line);border-radius:10px;padding:8px 10px;background:var(--sunk);display:grid}
.revkpi b{font-size:20px}
.stale{margin:14px 0 0;padding:10px 14px;border-radius:10px;background:rgba(194,65,12,.1);color:var(--bad);font-weight:700;font-size:13.5px}
.brief{margin-top:18px;background:var(--surface);border:1px solid var(--line);border-top:4px solid var(--ink);border-radius:16px;padding:18px 20px}
.bhead{display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap}
.blines{list-style:none;margin:12px 0 14px;padding:0;display:grid;grid-template-columns:1fr 1fr;gap:6px 24px;font-size:14px}
@media(max-width:820px){.blines{grid-template-columns:1fr}}
.blines li{display:grid;grid-template-columns:90px 1fr;gap:10px;align-items:baseline}
.blines li>b{font-size:12.5px;color:var(--muted)}
.bgrid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:820px){.bgrid{grid-template-columns:1fr}}
.bbox{border:1px solid var(--line);border-radius:12px;padding:12px 14px;background:var(--sunk)}
.bbox.wide{grid-column:1/-1}
.bbox h4{margin:0 0 8px;font-size:14px}
.bl{list-style:none;margin:0;padding:0;display:grid;gap:6px;font-size:13px}
.bl li{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.bl li.today{font-weight:700}
.bl a{color:var(--ink);text-decoration:none;flex:1 1 220px} .bl a:hover{text-decoration:underline}
.bl .dt{color:var(--muted);font-size:12px;min-width:68px}
.bsec{margin-bottom:8px}
.hbtns{display:flex;gap:8px}
.q{display:inline-grid;place-items:center;width:20px;height:20px;border-radius:50%;border:1.5px solid var(--faint);background:var(--surface);color:var(--muted);font-size:12px;font-weight:800;cursor:pointer;vertical-align:middle;padding:0;line-height:1}
.q:hover{border-color:var(--ink);color:var(--ink)}
h2 .q{width:22px;height:22px;margin-left:4px}
.chip{font-size:12px;font-weight:700;padding:2px 8px;border-radius:999px;background:var(--surface);border:1px solid var(--line);color:var(--muted);margin-left:6px}
.helpdlg{max-width:720px}
.helpdlg .hhead{display:flex;justify-content:space-between;align-items:center;gap:10px;position:sticky;top:0;background:var(--surface);padding-bottom:8px;border-bottom:1px solid var(--line)}
.helpdlg h4{margin:0;font-size:17px}
.helpdlg .x{background:none;border:0;font-size:18px;cursor:pointer;color:var(--muted);padding:4px 8px}
.hbody{font-size:13.5px;line-height:1.65;max-height:70vh;overflow:auto;padding-top:10px}
.hbody p{margin:0 0 10px} .hbody h5{margin:12px 0 4px;font-size:14px} .hbody ol{padding-left:20px;margin:0 0 10px}
.hbody .ht{width:100%;min-width:0;border-collapse:collapse;font-size:12.5px;margin:0 0 10px}
.hbody .ht th,.hbody .ht td{text-align:left;white-space:normal;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
.hbody .kinds,.hbody .rules,.hbody .auxrules{margin-bottom:10px}
.hbody .ht th,.hbody .ht td:first-child{white-space:nowrap}
.rbadges{display:flex;gap:10px}
.rbadge small{font-size:11px;opacity:.9}
.rbadge.okl{background:#0d9488} .rbadge.mute{background:var(--faint)}
.rspark .rl.dn{stroke:var(--bad)} .rspark .rl.up{stroke:var(--ok)}
.lgd{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:4px}
.lgd.dn{background:var(--bad)} .lgd.up{background:var(--ok)}
.rcols .upl li.on .ic,.rcols .upl li b{color:var(--ok)}
.kd{font-style:normal;font-size:10.5px;font-weight:800;color:var(--muted);border:1px solid var(--line);border-radius:4px;padding:0 4px;margin-right:6px}
.h2s{font-size:12px;font-weight:700;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:2px 8px;vertical-align:middle;margin-left:6px}
#macro{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:16px 20px}
.mx-sum{margin:0 0 10px;font-size:14px}
.mxtbl{min-width:620px}
.tr.tg{background:rgba(15,118,110,.12);color:var(--ok)} .tr.tb{background:rgba(194,65,12,.12);color:var(--bad)}
#outlook{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:18px 20px}
.olk-note{font-size:13.5px;padding:8px 12px;border-radius:10px;background:var(--sunk);margin-bottom:14px}
.olk-note.weak{border-left:4px solid var(--mid)}
.olk-idx{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:820px){.olk-idx{grid-template-columns:1fr}}
.oi{border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.oh{display:flex;gap:10px;align-items:baseline}
.oh b{font-size:16px}
.op{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;font-size:13px}
.op span{display:block;font-size:11.5px;text-align:left} .op b{font-size:17px}
.cone .cband{fill:var(--band);stroke:var(--bb);stroke-dasharray:3 3}
.cone .cl2{fill:none;stroke:var(--ink);stroke-width:1.6}
.cone .cmid{stroke-width:2;stroke-dasharray:5 3} .cone .cmid.u{stroke:var(--up)} .cone .cmid.d{stroke:var(--down)}
.cone .cnow{stroke:var(--line)}
.tr{font-size:12px;font-weight:800;padding:1px 8px;border-radius:999px;white-space:nowrap}
.tr.tu{background:rgba(212,42,60,.12);color:var(--up)} .tr.td{background:rgba(29,91,216,.12);color:var(--down)} .tr.tf{background:var(--sunk);color:var(--muted)}
.sectbl{min-width:880px}
#events .evtbl{min-width:640px}
.evtbl td{vertical-align:top;white-space:normal}
.evtbl tr.soon td{background:var(--sunk)}
.evtbl .dd{font-weight:800;color:var(--bad);font-size:12.5px}
.evtbl td,.evtbl th{text-align:left!important}
.sb{display:inline-block;font-size:11.5px;font-weight:700;padding:1px 7px;border-radius:999px;background:var(--sunk);border:1px solid var(--line);margin:1px 0}
.evp{white-space:nowrap}
.pcell{display:flex;align-items:center;gap:8px;justify-content:flex-end}
.pbar{position:relative;width:90px;height:8px;background:var(--sunk);border-radius:4px}
.pbar i{position:absolute;top:0;bottom:0;border-radius:4px} .pbar i.u{background:var(--up)} .pbar i.d{background:var(--down)}
.pbar s{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--faint)}
.ms{fill:none;stroke-width:1.5} .ms.u{stroke:var(--up)} .ms.d{stroke:var(--down)}
.fbtns{display:flex;gap:8px}
@media(max-width:760px){.fbtns{grid-column:1/-1}.fbtns .btn{flex:1}}
.mylink{display:flex;justify-content:space-between;align-items:center;margin-top:10px;padding:12px 16px;border-radius:12px;background:var(--surface);border:1px solid var(--line);color:var(--ink);text-decoration:none;font-weight:700;font-size:14px}
.mylink span{color:var(--muted);font-weight:500;font-size:13px}
.mylink:hover{border-color:var(--ink)}
[hidden]{display:none!important}
.back{display:inline-block;margin:20px 0 0;color:var(--muted);text-decoration:none;font-weight:700}
.mytbl{min-width:640px}
.mytbl tr.sel td{background:var(--sunk)}
.mytbl tbody tr{cursor:pointer}
.mytbl .vv{font-weight:800}
.mytbl .vv.ok{color:var(--ok)} .mytbl .vv.mid{color:var(--mid)} .mytbl .vv.no{color:var(--bad)}
.hold .acts{display:flex;gap:6px;align-items:center}
.btn.sm{padding:6px 12px;font-size:13px}
.rechead{margin-top:44px;padding-top:18px;border-top:2px solid var(--ink)}
.form{display:grid;grid-template-columns:minmax(0,2fr) minmax(0,1fr) minmax(0,1fr) auto;gap:10px;align-items:end;background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px}
.form label{display:grid;gap:4px;font-size:12.5px;color:var(--muted);font-weight:600}
.form input{font:inherit;font-size:15px;padding:10px 12px;border:1.5px solid var(--line);border-radius:10px;background:var(--sunk);color:var(--ink);width:100%}
.form .err{grid-column:1/-1;color:var(--bad);font-size:13px;margin:0}
@media(max-width:760px){.form{grid-template-columns:1fr 1fr}.form label:first-child{grid-column:1/-1}}
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
.chart .fib{stroke:#a16207;stroke-width:1;stroke-dasharray:1 3;opacity:.9}
.chart .fibx{fill:#a16207;font-size:10.5px;font-weight:700}
@media (prefers-color-scheme: dark){.chart .fib{stroke:#facc15}.chart .fibx{fill:#facc15}}
.chart .fibx,.chart .tlx,.chart .slx,.chart .poctx{paint-order:stroke;stroke:var(--surface);stroke-width:3.5px;stroke-linejoin:round}
.chart .tl{stroke:var(--up);stroke-width:1.2;stroke-dasharray:6 4}
.chart .sl{stroke:var(--down);stroke-width:1.2;stroke-dasharray:6 4}
.chart .tlx{fill:var(--up);font-size:11px;font-weight:700}
.chart .slx{fill:var(--down);font-size:11px;font-weight:700}
.lvbar .ax{stroke:var(--line);stroke-width:2}
.lvbar text{font-size:11px;font-weight:700}

/* ════════ 새 화면 (앱처럼: 홈 · 추천 · 내 종목 · 시장 · 더보기) ════════ */
html{-webkit-text-size-adjust:100%}
body{overflow-x:hidden}
.wrap{max-width:1180px;padding:0 16px calc(84px + env(safe-area-inset-bottom,0px))}
.wrap *{min-width:0}
header.top{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:12px;padding:10px 0;background:var(--bg);border-bottom:1px solid var(--line);margin:0 0 14px}
.brand{display:flex;flex-direction:column;text-decoration:none;color:var(--ink);flex:1}
.brand b{font-size:18px;font-weight:900;letter-spacing:-.02em;white-space:nowrap}
.brand .meta{margin:0;font-size:12px}
.iconbtn{width:40px;height:40px;border-radius:12px;border:1px solid var(--line);background:var(--surface);color:var(--ink);font-size:20px;cursor:pointer;flex:none}
.tabbar{display:flex;gap:4px}
.tabbar a{display:flex;align-items:center;gap:6px;padding:8px 12px;border-radius:10px;color:var(--muted);text-decoration:none;font-weight:800;font-size:14px}
.tabbar a i{font-style:normal}
.tabbar a.on{background:var(--ink);color:var(--surface)}
@media(max-width:899px){
  .tabbar{position:fixed;left:0;right:0;bottom:0;z-index:30;justify-content:space-around;gap:0;background:var(--surface);border-top:1px solid var(--line);
    padding:6px 4px calc(6px + env(safe-area-inset-bottom,0px));box-shadow:0 -4px 16px rgba(0,0,0,.06)}
  .tabbar a{flex-direction:column;gap:2px;padding:4px 2px;font-size:11px;flex:1;border-radius:10px}
  .tabbar a i{font-size:20px;line-height:1.1}
  .tabbar a.on{background:transparent;color:var(--ink)}
  .tabbar a.on i{transform:scale(1.12)}
}
.view h2{margin:4px 0 10px}
.vh{display:flex;align-items:center;justify-content:space-between}
.demo,.stale{margin:0 0 12px}
footer{margin:28px 0 0;padding-top:14px;border-top:1px solid var(--line);color:var(--faint);font-size:12px}

/* 홈 */
.hgrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
@media(max-width:720px){.hgrid{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:14px 16px}
.card.wide,.card.wx,.card.strip{grid-column:1/-1}
.card h3{margin:0;font-size:15px;font-weight:800}
.ch{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:10px}
.more{display:inline-block;margin-top:8px;font-size:13px;font-weight:700;color:var(--muted);text-decoration:none}
.ch .more{margin:0}
.wx{display:grid;grid-template-columns:auto 1fr auto;gap:14px;align-items:center;text-decoration:none;color:var(--ink);border-width:2px}
.wx.ok{border-color:var(--ok)} .wx.mid{border-color:var(--mid)} .wx.bad{border-color:var(--bad)}
.wxi{font-size:clamp(40px,11vw,60px);line-height:1}
.wxb{display:grid;gap:2px} .wxb small{color:var(--muted);font-weight:700;font-size:12px}
.wxb b{font-size:clamp(22px,6vw,30px);font-weight:900;letter-spacing:-.02em} .wxb span{font-size:14px;font-weight:600}
.wx.ok .wxb b{color:var(--ok)} .wx.mid .wxb b{color:var(--mid)} .wx.bad .wxb b{color:var(--bad)}
.wxs{display:grid;grid-template-columns:auto auto;gap:0 8px;align-items:baseline;font-size:12px;color:var(--muted)} .wxs b{font-size:18px;color:var(--ink);text-align:right}
@media(max-width:420px){.wx{grid-template-columns:auto 1fr}.wxs{grid-column:1/-1;grid-template-columns:repeat(4,auto);justify-content:start}}
.strip{display:flex;gap:8px;overflow-x:auto;padding:10px 12px;scrollbar-width:none}
.strip>div{flex:1 0 auto;display:grid;text-align:center;padding:4px 8px;min-width:72px}
.strip small{color:var(--muted);font-size:11.5px;font-weight:700} .strip b{font-size:16px}
.hlist{display:grid;gap:6px}
.hrow{display:grid;grid-template-columns:1fr auto auto;gap:10px;align-items:center;width:100%;text-align:left;font:inherit;color:var(--ink);background:var(--sunk);border:0;border-radius:12px;padding:10px 12px;cursor:pointer}
.chip{font-size:11.5px;font-weight:800;padding:2px 8px;border-radius:999px;white-space:nowrap;background:var(--sunk);color:var(--muted)}
.chip.ok{background:rgba(15,118,110,.14);color:var(--ok)} .chip.mid{background:rgba(183,121,31,.15);color:var(--mid)} .chip.no{background:rgba(194,65,12,.14);color:var(--bad)}
.evl{list-style:none;margin:0;padding:0;display:grid;gap:6px;font-size:14px}
.evl li{display:grid;grid-template-columns:48px 1fr auto;gap:8px;align-items:baseline}
.evl .dd{font-weight:900;color:var(--bad);font-size:13px}
.heat{display:grid;grid-template-columns:repeat(auto-fill,minmax(92px,1fr));gap:6px}
.ht2{border-radius:10px;padding:8px 8px;display:grid;gap:2px;color:var(--ink)}
.ht2 b{font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis} .ht2 span{font-size:13px;font-weight:800}

/* 추천 목록 */
.plist{display:grid;gap:6px}
.prow{display:grid;grid-template-columns:28px minmax(0,1fr) 72px 58px 58px 64px;gap:10px;align-items:center;width:100%;text-align:left;font:inherit;color:var(--ink);
  background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:10px 12px;cursor:pointer}
.card .prow{background:var(--sunk);border:0;grid-template-columns:28px minmax(0,1fr) 58px 58px}
.prow:hover{border-color:var(--ink)}
.prow .rk{display:grid;place-items:center;width:26px;height:26px;border-radius:8px;background:var(--sunk);font-weight:900;font-size:13px}
.prow.top .rk{background:var(--ink);color:var(--surface)}
.pn{display:grid;gap:1px} .pn b{font-size:15px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis} .pn small{font-size:12.5px;color:var(--muted)}
.pt{display:grid;text-align:right} .pt small{font-size:11px;color:var(--muted)} .pt b{font-size:14px}
.ps{position:relative;height:22px;border-radius:6px;background:var(--sunk);overflow:hidden;display:grid;place-items:center}
.ps i{position:absolute;left:0;top:0;bottom:0;background:var(--line)} .ps b{position:relative;font-size:12.5px}
.sp svg{width:72px;height:24px}
@media(max-width:620px){.prow{grid-template-columns:26px minmax(0,1fr) 54px 54px}.prow .sp,.prow .ps{display:none}}
@media(max-width:340px){.prow,.card .prow{grid-template-columns:24px minmax(0,1fr) 50px;gap:6px;padding:9px 8px}.prow .pt:last-of-type{display:none}.pn b{font-size:14px}}
.tabs{margin:0;gap:2px}
.tab{font-size:15px;padding:10px 12px 8px}
.tabdesc{margin:8px 0 12px}

/* 내 종목 */
.form{padding:12px}
@media(max-width:420px){.form{grid-template-columns:1fr 1fr;gap:8px}.form label:first-child{grid-column:1/-1}.fbtns{grid-column:1/-1}.fbtns .btn{flex:1}}
.fbtns .btn{white-space:nowrap}
.msum{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;margin:14px 0 8px;font-size:13px;color:var(--muted)} .msum b{font-size:16px}
.mlist{display:grid;gap:6px}
.mrow{display:grid;grid-template-columns:1fr auto;gap:6px;align-items:stretch}
.mmain{display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:10px;align-items:center;text-align:left;font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:10px 12px;cursor:pointer}
.mpl{display:grid;text-align:right} .mpl b{font-size:15px} .mpl small{font-size:11.5px}
.del{width:44px;border-radius:14px;border:1px solid var(--line);background:var(--surface);color:var(--muted);font-size:16px;cursor:pointer}
.del:hover{color:var(--bad);border-color:var(--bad)}
@media(max-width:400px){.mmain{grid-template-columns:minmax(0,1fr) auto}.mmain .chip{grid-column:1/-1;justify-self:start}}
.btn.danger{color:var(--bad);border-color:var(--bad)}

/* 시장 (접는 상자) */
.fold{background:var(--surface);border:1px solid var(--line);border-radius:16px;margin-bottom:10px;overflow:hidden}
.fold>summary{list-style:none;cursor:pointer;padding:14px 16px;font-weight:800;font-size:16px;display:flex;align-items:center;gap:8px}
.fold>summary::-webkit-details-marker{display:none}
.fold>summary::after{content:"▾";margin-left:auto;color:var(--muted);transition:transform .2s}
.fold[open]>summary::after{transform:rotate(180deg)}
.fold>section{border:0;border-radius:0;margin:0;padding:0 16px 16px;background:transparent}
.fold .risk,.fold #outlook,.fold #macro{border:0;padding:0 16px 16px}
.bgrid{grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr))}

/* 더보기 */
.morelink{display:block;margin-top:12px;padding:14px 16px;border-radius:16px;background:var(--surface);border:1px solid var(--line);color:var(--ink);font-weight:800;text-decoration:none}
.scoreboard,.revlink{margin-top:0;margin-bottom:12px}

/* 자세히 보기 시트 (휴대폰: 아래에서 올라오는 전체 화면 / PC: 가운데 창) */
.sheet{padding:0;border:0;background:transparent;max-width:none;max-height:none;width:100%;height:100%;margin:0}
.sheet::backdrop{background:rgba(0,0,0,.45)}
.shin{position:absolute;left:50%;transform:translateX(-50%);bottom:0;width:min(1000px,100%);max-height:92dvh;max-height:92vh;overflow:auto;background:var(--bg);
  border-radius:20px 20px 0 0;padding:0 14px calc(18px + env(safe-area-inset-bottom,0px));overscroll-behavior:contain}
@media(min-width:900px){.shin{top:4vh;bottom:auto;border-radius:20px;max-height:92vh}}
.shbar{position:sticky;top:0;z-index:2;display:flex;justify-content:center;align-items:center;padding:10px 0 8px;background:var(--bg)}
.grab{width:44px;height:5px;border-radius:3px;background:var(--line)}
.shbar .x{position:absolute;right:0;top:6px;width:36px;height:36px;border-radius:50%;border:1px solid var(--line);background:var(--surface);color:var(--ink);font-size:15px;cursor:pointer}
.sheet .pick{margin:0}
.sheet .pick .big{display:none}
.mini-fold{margin-top:12px}
.mini-fold>summary{cursor:pointer;font-weight:700;font-size:13px;color:var(--muted);padding:6px 0}
.sheet .hold{margin-top:0}

/* 표가 화면보다 넓으면 표 안에서만 옆으로 밀기 */
.gsec .ht,.hbody .ht{display:block;overflow-x:auto;max-width:100%}
.gsec{overflow-wrap:anywhere}

/* 아주 좁은 화면 (폴드 커버 등) */
@media(max-width:360px){
  body{font-size:13.5px}
  .wrap{padding-left:10px;padding-right:10px}
  .brand b{font-size:16px}
  .card{padding:12px}
  .price{font-size:22px} .name{font-size:19px}
  .tgbox{grid-template-columns:1fr 1fr}.tgbox>div:nth-child(3){grid-column:1/-1}
  .kv{grid-template-columns:1fr 1fr}
  .pos-grid{grid-template-columns:1fr}
  .rbadges{flex-direction:column}
  .revkpi,.olk-idx .op{grid-template-columns:1fr 1fr}
}
@media(min-width:900px){.wrap{padding-bottom:40px}}

/* 읽기 쉬운 결론 */
.story{margin:12px 0;padding:12px 14px;border-radius:14px;background:var(--surface);border:1px solid var(--line)}
.shead{margin:0 0 8px;font-weight:800;font-size:15px;line-height:1.45}
.sblk{margin-top:8px}.sblk>b{display:block;font-size:12.5px;color:var(--muted);margin-bottom:3px}
.sblk ul{margin:0;padding-left:18px;font-size:13.5px;display:grid;gap:2px}.sblk p{margin:0;font-size:13.5px}
.sblk.warn>b{color:var(--bad)}
.story .src{margin:10px 0 0;font-size:11.5px;color:var(--faint)}
/* 유튜브 */
.ytrow{display:grid;grid-template-columns:120px 1fr;gap:10px;align-items:center;width:100%;text-align:left;font:inherit;color:var(--ink);background:none;border:0;padding:0;cursor:pointer}
.ytrow img{width:120px;aspect-ratio:16/9;object-fit:cover;border-radius:10px;background:var(--sunk)}
.ytrow b{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;font-size:14px;line-height:1.35}
.ytrow small{display:block;margin-top:2px;font-size:12px}
.ychips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.ych{display:inline-flex;gap:6px;align-items:baseline;font:inherit;font-size:12.5px;padding:4px 10px;border-radius:999px;border:1px solid var(--line);background:var(--sunk);color:var(--ink);cursor:pointer}
.ych small{font-size:11px;font-weight:800}.ych.ok small{color:var(--ok)}.ych.mid small{color:var(--mid)}.ych.bad small{color:var(--bad)}
.ylist{display:grid;gap:10px;margin-top:12px}
.ysum{display:flex;gap:24px;flex-wrap:wrap}.ysum>div{display:grid}.ysum small{color:var(--muted);font-weight:700;font-size:12px}.ysum b{font-size:24px}
.yframe{position:relative;width:100%;aspect-ratio:16/9;border-radius:14px;overflow:hidden;background:#000}
.yframe iframe{position:absolute;inset:0;width:100%;height:100%;border:0}
.ytv h3{margin:12px 0 2px;font-size:17px}
.ytab{text-decoration:none;color:var(--muted)}
/* 급등주 */
.gstat{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:8px;margin-bottom:12px}
.gstat>div{display:grid;padding:10px 12px;border-radius:12px;background:var(--sunk)}.gstat small{color:var(--muted);font-weight:700;font-size:12px}.gstat b{font-size:20px}
.gcard{display:grid;grid-template-columns:44px 1fr;gap:10px;padding:14px 0;border-top:1px solid var(--line)}
.gno{font-size:28px;font-weight:900;color:var(--line);line-height:1}
.gcard h4{margin:0;font-size:17px}.gcard .price{font-size:20px;margin:2px 0 6px}
.glink{display:grid;gap:2px;text-decoration:none;color:var(--ink);font-weight:800}
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <a class="brand" href="#home"><b>오늘의 종목 발굴</b><span class="meta tn" id="meta"></span></a>
    <nav class="tabbar" id="tabbar" aria-label="화면 이동">
      <a href="#home" data-v="home"><i aria-hidden="true">🏠</i><span>홈</span></a>
      <a href="#picks" data-v="picks"><i aria-hidden="true">⭐</i><span>추천</span></a>
      <a href="#mine" data-v="mine"><i aria-hidden="true">💼</i><span>내 종목</span></a>
      <a href="#market" data-v="market"><i aria-hidden="true">📈</i><span>시장</span></a>
      <a href="#more" data-v="more"><i aria-hidden="true">☰</i><span>더보기</span></a>
    </nav>
    <button class="iconbtn" id="refresh" aria-label="새로고침">↻</button>
  </header>
  <div class="demo" id="demo" hidden>가상 데이터로 만든 미리보기입니다. 실제 종목이 아닙니다.</div>
  <div class="stale" id="stale" hidden></div>

  <section class="view" id="v-home"></section>

  <section class="view" id="v-picks" hidden>
    <div class="vh"><h2>오늘의 추천 <button class="q" data-help="picks" aria-label="설명 보기">?</button></h2></div>
    <div class="tabs" role="tablist" id="tabs"></div>
    <p class="tabdesc" id="tabdesc"></p>
    <div id="panel"></div>
  </section>

  <section class="view" id="v-mine" hidden>
    <div class="vh"><h2>내 종목 <button class="q" data-help="mine" aria-label="설명 보기">?</button></h2></div>
    <div class="form">
      <label>종목명 또는 코드<input id="fStock" list="stocklist" placeholder="예: 삼성전자" autocomplete="off"></label>
      <label>수량 (주)<input id="fQty" type="number" inputmode="numeric" min="1" placeholder="100"></label>
      <label>평균 매입가 (원)<input id="fAvg" type="number" inputmode="numeric" min="1" placeholder="71000"></label>
      <span class="fbtns"><button class="btn" id="fCheck">점검</button><button class="btn ghost" id="fSave">저장</button></span>
      <p class="err" id="fErr" hidden></p>
    </div>
    <datalist id="stocklist"></datalist>
    <div id="mylist"></div>
  </section>

  <section class="view" id="v-market" hidden>
    <details class="fold" open><summary>🚦 시장 신호등 <button class="q" data-help="signals" aria-label="설명 보기">?</button></summary><section class="risk" id="risk"></section></details>
    <details class="fold"><summary>🔮 5일 전망 · 섹터 <button class="q" data-help="outlook" aria-label="설명 보기">?</button></summary><section id="outlook"></section></details>
    <details class="fold"><summary>🗓 다가오는 일정 <button class="q" data-help="events" aria-label="설명 보기">?</button></summary><section id="events"></section></details>
    <details class="fold"><summary>🌎 글로벌 지표 <button class="q" data-help="macro" aria-label="설명 보기">?</button></summary><section id="macro"></section></details>
    <details class="fold"><summary>🚀 어제의 급등주 해부 <button class="q" data-help="gainers" aria-label="설명 보기">?</button></summary><section id="gainers"></section></details>
    <details class="fold"><summary>📰 오늘의 뉴스 <button class="q" data-help="brief" aria-label="설명 보기">?</button></summary><section id="brief"></section></details>
  </section>

  <section class="view" id="v-yt" hidden></section>

  <section class="view" id="v-more" hidden>
    <section class="scoreboard" id="score"></section>
    <section class="revlink" id="revlink"></section>
    <a class="morelink" href="#yt">📺 한균수의 주식사용설명서 · 언급 종목 분석 →</a>
    <a class="morelink" href="#guide">📖 상세 설명 · 계산 방법 보기 →</a>
  </section>

  <div class="view" id="guideview" hidden></div>
  <div class="view" id="reviewview" hidden></div>
  <footer>참고 자료이며 수익을 보장하지 않습니다. 투자 판단과 책임은 본인에게 있습니다.</footer>
</div>

<dialog id="sheet" class="sheet" aria-label="자세히 보기"><div class="shin">
  <div class="shbar"><span class="grab" aria-hidden="true"></span><button class="x" id="sheetClose" aria-label="닫기">✕</button></div>
  <div id="sheetBody"></div>
</div></dialog>

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
<dialog id="help" class="helpdlg"><div class="in">
  <div class="hhead"><h4 id="helpTitle"></h4><button class="x" id="helpClose" aria-label="닫기">✕</button></div>
  <div id="helpBody" class="hbody"></div>
  <a class="hmore" href="#guide" id="helpMore">📖 상세 설명 페이지에서 전체 보기 →</a>
</div></dialog>

<script>
const D = __DATA__;
const STRAT = {1:{n:'추세 눌림목',c:'s1',v:'--s1',d:'상승 추세 중 20일선까지 쉬어 내려온 종목'},
               2:{n:'거래량 돌파',c:'s2',v:'--s2',d:'거래량과 함께 박스권을 뚫고 나온 종목'},
               3:{n:'과매도 반등',c:'s3',v:'--s3',d:'볼린저 하단을 찍고 반등을 시작한 종목'}};
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
  const W=Math.max(240,box.clientWidth),small=W<560;
  const m={l:4,r:small?58:66,t:24,b:20},hP=small?220:300,hV=56,hM=50,hR=70,g=14;
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
  if(!small&&!(s.fib||[]).length)tx(svg,m.l+pw-vpw-4,YP(lo+(poc+.5)*bh)+4,'매물대 최대','poctx','end');
  // 볼린저 · 60일선
  const up=[],dn=[];
  s.bbu.forEach((v,i)=>{if(v!=null&&s.bbl[i]!=null){up.push(X(i).toFixed(1)+','+YP(v).toFixed(1));dn.unshift(X(i).toFixed(1)+','+YP(s.bbl[i]).toFixed(1));}});
  if(up.length)el('path',{d:'M'+up.join('L')+'L'+dn.join('L')+'Z',class:'band'},svg);
  path(svg,s.bbu,X,YP,'bb');path(svg,s.bbl,X,YP,'bb');path(svg,s.bbm,X,YP,'bbm');path(svg,s.ma60,X,YP,'ma');
  // 피보나치 되돌림
  (s.fib||[]).forEach(([lab,v])=>{ if(v==null||v<lo||v>hi)return; const y=YP(v);
    el('line',{x1:m.l,x2:m.l+pw,y1:y,y2:y,class:'fib'},svg); if(!small)tx(svg,m.l+pw-vpw-6,y-3,'Fib '+lab+' '+f0(v),'fibx','end'); });
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
  const k=60,W=Math.max(200,box.clientWidth),H=96,m=6;
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
  return `<div class="news"><div class="lbl" style="margin:0 0 4px">📰 최근 한 달 뉴스 <button class="q" data-help="news" aria-label="설명 보기">?</button></div><ul>${n.map(x=>
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
  return `<div class="aibox aibtn"><p><b style="color:var(--ink)">🤖 AI 분석</b> <button class="q" data-help="ai" aria-label="설명 보기">?</button><br>열리면 <b>Enter</b>만 누르세요</p>
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
function helpHow(){
  const c=D.cfg;
  const rule=k=>`<div class="rule"><h5 class="${STRAT[k].c}">${STRAT[k].n}</h5>
    <ol>${D.rules[k].map(([l,w])=>`<li>${esc(l)} <span>${w}점</span></li>`).join('')}</ol>
    <div class="tg">1차 목표: ${D.targets[k][0]}<br>2차 목표: ${D.targets[k][1]}<br>손절: ${D.targets[k][2]}</div></div>`;
  return `
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
        <div><b>세력 패턴</b> (최근 20일): 장대양봉(+8%·거래량 3배) 뒤 거래량 줄며 눌림 +5, 좁은 박스권에서 OBV·거래량 증가(조용한 매집) +4, 저점권 대량 거래 짧은 몸통 캔들(매집봉) +4, 대량 장대음봉·장대양봉 중간 이탈(이탈 의심) −6 · 수급은 외국인·기관 3일 이상 연속 동반 매수면 +3 추가</div>
        <div><b>피보나치</b> (최근 120일 고점·저점): 눌림목은 상승폭의 50~61.8% 되돌림 +5, 38.2~50% +3, 78.6% 넘게 빠지면 −3 · 반등은 61.8~78.6% 깊은 지지 +5 · 차트에 노란 점선으로 표시</div>
      </div></div>
    <div><h4>앞으로 5일 전망</h4><p>코스피·코스닥과 섹터 대표 ETF 20개의 과거 약 5년치 데이터로 '5일 뒤 올랐는지'를 학습한 확률 모델(로지스틱 회귀)입니다. 최근 수익률, 이동평균과의 거리, RSI, 변동성, 고점 대비 하락폭, VIX·환율·나스닥·반도체 지수 변화를 입력으로 씁니다. 학습에 쓰지 않은 최근 기간에서 맞힌 비율을 함께 보여드리니, 그 숫자가 '찍기'보다 크게 높지 않으면 방향보다 예상 흔들림 폭을 참고하세요.</p></div>
    <div><h4>시장 방향 신호등</h4><p>글로벌 지표 15개(VIX·원달러 환율·달러인덱스·미국 10년 금리·국제유가·금·구리·하이일드 채권·S&P500·나스닥·반도체·닛케이·상해·대만·비트코인)의 20일 변화로 주식에 유리·불리를 판단해 함께 반영합니다. 하락 위험 점수와 함께 상승 신호 점수(오르는 종목 급증, 바닥 확인일, 신고가 증가, VIX 진정, 환율 안정, 지수 20일선 회복 등)도 계산합니다. 코스피·코스닥 지수, 전 종목의 20일선 위 비율과 60일 신저가 비율, 미국 공포지수(VIX)·원달러 환율·나스닥·반도체 지수로 0~100점을 계산합니다. 하락 전에 먼저 나타나는 <b>선행 신호</b>(분산일 누적, 지수와 종목 수의 괴리, 변동성 확대, VIX·환율 급등)와 이미 하락 중일 때 켜지는 <b>진행 신호</b>(지수 이동평균 이탈, 신저가 급증 등)로 나눠 보여줍니다. 50점 이상이면 추천 종목 위에 경고가 붙고, 과매도 반등 탭을 먼저 보여드립니다.</p></div>
    <div><h4>4단계 · 순위</h4><p>최종 점수 = 기본 점수 + 가감점 (0~100점). 전략별로 1~3위는 매수 후보, 4~6위는 지켜볼 종목, 7~16위는 목록으로 보여줍니다. 데이터는 네이버 금융 기준이며 평일 아침마다 전 거래일 종가로 다시 계산합니다.</p></div>`;
}

/* ───── 도움말 (물음표) ───── */
const T=(rows,head)=>`<table class="ht"><thead><tr>${head.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(c=>`<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
const RULE=k=>T(D.rules[k].map(([l,w])=>[esc(l),w+'점']),['조건','배점'])+`<p>조건을 모두 채우면 배점 전부, 벗어나면 벗어난 정도만큼 점수가 줄어듭니다 (예: RSI 40~55 조건에서 RSI 60이면 10만큼 허용 범위 중 5를 벗어나 절반 점수).</p>
  <p><b>목표·손절</b> · 1차: ${D.targets[k][0]} · 2차: ${D.targets[k][1]} · 손절: ${D.targets[k][2]}</p>`;
const HELP={
  how:['이용 안내',()=>helpHow()],
  mine:['내 종목',()=>`<ol><li>종목명(또는 6자리 코드), 보유 수량, 평균 매입가를 입력합니다.</li><li><b>점검</b>: 한 번만 보고 저장하지 않습니다.</li><li><b>저장</b>: 아래 목록에 모아 두고 매일 새 데이터로 다시 점검합니다. 홈 화면에도 나옵니다.</li><li><b>삭제</b>: 목록 오른쪽 <b>✕</b>를 누르거나, 종목을 눌러 열린 화면의 <b>🗑 삭제</b>를 누릅니다.</li><li><b>수량·평단 수정</b>: 같은 종목을 새 값으로 다시 저장하면 덮어씁니다.</li></ol><p>저장 내용은 이 휴대폰(브라우저)에만 남고 서버로 가지 않습니다. 다른 기기에서는 따로 저장해야 합니다.</p>`],
  hold:['차트 기준 전략은 이렇게 정합니다',()=>T([
      ['손절 검토 / 이익 보호 매도','주가 &lt; 60일선, 60일선 하락 중, MACD 히스토그램 음수이며 하락 확대'],
      ['일부 익절 (1/3)','RSI ≥ 70 또는 볼린저 상단 돌파 + 수익률 ≥ 10%'],
      ['보유 (추세 유지)','주가 &gt; 20일선 &gt; 60일선, 60일선 상승 중'],
      ['보유 (반등 대기)','RSI ≤ 35 + MACD 히스토그램 반등'],
      ['손절 검토','60일선 아래 + 손실 −10% 이하'],
      ['보유 · 관망','위 어디에도 해당하지 않음']],['판단','조건'])+
    `<p><b>손절 참고가</b> = 상승 추세면 max(20일선 − 1×ATR, 20일 저점), 아니면 아래 매물대(또는 20일 저점) − 0.5×ATR. 현재가보다 높게 나오면 현재가 − 1.5×ATR.</p>
     <p><b>1차 목표</b> = 볼린저 상단·20일 고점·위쪽 매물대 중 현재가보다 1% 이상 높은 가장 가까운 값 (없으면 현재가 + 1.5×ATR) · <b>2차 목표</b> = max(60일 고점, 현재가 + 2.5×ATR)</p>
     <p>ATR은 하루 평균 변동폭(14일)입니다.</p>`],
  signals:['시장 방향 신호등 계산법',()=>{const R=D.risk;if(!R)return '';
    const row=x=>[x.kind,esc(x.label),x.on?'<b>켜짐</b>':'꺼짐',x.val?esc(x.val):''];
    return `<p><b>선행</b>은 방향이 바뀌기 전에 먼저 켜지는 신호, <b>진행·확인</b>은 이미 움직이고 있을 때 켜지는 신호입니다. 켜진 신호의 점수를 더해 0~100점으로 만듭니다.</p>
      <p>하락 위험: 0~24 안정 · 25~49 주의 · 50~74 경계 · 75+ 위험 &nbsp;|&nbsp; 상승 신호: 0~14 없음 · 15~34 약함 · 35~59 개선 · 60+ 강함</p>
      <h5>하락 쪽</h5>${T(R.dn.map(row),['구분','신호','오늘','값'])}<h5>상승 쪽</h5>${T(R.upl.map(row),['구분','신호','오늘','값'])}
      <p><b>분산일</b>: 지수가 0.2% 이상 하락하면서 거래량이 전날보다 많은 날. 25일 안에 4~5일 이상 쌓이면 큰손이 팔고 있다는 신호로 봅니다.<br><b>바닥 확인일</b>: 최근 저점 이후 4일째부터 지수가 거래량 증가와 함께 +1.5% 이상 오른 날.</p>
      <p>지수 데이터: ${esc(R.src)}. 폭락이나 상승을 정확히 맞히는 지표는 없고, 여러 신호가 동시에 켜질수록 가능성이 높아진다는 뜻입니다.</p>`;}],
  macro:['글로벌 지표 판정 기준',()=>`<p>최근 20일 변화가 아래 기준을 넘으면 방향에 따라 주식에 유리·불리로 봅니다. 기준 안이면 중립입니다.</p>`+T([
      ['원·달러 환율, 달러인덱스','±2%','오르면 불리 (외국인 매도 압력)'],['미국 10년 국채금리','±7%','오르면 불리 (성장주 할인율 상승)'],
      ['국제유가(WTI)','±12%','오르면 불리 (물가·비용 부담)'],['금','±5%','오르면 불리 (안전자산 선호)'],
      ['구리','±5%','오르면 유리 (경기 회복 기대)'],['하이일드 채권','±2%','오르면 유리 (신용위험 축소)'],
      ['S&P500·닛케이·상해·대만','±4~5%','오르면 유리'],['나스닥 ±5%, 반도체 ±7%','','오르면 유리'],['비트코인','±15%','오르면 유리 (위험 선호)'],
      ['VIX','수준·5일 변화','25 이상 또는 5일 +20% → 불리, 16 미만 또는 5일 −15% → 유리']],['지표','기준','해석'])+
    `<p>해외 시장은 시차 때문에 전날 밤 마감값입니다. 데이터는 야후 파이낸스입니다.</p>`],
  outlook:['5일 전망은 이렇게 만듭니다',()=>{const O=D.outlook;if(!O)return '<p>오늘은 계산하지 못했습니다.</p>';
    return `<p>코스피·코스닥과 섹터 대표 ETF 20개의 과거 데이터(${O.since}부터 ${f0(O.n)}개 표본)로 <b>"5일 뒤 올랐는가"</b>를 학습합니다. 로지스틱 회귀와 그래디언트 부스팅 두 모델을 만들어 중간 기간 성적이 더 좋은 쪽을 고르며, 오늘은 <b>${esc(O.model||'로지스틱 회귀')}</b>를 썼습니다. 입력 지표는 ${O.feats}개입니다 (최근 수익률, 20·60일선과의 거리, 20일선 기울기, RSI, 변동성, 고점 대비 하락폭, 코스피 대비 강도, 글로벌 지표 16개의 5·20일 변화, 미국 장단기 금리차, 월말·월초, 일정 4가지).</p>
      <p><b>정확도 검증</b>: 앞 60% 기간으로 학습 → 다음 20%로 모델 선택 → 학습·선택에 전혀 쓰지 않은 마지막 20% 기간(${esc(O.test_from||'')}~)에서 맞힌 비율이 <b>${O.hit}%</b>입니다. 많이 나온 쪽만 찍었을 때는 ${O.base}%입니다.${O.hit_conf!=null?` 확률이 60% 이상이거나 40% 이하로 뚜렷할 때만 보면 ${O.hit_conf}%입니다.`:''}</p>
      ${O.top?`<p><b>모델이 가장 무겁게 본 지표</b>: ${O.top.map(t=>`${esc(t.name)}(${t.dir})`).join(', ')}</p>`:''}
      <p><b>과거 비슷한 때 평균</b> = 같은 확률 구간(10% 단위)에 있었던 과거 사례들의 5일 뒤 평균 등락률<br>
      <b>예상 흔들림</b> = 최근 20일 일간 등락률의 표준편차 × √5 (약 68%가 이 범위 안에 들어옴)<br>
      <b>현재 추세</b> = 상승: 주가 &gt; 20일선 &gt; 60일선 + 20일선 상승 / 하락: 반대 / 나머지 횡보</p>
      <p>5일 뒤 시장은 전문 기관도 크게 맞히기 어렵습니다. 적중률이 찍기보다 크게 높지 않으면 방향보다 흔들림 폭을 참고하세요.</p>`;}],
  picks:['추천 종목 순위 매기는 법',()=>`<p><b>1단계 · 기본 필터</b>: 코스피·코스닥, 거래대금 ${f0(D.cfg.min_amount/1e8)}억 이상, 시가총액 ${f0(D.cfg.min_mcap/1e8)}억 이상, 주가 ${f0(D.cfg.min_price)}원 이상, 상장 6개월 이상. 우선주·ETF·ETN·스팩·적자 기업 제외.</p>
      <p><b>2단계 · 기본 점수 (100점)</b>: 전략마다 조건 5개를 얼마나 채웠는지 (각 카드의 '기본 조건' ? 참고).</p>
      <p><b>3단계 · 보조 지표 가감점</b>: 손익비, 수급, MACD, OBV, 상대강도, 피보나치, 세력 패턴.</p>
      <p><b>최종 점수</b> = 기본 점수 + 가감점 (0~100). 1~3위 매수 후보, 4~6위 지켜볼 종목, 7~16위 목록.</p>
      <p>📊 차트 결론은 공식으로 계산한 결과이고, 🤖 AI 결론은 Claude가 기사를 읽고 낸 의견입니다.</p>`],
  rules1:['추세 눌림목 · 기본 조건',()=>RULE(1)],rules2:['거래량 돌파 · 기본 조건',()=>RULE(2)],rules3:['과매도 반등 · 기본 조건',()=>RULE(3)],
  aux:['보조 지표 가감점',()=>T([
      ['손익비','(1차 목표 − 현재가) ÷ (현재가 − 손절가)','2 이상 +8 · 1.5 이상 +4 · 1 미만 −12'],
      ['수급 (5일)','외국인·기관 순매수 합계','둘 다 매수 +6 · 합계 매수 +3 · 둘 다 매도 −6 · 3일 이상 연속 동반 매수 +3'],
      ['MACD 히스토그램','(12일 EMA − 26일 EMA) − 그 9일 EMA','반등: 0선 돌파·하락 둔화 +5 / 그 외: 상승 확대 +3, 꺾이면 감점'],
      ['OBV','오른 날 거래량 더하고 내린 날 빼서 누적','가격 저점은 낮아지는데 OBV는 높아짐(강세 다이버전스) +3~6'],
      ['상대강도','종목 20일 수익률 − 시장 중간값','+5%p 넘으면 +4 · 0 넘으면 +2 · −10%p 밑이면 −4 (반등 전략은 기준 완화)'],
      ['피보나치','120일 고점·저점 사이 되돌림 비율','눌림목 50~61.8% +5 · 38.2~50% +3 / 반등 61.8~78.6% +5'],
      ['세력 패턴','최근 20일 캔들·거래량 모양','장대양봉 후 눌림 +5 · 조용한 매집 +4 · 저점 매집봉 +4 · 이탈 의심 −6'],
      ['52주 고가 근접','현재가 ÷ 최근 250일 최고가','90% 이상이면 +3 (눌림목·돌파) — 신고가 근처 종목이 더 오르는 경향(모멘텀)'],
      ['변동성','하루 평균 변동폭(ATR) ÷ 현재가','7% 넘으면 −3 — 너무 출렁이는 종목은 손절에 쉽게 걸림'],
      ['실적','네이버 EPS·PER','적자면 추천에서 제외 · PER 80 초과면 −2']],['지표','계산','점수'])],
  target:['목표가 · 손절가 · 손익비',()=>T([[STRAT[3].n,D.targets[3][0],D.targets[3][1],D.targets[3][2]],[STRAT[1].n,D.targets[1][0],D.targets[1][1],D.targets[1][2]],[STRAT[2].n,D.targets[2][0],D.targets[2][1],D.targets[2][2]]],['전략','1차 목표','2차 목표','손절'])+
    `<p>1차 목표와 현재가 사이에 거래가 많이 쌓인 가격대(매물대)가 있으면 1차 목표를 그 바로 아래로 낮춥니다. 손절가가 현재가보다 높게 나오면 현재가 − 1.5×ATR로 바꿉니다.</p>
     <p><b>손익비</b> = (1차 목표 − 현재가) ÷ (현재가 − 손절가). 1 : 2라면 잃을 수 있는 금액의 2배를 노린다는 뜻입니다.</p>`],
  news:['최근 한 달 뉴스',()=>`<p>매일 아침 구글 뉴스에서 종목명이 제목에 들어간 최근 30일 국내 기사를, 구글 뉴스(영문명)·야후 파이낸스에서 해외 기사를 모아 제목만 보여줍니다. AI가 판단한 내용이 아니며, 제목을 누르면 원문이 열립니다.</p>`],
  brief:['오늘의 브리핑',()=>`<p>평일 아침 <b>6시 20분, 7시 5분, 7시 30분</b> 세 번 자동으로 다시 만들어서, 넥스트레이드 프리마켓(8시)과 정규장(9시) 전에 최신 자료가 올라가 있도록 했습니다. GitHub 사정으로 한 번이 늦어도 다음 회차가 갱신합니다. 페이지를 열어 두는 동안에도 10분마다 새 버전을 확인해 자동으로 바꿔 보여줍니다.</p>
      <ul><li><b>밤사이 미국·환율·원자재</b>: 야후 파이낸스 전날 종가 대비</li><li><b>시장 신호등·5일 전망·글로벌 지표</b>: 아래 각 섹션 요약</li><li><b>이번 주 일정</b>: 다가오는 일정 중 7일 이내</li>
      <li><b>실적 발표</b>: 구글 뉴스의 최근 하루 실적 기사${D.brief&&D.brief.dart?' + DART의 잠정실적 공시(최근 1일)와 기업설명회(IR) 개최 공시(최근 10일)':'. DART API 키(무료)를 등록하면 잠정실적·IR 공시 목록도 함께 나옵니다'}</li>
      <li><b>국내·해외 증시 뉴스</b>: 구글 뉴스 최근 하루 기사 제목</li><li><b>섹터</b>: 최근 5일 ±3% 넘게 움직였거나 3일 안에 일정이 있는 섹터의 기사</li></ul>
      <p>뉴스는 제목만 모은 것이며 AI 판단이 아닙니다.</p>`],
  yt:['유튜브 언급 종목 분석',()=>`<p>${esc((D.yt||{channel:{}}).channel.name||'')} 채널의 최근 영상 15개를 매일 아침 유튜브 공개 RSS로 받아, <b>영상 제목과 설명란</b>에 나온 종목명을 찾아 우리 방식으로 분석합니다.</p>
      <ul><li><b>우리 판정</b>: 차트 추세(20·60일선), 과열(RSI 70 이상·볼린저 상단 돌파), 세력 이탈 패턴으로 '우리 방식도 긍정적 / 관망 / 추격 주의 / 주의'를 붙입니다. 종목을 누르면 세 전략 중 가장 잘 맞는 전략의 점수·목표가·손절가·차트가 열립니다.</li>
      <li><b>영상 이후 수익</b>: 영상 공개 후 첫 장 시가에 샀다고 가정하고 지금까지(비용 ${D.cfg&&D.bt?D.bt.cost:0.25}% 차감) 계산합니다. 시장 대비는 같은 시장 종목 중간값과 비교합니다.</li>
      <li><b>한계</b>: 영상 안에서 말로만 언급한 종목은 찾지 못합니다(자막을 읽지 않음). 그런 종목은 '내 종목' 탭에서 직접 점검하세요. 짧은 이름(예: 두 글자)은 다른 단어와 헷갈릴 수 있어 일부는 제외합니다.</li></ul>
      <p>이 기능은 채널의 의견을 평가하려는 것이 아니라, 같은 종목을 우리 기준으로 한 번 더 확인하는 참고용입니다.</p>`],
  gainers:['어제의 급등주 해부',()=>`<p>전 거래일 <b>+15% 이상</b> 오르고 거래대금 10억 원 이상인 종목을 최대 5개 보여줍니다.</p>
      <ul><li><b>확인된 재료</b>: 최근 이틀 구글 뉴스 중 종목명이 제목에 들어간 기사. 없으면 테마성 상승일 수 있다고 표시합니다.</li>
      <li><b>차트 판정</b>: 추세·과열·패턴으로 붙인 우리 의견.</li>
      <li><b>과거 2년 통계</b>: 같은 크기로 급등한 종목들이 <b>다음 날</b> 시가·종가 기준으로 평균 얼마였고 몇 %가 올랐는지, 5일 뒤는 어땠는지(거래대금 10억 이상만). 급등 다음 날 추격 매수가 평균적으로 유리한지 판단하는 근거입니다.</li></ul>`],
  events:['다가오는 일정',()=>`<p>앞으로 45일 안의 일정을 보여줍니다. D-5 이내는 회색 배경으로 강조합니다.</p>
      <h5>일정 출처</h5><ul>
      <li><b>검색해서 넣은 일정</b>: 바이오 학회(JP모건 헬스케어·AACR·ASCO·ESMO·ASH), IT·반도체 행사(CES·엔비디아 GTC·컴퓨텍스·삼성 언팩·애플 발표), 미국 FOMC, 한국은행 금통위. '예상 날짜'는 주최 측 공식 발표 전 관례 날짜입니다.</li>
      <li><b>규칙으로 자동 계산</b>: 선물옵션 동시만기(3·6·9·12월 둘째 목요일, 6·12월은 코스피200 정기변경), 실적 시즌 시작(1·4·7·10월 7일 무렵), 월간 수출입 동향(매월 1일), 연말 배당락(12월 마지막 거래일 하루 전).</li></ul>
      <h5>과거 패턴 계산</h5><p><b>일정 전 5일</b> = 일정 전날 종가 ÷ 6거래일 전 종가 − 1 · <b>일정 후 5일</b> = 일정 후 4거래일 종가 ÷ 일정 전날 종가 − 1. 관련 섹터 대표 ETF(시장 전체 일정은 코스피)로 과거 약 5년간 같은 일정을 모아 평균과 상승 비율을 냅니다. 표본이 3번 미만이면 표시하지 않습니다.</p>
      <h5>5일 전망 모델에 들어가는 값</h5><p>섹터 일정이 5거래일 안에 있는지, 시장 전체 일정이 5거래일 안에 있는지, 섹터 일정 직후인지, 실적 시즌인지 4가지를 입력으로 넣습니다.</p>
      <h5>일정 추가하기</h5><p>GitHub 저장소에 <b>events.json</b> 파일을 만들고 <code>[{"date":"2027-05-20","name":"행사명","sectors":["반도체"]}]</code> 형식으로 적으면 다음 실행부터 반영됩니다. 섹터 이름은 표의 섹터 이름과 같게 쓰고, 시장 전체 일정은 <code>"*"</code>로 씁니다. 몇 달에 한 번 Claude에게 "일정 업데이트해줘"라고 하시면 다시 검색해서 채워 드립니다.</p>`],
  ai:['🤖 AI 분석 버튼',()=>`<ol><li>버튼을 누르면 Claude가 이 종목의 차트 결과와 오늘 모은 뉴스 제목이 <b>입력된 상태로</b> 열립니다.</li><li><b>Enter</b>만 누르면 Claude가 최근 한 달 국내·해외(영문) 기사와 공시를 검색해 매수/관망/주의 의견을 냅니다.</li><li>입력칸이 비어 있으면 '요청 문장 복사'를 눌러 붙여넣으세요.</li></ol><p>사장님의 Claude 구독 안에서 처리되어 추가 비용이 없습니다.</p>`],
};
const helpDlg=document.getElementById('help');
document.addEventListener('click',e=>{const q=e.target.closest('[data-help]');if(!q)return;e.preventDefault();e.stopPropagation();
  const h=HELP[q.dataset.help];if(!h)return;document.getElementById('helpTitle').textContent=h[0];document.getElementById('helpBody').innerHTML=h[1]();helpDlg.showModal();},true);
document.getElementById('helpClose').onclick=()=>helpDlg.close();
document.getElementById('helpMore').onclick=()=>helpDlg.close();
helpDlg.addEventListener('click',e=>{if(e.target===helpDlg)helpDlg.close();});


/* ───── 읽기 쉬운 결론 (확인된 근거 · 해석 · 확인할 것) ───── */
function storyHTML(x){
  const ok=x.checks.filter(k=>k.s>=.99),miss=x.checks.filter(k=>k.s<.99),plus=x.aux.filter(a=>a.d>0),minus=x.aux.filter(a=>a.d<0);
  const rr=x.tg.rr,R=D.risk;
  const head=ok.length>=4&&rr!=null&&rr>=1.5?`${STRAT[x.strat].n} 조건이 잘 맞고, 목표까지 여지가 손실 폭보다 큽니다`:
    ok.length>=4?`${STRAT[x.strat].n} 조건은 맞지만, 손익비는 ${rr==null?'-':'1:'+rr.toFixed(1)}로 크지 않습니다`:
    `${STRAT[x.strat].n} 조건 ${ok.length}/${x.checks.length}개만 충족 · 나머지 조건 확인이 필요합니다`;
  const facts=[...ok.map(k=>`${k.label} <span class="vl">${esc(k.v)}</span>`),...plus.map(a=>`${a.label} <span class="vl">${esc(a.v)}</span>`)].slice(0,6);
  const reading=[`${STRAT[x.strat].d}.`,`1차 목표 ${f0(x.tg.t1)}원(${pctS(x.tg.t1p)})까지 오를 여지, 손절 ${f0(x.tg.stop)}원(${pctS(x.tg.sp)})이면 ${rr==null?'':'손익비 1:'+rr.toFixed(1)}.`];
  if(x.read)reading.push(`차트 판정: ${x.read.trend}${x.read.hot?' · 단기 과열':''}.`);
  const check=[...miss.map(k=>`${k.label} <span class="vl">${esc(k.v)}</span>`),...minus.map(a=>`${a.label} <span class="vl">${esc(a.v)}</span>`)];
  if(x.tg.note)check.push(esc(x.tg.note));
  if(R&&R.score>=50)check.push(`시장 위험 '${R.level}' 단계 · 매수 금액 줄이기`);
  check.push('관리종목·투자경고 여부와 최근 공시');
  return `<div class="story"><p class="shead">${head}</p>
    <div class="sblk"><b>확인된 근거</b><ul>${facts.length?facts.map(f=>`<li>${f}</li>`).join(''):'<li class="vl">뚜렷한 근거 없음</li>'}</ul></div>
    <div class="sblk"><b>해석</b><p>${reading.join(' ')}</p></div>
    <div class="sblk warn"><b>확인할 것</b><ul>${check.slice(0,6).map(f=>`<li>${f}</li>`).join('')}</ul></div>
    <p class="src tn">기준: ${D.asof} 정규장 종가 · 데이터 네이버 금융 · ${esc(D.generated)} 계산</p></div>`;
}

/* ───── 유튜브 ───── */
const YT=D.yt||null;
const ytStock=code=>YT&&YT.stocks[code];
function mentChips(v){return v.ments.length?v.ments.map(mn=>{const s=ytStock(mn.code);if(!s)return '';const pf=mn.perf;
  return `<button class="ych ${s.read.tone}" data-yts="${mn.code}"><b>${esc(s.name)}</b>${pf?`<span class="tn ${pf.now>=0?'upc':'downc'}">${pctS(pf.now,1)}</span>`:''}<small>${esc(s.read.op)}</small></button>`;}).join(''):'<span class="vl">제목·설명에서 찾은 종목 없음</span>';}
function openVideo(i){
  const v=YT.videos[i];
  const prompt=`한균수의 주식사용설명서 유튜브 영상 "${v.title}" (${v.date}, https://youtu.be/${v.vid})에서 언급된 종목: ${v.ments.map(m=>(ytStock(m.code)||{}).name).filter(Boolean).join(', ')||'없음'}.\n`+
    v.ments.map(m=>{const s=ytStock(m.code);return s?`- ${s.name}: 우리 차트 판정 ${s.read.op}(${s.read.trend}, RSI ${s.read.rsi}), ${STRAT[s.strat].n} ${s.score}점, 목표 ${f0(s.tg.t1)}원·손절 ${f0(s.tg.stop)}원`+(m.perf?`, 영상 이후 ${pctS(m.perf.now,1)}`:''):'';}).join('\n')+
    `\n\n이 종목들의 최근 한 달 국내·해외 뉴스와 공시를 검색해서, 영상이 다룬 투자 포인트가 여전히 유효한지 우리 차트 판정과 비교해줘. 종목별로 매수/관망/주의 의견, 근거 3가지, 리스크, 기사 링크를 알려줘.`;
  openSheet(`<div class="ytv"><div class="yframe"><iframe src="https://www.youtube-nocookie.com/embed/${esc(v.vid)}" title="${esc(v.title)}" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen loading="lazy"></iframe></div>
    <h3>${esc(v.title)}</h3><p class="vl tn" style="text-align:left">${v.date} · 조회 ${f0(v.views)}회 · <a href="https://youtu.be/${esc(v.vid)}" target="_blank" rel="noopener">유튜브에서 보기</a></p>
    <div class="lbl">언급 종목 · 우리 방식 판정</div><div class="ychips">${mentChips(v)}</div>
    <div class="aibox aibtn" style="margin-top:12px"><p><b style="color:var(--ink)">🤖 영상 종목 AI 점검</b><br>영상 제목과 우리 분석이 입력된 채로 Claude가 열립니다. Enter만 누르세요.</p><a class="btn ai" href="${claudeURL(prompt,null)}" target="_blank" rel="noopener">🤖 Claude에서 점검</a></div></div>`);
}
function openYtStock(code){const x=ytStock(code)||(D.gainers&&D.gainers.items.find(g=>g.code===code));if(!x)return;
  openSheet(pickCard(x),()=>{const b=sheet.querySelector('.chartbox');if(b&&x.series)drawChart(b,x.series);});}
function ytCardHTML(){
  if(!YT||!YT.videos.length)return '';
  const v=YT.videos[0];
  return `<div class="card yt"><div class="ch"><h3>📺 ${esc(YT.channel.name)}</h3><a class="more" href="#yt">전체 →</a></div>
    <button class="ytrow" data-vid="0"><img src="https://i.ytimg.com/vi/${esc(v.vid)}/mqdefault.jpg" alt="" loading="lazy"><span><b>${esc(v.title)}</b><small class="vl">${v.date}</small></span></button>
    <div class="ychips">${mentChips(v)}</div></div>`;
}
function renderYt(){
  const box=document.getElementById('v-yt');
  if(!YT){box.innerHTML='<div class="empty">오늘은 유튜브 영상을 받지 못했습니다.</div>';return;}
  const S=YT.summ;
  box.innerHTML=`<div class="vh"><h2>📺 ${esc(YT.channel.name)} <button class="q" data-help="yt" aria-label="설명 보기">?</button></h2><a class="more" href="${esc(YT.channel.url)}" target="_blank" rel="noopener">채널 열기 ↗</a></div>
    ${S?`<div class="card ysum tn"><div><small>언급 종목 성적 (영상 다음 장 시가 매수 가정)</small><b class="${S.now_ex>=0?'pos':'neg'}">${pctS(S.now_ex,2)}</b><span class="vl">시장 대비 · ${S.n}건 · 오른 비율 ${S.win}%</span></div>
      ${S.ex5!=null?`<div><small>5거래일 뒤</small><b class="${S.ex5>=0?'pos':'neg'}">${pctS(S.ex5,2)}</b><span class="vl">시장 대비 · ${S.n5}건</span></div>`:''}</div>`:''}
    <div class="ylist">${YT.videos.map((v,i)=>`<div class="card yv"><button class="ytrow" data-vid="${i}"><img src="https://i.ytimg.com/vi/${esc(v.vid)}/mqdefault.jpg" alt="" loading="lazy"><span><b>${esc(v.title)}</b><small class="vl">${v.date} · 조회 ${f0(v.views)}</small></span></button><div class="ychips">${mentChips(v)}</div></div>`).join('')}</div>`;
}

/* ───── 어제의 급등주 해부 ───── */
function renderGainers(){
  const G=D.gainers,box=document.getElementById('gainers');
  if(!G||!G.items.length){box.innerHTML='<div class="empty">어제는 +15% 넘게 오른 종목이 없습니다.</div>';return;}
  const st=Object.entries(G.stats||{});
  box.innerHTML=(st.length?`<div class="gstat">${st.map(([k,v])=>`<div><small>과거 2년 ${esc(k)} 다음 날</small><b class="tn ${v.d1>=0?'upc':'downc'}">${pctS(v.d1,2)}</b><span class="vl tn">시가 ${pctS(v.gap,2)} · 오른 비율 ${v.up1}% · 5일 뒤 ${pctS(v.d5,1)} · ${f0(v.n)}건</span></div>`).join('')}</div>`:'')+
  G.items.map((x,i)=>`<article class="gcard">
      <div class="gno tn">${String(i+1).padStart(2,'0')}</div>
      <div class="gbody"><h4>${esc(x.name)} <span class="code tn">${x.code}</span></h4>
        <div class="price tn">${f0(x.close)}<span>${pct(x.chg)}</span><small class="vl"> 거래대금 ${won(x.amount)}</small></div>
        <p class="shead">${x.read.trend}${x.read.hot?' · 단기 과열':''} → <span class="chip ${x.read.tone}">${esc(x.read.op)}</span></p>
        <div class="sblk"><b>확인된 재료 (뉴스 제목)</b>${x.news&&x.news.length?`<ul>${x.news.map(n=>`<li><a href="${esc(n.u)}" target="_blank" rel="noopener">${esc(n.t)}</a> <span class="vl">${esc(n.s||'')} ${esc(n.d||'')}</span></li>`).join('')}</ul>`:'<p class="vl" style="text-align:left">최근 이틀 뉴스에서 재료를 찾지 못했습니다 → 테마성 상승일 수 있음</p>'}</div>
        <div class="sblk warn"><b>확인할 것</b><ul><li>공시로 확인된 계약·실적인지, 기대감인지</li><li>관리종목·투자경고·거래정지 여부</li><li>다음 날 시가 갭과 거래대금 유지 여부</li></ul></div>
        <button class="btn ghost sm" data-yts="${x.code}">차트·조건 자세히</button></div></article>`).join('');
}
/* ───── 시장 · 탭 ───── */
const RLV={'안정':['ok','신규 매수 가능. 평소대로 전략을 쓰세요.'],
  '주의':['mid','경고 신호가 하나둘 켜졌습니다. 신규 매수 금액을 평소보다 줄이고, 손절가를 꼭 지키세요.'],
  '경계':['bad','하락 위험이 커졌습니다. 신규 매수는 평소의 절반 이하로 줄이고, 수익 난 종목은 일부 이익을 확정하는 것을 검토하세요.'],
  '위험':['bad','하락장이 진행 중일 가능성이 큽니다. 신규 매수는 쉬고 현금 비중을 늘리는 것을 검토하세요. 손절가를 깬 종목은 미루지 말고 정리하세요.']};
function renderRisk(){
  const R=D.risk;if(!R){document.getElementById('risk').hidden=true;return;}
  const [tone,act]=RLV[R.level];
  const utone=R.up>=60?'ok':R.up>=35?'okl':'mute';
  let msg=act;
  if(R.score>=50&&R.up>=35)msg='하락 신호와 바닥 신호가 함께 켜졌습니다. 하락이 마무리되는 구간일 수 있으니, 바로 크게 사지 말고 지수가 20일선을 회복하는지 확인하며 조금씩 나눠 사는 것을 검토하세요.';
  else if(R.score<25&&R.up>=60)msg='상승 신호가 강합니다. 추세 눌림목·거래량 돌파 전략이 잘 맞는 구간입니다. 손절가만 지키면서 평소 비중으로 접근해 보세요.';
  else if(R.score<50&&R.up>=35)msg='상승 쪽 신호가 늘고 있습니다. '+act;
  const h=R.hist,W=520,H=78,X=i=>8+i*(W-16)/Math.max(1,h.length-1),Y=v=>H-14-v/100*(H-20);
  const spark=`<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="최근 30거래일 하락 위험과 상승 신호 점수" class="rspark">
    ${[25,50,75].map(v=>`<line x1="8" x2="${W-8}" y1="${Y(v)}" y2="${Y(v)}" class="rg"/>`).join('')}
    <polyline points="${h.map((p,i)=>X(i)+','+Y(p.s)).join(' ')}" class="rl dn"/>
    <polyline points="${h.map((p,i)=>X(i)+','+Y(p.u)).join(' ')}" class="rl up"/>
    <text x="8" y="${H-1}" class="rx">${h[0].d}</text><text x="${W-8}" y="${H-1}" class="rx" text-anchor="end">${h[h.length-1].d}</text></svg>`;
  const tr=(now,prev,word)=>prev==null?'':now>prev?`${word} 커지는 중 (5일 전 ${prev})`:now<prev?`${word} 줄어드는 중 (5일 전 ${prev})`:`5일 전과 같음`;
  const li=x=>`<li class="${x.on?'on':''}"><span class="ic">${x.on?'●':'○'}</span><span><em class="kd">${x.kind}</em>${esc(x.label)}${x.val?` <span class="vl">${esc(x.val)}</span>`:''}</span><b class="tn">${x.on?'+'+x.pts:''}</b></li>`;
  const idx=Object.entries(R.idx).map(([k,v])=>`${k} ${f0(v.c)} ${pct(v.chg)} · 60일 고점 대비 ${v.dd}%`).join(' &nbsp;|&nbsp; ');
  document.getElementById('risk').innerHTML=`
    <div class="rtop">
      <div class="rbadges">
        <div class="rbadge ${tone}"><span>하락 위험 · ${R.level}</span><b class="tn">${R.score}</b><small class="tn">${tr(R.score,R.prev5,'위험')}</small></div>
        <div class="rbadge ${utone}"><span>상승 신호 · ${R.ulevel}</span><b class="tn">${R.up}</b><small class="tn">${tr(R.up,R.uprev5,'신호')}</small></div>
      </div>
      <div class="rmain"><p class="ract">${msg}</p><p class="vl tn" style="text-align:left">${idx}</p></div>
      <div class="rchart">${spark}<p class="vl" style="text-align:center;margin:0"><span class="lgd dn"></span>하락 위험 &nbsp; <span class="lgd up"></span>상승 신호 · 최근 30거래일</p></div>
    </div>
    <div class="rcols">
      <div><h4>📉 하락 쪽 신호</h4><ul>${R.dn.map(li).join('')}</ul></div>
      <div><h4>📈 상승 쪽 신호</h4><ul class="upl">${R.upl.map(li).join('')}</ul></div>
    </div>
    ${R.ext.length?'':'<p class="vl" style="text-align:left;margin:10px 0 0">오늘은 해외 지표를 받지 못했습니다.</p>'}`;
}
function cone(o){
  const c=o.spark.filter(x=>x!=null),n=c.length,W=360,H=120,pad=6,F=5;
  const last=c[n-1],mid=last*(1+o.er/100),hiP=mid*(1+o.band/100),loP=mid*(1-o.band/100);
  const lo=Math.min(...c,loP),hi=Math.max(...c,hiP),X=i=>pad+i*(W-2*pad)/(n-1+F),Y=v=>H-pad-(v-lo)/(hi-lo)*(H-2*pad);
  const up=o.p>=50;
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="${o.name} 최근 60일과 5일 예상 범위" class="cone">
    <polygon points="${X(n-1)},${Y(last)} ${X(n-1+F)},${Y(hiP)} ${X(n-1+F)},${Y(loP)}" class="cband"/>
    <polyline points="${c.map((v,i)=>X(i)+','+Y(v)).join(' ')}" class="cl2"/>
    <line x1="${X(n-1)}" y1="${Y(last)}" x2="${X(n-1+F)}" y2="${Y(mid)}" class="cmid ${up?'u':'d'}"/>
    <line x1="${X(n-1)}" x2="${X(n-1)}" y1="${pad}" y2="${H-pad}" class="cnow"/></svg>`;
}
function renderMacro(){
  const M=D.macro||[],box=document.getElementById('macro');
  if(!M.length){box.innerHTML='<div class="empty">오늘은 해외 지표를 받지 못했습니다.</div>';return;}
  const g=M.filter(x=>x.eff>0).length,b=M.filter(x=>x.eff<0).length;
  const eff=e=>e>0?'<span class="tr tg">주식에 유리</span>':e<0?'<span class="tr tb">주식에 불리</span>':'<span class="tr tf">중립</span>';
  const fmtv=(k,v)=>v==null?'-':k==='us10y'?v.toFixed(2)+'%':v>=1000?f0(v):v.toFixed(2);
  box.innerHTML=`<p class="mx-sum">${M.length}개 지표 중 <b class="pos">유리 ${g}개</b> · <b class="neg">불리 ${b}개</b> · 중립 ${M.length-g-b}개 </p>
    <div class="tablewrap"><table class="mxtbl"><thead><tr><th>지표</th><th>최근 60일</th><th>현재</th><th>5일</th><th>20일</th><th>주식 영향</th></tr></thead><tbody>
    ${M.map(x=>`<tr class="tn"><td class="nm"><b>${esc(x.name)}</b><span>${x.d} 기준</span></td><td>${miniSpark(x.spark)}</td><td>${fmtv(x.key,x.v)}</td><td>${pct(x.c5,1)}</td><td>${pct(x.c20,1)}</td><td>${eff(x.eff)}</td></tr>`).join('')}
    </tbody></table></div>
`;
}
function renderBrief(){
  const B=D.brief,box=document.getElementById('brief');
  if(!B){box.innerHTML='<div class="empty">오늘은 뉴스를 받지 못했습니다.</div>';return;}
  const item=x=>`<li><span class="dt tn">${esc(x.d||'')}</span>${x.en?'<span class="en">해외</span>':''}<a href="${esc(x.u)}" target="_blank" rel="noopener">${esc(x.t)}</a></li>`;
  const list=(arr,empty)=>arr&&arr.length?`<ul class="bl">${arr.map(item).join('')}</ul>`:`<p class="vl" style="text-align:left;margin:0">${empty}</p>`;
  box.innerHTML=`<div class="bgrid">
      <div class="bbox"><h4>📊 실적 발표</h4>${list(B.earnings,'최근 하루 실적 소식이 없습니다.')}</div>
      <div class="bbox"><h4>🇰🇷 국내 증시</h4>${list(B.kr,'뉴스를 받지 못했습니다.')}</div>
      <div class="bbox"><h4>🌎 해외 증시</h4>${list(B.gl,'뉴스를 받지 못했습니다.')}</div>
      ${(B.sectors||[]).length?`<div class="bbox"><h4>🔥 움직임 큰 섹터</h4>${B.sectors.map(x=>`<div class="bsec"><span class="sb">${esc(x.sector)}</span>${x.r5!=null?` <span class="vl">5일 ${pct(x.r5,1)}</span>`:''}${list(x.items,'')}</div>`).join('')}</div>`:''}
    </div>`;
}
function checkFresh(){
  const now=new Date(),gen=new Date((D.gen_iso||'').replace(' ','T'));
  const wk=now.getDay()>=1&&now.getDay()<=5,today6=new Date(now.getFullYear(),now.getMonth(),now.getDate(),6,0);
  const stale=wk&&now>=today6&&(isNaN(gen)||gen<today6);
  const el=document.getElementById('stale');
  if(stale){el.hidden=false;el.innerHTML=`⚠️ 아직 오늘 아침 자료로 갱신되지 않았습니다 (마지막 갱신 ${esc(D.generated)}). 평일 오전 6시 20분·7시 5분·7시 30분에 자동 갱신되며, 잠시 후 새로고침해 보세요.`;}
  // 열려 있는 동안에도 새 버전이 올라오면 자동으로 바꿔 보여줌
  fetch('version.json?t='+Date.now(),{cache:'no-store'}).then(r=>r.ok?r.json():null).then(v=>{
    if(v&&v.generated&&D.gen_iso&&v.generated>D.gen_iso){const u=new URL(location.href);u.searchParams.set('v',v.generated);location.replace(u.toString());}
  }).catch(()=>{});
}
function renderScore(){
  const box=document.getElementById('score'),bt=D.bt,O=D.outlook,R=D.risk;
  const cards=[];
  if(bt){const vc=v=>v==='효과 있음'?'tg':v==='역효과'?'tb':'tf';
    const rows=[3,1,2].filter(k=>bt.strat[k]&&bt.strat[k].h3).map(k=>{const v=bt.strat[k],x=v.h3;
      return `<tr class="tn"><td><span class="tag ${STRAT[k].c}">${STRAT[k].n}</span></td><td class="${x.ex>=0?'pos':'neg'}"><b>${pctS(x.ex,2)}</b></td><td>${x.win}%</td><td>${x.t==null?'-':x.t}</td><td><span class="tr ${vc(v.verdict)}">${v.verdict}</span></td></tr>`;}).join('');
    cards.push(`<div class="sc wide"><h4>추천 전략 과거 검증 <span class="vl">${bt.start} ~ ${bt.end} · ${bt.n_days}거래일 · 비용 ${bt.cost}% 차감</span></h4>
      <div class="tablewrap flat"><table class="sctbl"><thead><tr><th>전략</th><th>3일 시장 대비</th><th>승률</th><th>t값</th><th>판정</th></tr></thead><tbody>${rows}</tbody></table></div>
      <p class="vl" style="text-align:left;margin:4px 0 0">t값 2 이상이고 앞·뒤 기간 모두 좋아야 '효과 있음'입니다.</p></div>`);}
  if(O){const good=O.hit>O.base+1;
    cards.push(`<div class="sc"><h4>5일 전망 적중률</h4><div class="big2 tn ${good?'pos':'neg'}">${O.hit}%</div><p class="vl" style="text-align:left;margin:0">찍기 ${O.base}% · ${esc(O.model||'')} · ${O.test_from||''}~ 기간</p><p class="verdict2 ${good?'pos':'neg'}">${good?'찍기보다 조금 나음':'찍기와 비슷 · 흔들림 폭만 참고'}</p></div>`);}
  if(R&&R.valid&&R.valid.length){const g=k=>R.valid.find(v=>v.lab===k);const a=g('경계 이상'),b=g('안정');
    cards.push(`<div class="sc"><h4>신호등 과거 검증 <span class="vl">이후 5일 코스피</span></h4>
      <ul class="scl tn">${R.valid.filter(v=>v.kind==='하락 위험').map(v=>`<li><span>${v.lab}</span><b class="${v.f5>=0?'upc':'downc'}">${pctS(v.f5,2)}</b><span class="vl">${v.up5}% 상승 · ${v.n}일</span></li>`).join('')}</ul>
      ${a&&b?`<p class="verdict2 ${a.f5<b.f5?'pos':'neg'}">${a.f5<b.f5?'경계일 때 실제로 더 약했음':'경계 신호가 잘 맞지 않았음'}</p>`:''}</div>`);}
  if(!cards.length){box.hidden=true;return;}
  box.innerHTML=`<div class="bhead"><h2 style="margin:0">📏 이 도구의 성적표 <a class="q" href="#guide-verify" aria-label="자세히">?</a></h2><span class="vl">과거 실제 데이터로 다시 계산한 결과</span></div><div class="scgrid">${cards.join('')}</div>`;
}
function btGuide(bt){
  const hz=(s)=>[1,3,5].map(h=>{const x=s&&s['h'+h];return x?[`${h}일`,x.n+'건',pctS(x.avg,2),pctS(x.med,2),x.win+'%',`<b>${pctS(x.ex,2)}</b>`,pctS(x.worst,1),x.t==null?'-':x.t]:null;}).filter(Boolean);
  const head=['보유','건수','평균','중앙값','승률','시장 대비','최대 손실','t값'];
  let h=`<h5>추천 전략 과거 검증 (${bt.start} ~ ${bt.end}, ${bt.n_days}거래일)</h5>
    <p>그 기간 <b>매 거래일</b>마다 '그날 이 프로그램이 골랐을 전략별 1~3위'를 다시 계산했습니다. 진입은 <b>다음 날 시가</b>, 수익은 1·3·5거래일 뒤 종가 기준이며 <b>왕복 비용 ${bt.cost}%</b>를 뺐습니다. 시장 대비는 <b>같은 시장(코스피/코스닥)</b> 종목 중간값과 비교했습니다. t값은 날짜별 평균으로 계산해 같은 날 몰린 결과에 휘둘리지 않게 했고, <b>2 이상</b>이어야 우연이 아닐 가능성이 높습니다.</p>`;
  [3,1,2].forEach(k=>{const v=bt.strat[k];if(!v)return;const rg=bt.regime[k]||{},hf=bt.half[k]||{};
    h+=`<h5 class="${STRAT[k].c}">${STRAT[k].n} · ${v.verdict} <span class="vl">목표 도달 ${v.tgt??'-'}% · 손절 도달 ${v.stp??'-'}%</span></h5>${T(hz(v),head)}`;
    const r3=(o)=>o&&o.h3?`${pctS(o.h3.ex,2)} (${o.h3.n}건, t=${o.h3.t??'-'})`:'-';
    h+=T([['강세장 (20일선 위 종목 50% 이상)',r3(rg['강세'])],['약세장',r3(rg['약세'])],[`앞 기간 ${hf.first||''}`,r3(hf.a)],[`뒤 기간 ${hf.second||''}`,r3(hf.b)]],['구분','3일 시장 대비']);});
  const g=bt.vb_grid||[];
  if(g.length){h+=`<h5>거래량 돌파 · 기준값 바꿔보기 (3일 시장 대비)</h5><p>그날 후보 상위 30개 안에서 기준을 바꿔 다시 고른 결과입니다. 한 칸만 좋고 주변 칸이 나쁘면 우연일 가능성이 큽니다.</p>`+
    T(g.map(x=>[`${x.vmin}배 이상`,x.cmax>=30?'제한 없음':`+${x.cmax}% 이하`,x.n+'건',`<b>${pctS(x.ex3,2)}</b>`,x.win+'%',x.t??'-',`${x.ex3_a==null?'-':pctS(x.ex3_a,2)} / ${x.ex3_b==null?'-':pctS(x.ex3_b,2)}`]),['거래량','당일 상승','건수','시장 대비','승률','t값','앞 / 뒤 기간']);}
  const sh=bt.vb_shadow||{};
  if(sh.pass_||sh.fail){const f=o=>o&&o.h3?`${pctS(o.h3.ex,2)} (${o.h3.n}건)`:'-';
    h+=`<h5>거래량 돌파 · 그림자 필터</h5><p>당일 +${sh.rule.max_day_change}% 이하 · 종가가 고가의 ${Math.round(sh.rule.min_close_to_high*100)}% 이상 · 거래량 ${sh.rule.min_vol_ratio}배 이상인지 <b>기록만</b> 하고 실제 추천에는 아직 쓰지 않습니다.</p>`+T([['통과',f(sh.pass_)],['미통과',f(sh.fail)]],['','3일 시장 대비']);}
  h+=`<p class="vl" style="text-align:left">한계: 지금 상장된 종목으로만 계산해서 그 사이 상장폐지된 종목이 빠져 있습니다(결과가 실제보다 조금 좋게 나올 수 있음). 과거 수급·실적·매물대 자료가 없어 그 부분은 빼고 계산했습니다. 뉴스·AI 판단은 과거로 재현할 수 없어 실시간 기록으로만 확인합니다.</p>`;
  return h;
}
function renderGuide(){
  const g=document.getElementById('guideview'),bt=D.bt,O=D.outlook,R=D.risk,S=D.status||{};
  const sec=(id,title,body)=>`<section class="gsec" id="guide-${id}"><h3>${title}</h3>${body}</section>`;
  const st=Object.entries(S).map(([k,v])=>[`${v.ok===null?'➖':v.ok?'✅':'⚠️'} ${esc(k)}`,esc(v.detail)]);
  const btRows=[];
  const toc=[['verify','성적표 · 팩트 체크'],['data','데이터 수집 상태'],['how','전체 흐름'],['picks','추천 순위'],['rules','전략별 기본 조건'],['aux','보조 지표'],['target','목표가·손절가'],['hold','내 종목 점검'],['signals','시장 방향 신호등'],['macro','글로벌 지표'],['outlook','5일 전망 모델'],['events','다가오는 일정'],['brief','오늘의 브리핑'],['ai','AI 분석·뉴스']];
  g.innerHTML=`<a class="back" href="#">← 대시보드로 돌아가기</a><h2 style="margin-top:12px">📖 상세 설명 · 검증 결과</h2>
    <nav class="gtoc">${toc.map(([id,t])=>`<a href="#guide-${id}">${t}</a>`).join('')}</nav>
    ${sec('verify','📏 성적표 · 팩트 체크',
      (bt?btGuide(bt):'<p>과거 성적을 계산하지 못했습니다.</p>')+
      (O?`<h5>5일 전망 모델</h5><p>모델 후보 중간 기간 성적: ${Object.entries(O.va||{}).map(([k,v])=>`${esc(k)} ${v}%`).join(' · ')} → <b>${esc(O.model)}</b> 선택. 마지막 기간(${esc(O.test_from)}~) 적중률 <b>${O.hit}%</b>, 찍기 ${O.base}%.</p>
       ${(O.calib||[]).length?T(O.calib.map(c=>[c.lab,c.n+'건',c.up+'%',pctS(c.ret,2)]),['모델이 말한 상승 확률','표본','실제 상승 비율','실제 5일 평균']):''}
       <p class="vl" style="text-align:left">모델이 "55% 이상"이라고 한 구간에서 실제 상승 비율이 더 높고, "45% 미만" 구간에서 더 낮아야 믿을 만한 모델입니다.</p>`:'')+
      (R&&R.valid&&R.valid.length?`<h5>시장 방향 신호등 (${esc(R.vperiod)})</h5>${T(R.valid.map(v=>[v.kind,v.lab,v.n+'일',pctS(v.f5,2),v.up5+'%',v.f20==null?'-':pctS(v.f20,2)]),['점수','단계','일수','이후 5일 코스피','5일 상승 비율','이후 20일'])}
       <p class="vl" style="text-align:left">시장 폭(20일선 위 종목 비율) 자료가 1년치뿐이라 표본이 적습니다. 참고용으로 보세요.</p>`:''))}
    ${sec('data','📡 데이터 수집 상태 (오늘)',T(st,['출처','결과'])+'<p class="vl" style="text-align:left">모두 무료 공개 데이터입니다. ⚠️가 있으면 그 항목은 오늘 빠졌거나 대체 자료로 계산했습니다.</p>')}
    ${sec('how','전체 흐름',helpHow())}
    ${sec('picks',HELP.picks[0],HELP.picks[1]())}
    ${sec('rules','전략별 기본 조건',[3,1,2].map(k=>`<h5 class="${STRAT[k].c}">${STRAT[k].n}</h5>${RULE(k)}`).join(''))}
    ${['aux','target','hold','signals','macro','outlook','events','brief'].map(k=>sec(k,HELP[k][0],HELP[k][1]())).join('')}
    ${sec('ai','AI 분석 · 뉴스',HELP.ai[1]()+HELP.news[1]())}
    <p class="vl" style="text-align:left;margin-top:20px">이 도구의 모든 결과는 참고 자료이며 수익을 보장하지 않습니다. 투자 판단과 책임은 본인에게 있습니다.</p>`;
}
const SEV={bad:['🔴','수정 필요'],warn:['🟠','점검 권장'],info:['🔵','참고'],ok:['🟢','양호']};
function revPrompt(){
  const rv=D.review,ts=D.tips||[];
  const bt=D.bt,st=(rv?Object.entries(rv.strat).map(([k,v])=>`[실시간] ${STRAT[k].n}: ${v.n}건·${v.days}일`+[1,3,5].map(h=>v['h'+h]?` / ${h}일 평균 ${pctS(v['h'+h].avg,2)}, 시장대비 ${pctS(v['h'+h].ex,2)}, t=${v['h'+h].t}`:'').join('')).join('\n'):'')+
    (bt?'\n'+Object.entries(bt.strat).map(([k,v])=>`[과거 ${bt.start}~${bt.end}] ${STRAT[k].n}: ${v.n}건 · 3일 시장대비 ${v.h3?pctS(v.h3.ex,2):'-'}, t=${v.h3?v.h3.t:'-'}, 판정 ${v.verdict}`).join('\n'):'');
  return `내 주식 대시보드(stock_dashboard.py)의 예측 기록과 실제 결과를 비교한 점검 결과야.\n기록 ${rv?rv.n_snap:0}일치 (${rv?rv.first:''}~${rv?rv.last:''})\n${st}\n`+
    (rv&&rv.ol?`5일 전망: ${rv.ol.n}건 적중 ${rv.ol.hit}%, 흔들림 범위 안 ${rv.ol.inband}%\n`:'')+
    `자동 진단:\n${ts.map(t=>`- [${SEV[t.sev][1]}] ${t.title}: ${t.detail}${t.fix?` → 제안: ${t.fix}`:''}`).join('\n')}\n\n`+
    `이 결과를 보고 프로그램에서 무엇을 어떻게 고치면 좋을지 우선순위대로 정리해 주고, stock_dashboard.py에서 바꿔야 할 부분(조건 기준값, 점수, 함수)을 구체적으로 알려줘.`;
}
function renderRevLink(){
  const rv=D.review,ts=D.tips||[],box=document.getElementById('revlink');
  const last=rv&&rv.daylog&&rv.daylog.length?rv.daylog[rv.daylog.length-1]:null;
  const main=ts.filter(t=>t.sev!=='ok').slice(0,2);
  box.innerHTML=`<div class="bhead"><h2 style="margin:0">🔧 프로그램 점검</h2><a class="btn" href="#review">점검 페이지 열기 →</a></div>
    <div class="rlrow">${last?`<div><span class="vl">어제(${last.day}) 추천 1~3위 ${last.n}종목</span><b class="tn ${last.avg1>=0?'upc':'downc'}">${pctS(last.avg1,2)}</b><span class="vl">시장 ${pctS(last.b1,2)} · 승률 ${last.win1}%</span></div>`:'<div><span class="vl">예측 기록을 쌓는 중입니다.</span></div>'}
      ${main.length?`<ul class="tips mini">${main.map(t=>`<li class="${t.sev}"><span>${SEV[t.sev][0]}</span><b>${esc(t.title)}</b></li>`).join('')}</ul>`:'<p class="vl" style="margin:0">지금은 수정이 필요한 항목이 없습니다.</p>'}</div>`;
}
function renderReview(){
  const g=document.getElementById('reviewview'),rv=D.review,ts=D.tips||[];
  const tipsH=`<ul class="tips">${ts.map(t=>`<li class="${t.sev}"><div class="th"><span>${SEV[t.sev][0]} ${SEV[t.sev][1]}</span><b>${esc(t.title)}</b></div><div class="vl" style="text-align:left">${esc(t.detail)}</div>${t.fix?`<div class="fix">제안 · ${esc(t.fix)}</div>`:''}</li>`).join('')}</ul>`;
  let body='';
  if(rv&&rv.n_snap){
    const L=rv.latest||[],ld=rv.daylog[rv.daylog.length-1];
    const latestH=L.length?`<div class="tablewrap"><table class="revtbl"><thead><tr><th>전략</th><th>종목</th><th>예측 (목표 / 손절)</th><th>시가 매수 → 종가</th><th>시장 대비</th><th>지금까지</th><th>결과</th></tr></thead><tbody>
      ${L.filter(x=>x.rank<=3).map(x=>`<tr class="tn"><td><span class="tag ${STRAT[x.k].c}">${STRAT[x.k].n} ${x.rank}위</span></td><td class="nm"><b>${esc(x.name)}</b><span>${x.code}</span></td>
        <td>${f0(x.t1)} / ${f0(x.stop)}<br><span class="vl">매수가 ${f0(x.entry)}</span></td><td class="${x.r1>=0?'upc':'downc'}"><b>${pctS(x.r1,2)}</b></td><td class="${x.ex1>=0?'pos':'neg'}">${pctS(x.ex1,2)}</td>
        <td class="${x.now>=0?'upc':'downc'}">${pctS(x.now,2)} <span class="vl">(${x.held}일)</span></td><td>${x.out==='목표'?'<span class="tr tg">목표 도달</span>':x.out==='손절'?'<span class="tr tb">손절 도달</span>':'<span class="tr tf">진행 중</span>'}</td></tr>`).join('')}</tbody></table></div>`:'';
    const S=rv.strat,cum=Object.keys(S).length?`<div class="tablewrap"><table class="revtbl"><thead><tr><th>전략</th><th>건수 · 일수</th><th>1일 시장 대비</th><th>3일 시장 대비</th><th>5일 시장 대비</th><th>3일 중앙값</th><th>최대 손실</th><th>t값 (3일)</th><th>상태</th></tr></thead><tbody>
      ${[3,1,2].filter(k=>S[k]).map(k=>{const v=S[k],e=h=>v['h'+h]?`<span class="${v['h'+h].ex>=0?'pos':'neg'}">${pctS(v['h'+h].ex,2)}</span>`:'-';
        return `<tr class="tn"><td><span class="tag ${STRAT[k].c}">${STRAT[k].n}</span></td><td>${v.n}건 · ${v.days}일</td><td>${e(1)}</td><td><b>${e(3)}</b></td><td>${e(5)}</td><td>${v.h3?pctS(v.h3.med,2):'-'}</td><td>${v.h1?pctS(Math.min(v.h1.worst,v.h3?v.h3.worst:99),1):'-'}</td><td>${v.h3&&v.h3.t!=null?v.h3.t:'-'}</td><td>${v.watch?'<span class="tr tf">관찰 중</span>':(v.h3&&v.h3.t>=2?'<span class="tr tg">시장보다 나음</span>':v.h3&&v.h3.t<=-2?'<span class="tr tb">시장보다 못함</span>':'<span class="tr tf">차이 불분명</span>')}</td></tr>`;}).join('')}</tbody></table></div>
      <p class="vl" style="text-align:left">비용 ${rv.cost}% 차감 · 같은 시장 종목 중간값과 비교 · 전략별 ${'30'}건·${'15'}거래일이 쌓이기 전에는 '관찰 중'으로만 표시합니다.</p>`:'';
    const O=rv.ol,olH=O?`<div class="revkpi tn"><div><span class="vl">검증된 예측</span><b>${O.n}건</b></div><div><span class="vl">방향 적중</span><b class="${O.hit>=50?'pos':'neg'}">${O.hit}%</b></div><div><span class="vl">흔들림 범위 안</span><b>${O.inband}%</b><span class="vl">기대 약 68%</span></div>
      <div><span class="vl">'55% 이상'이라 한 때 실제 상승</span><b>${O.up_hi==null?'-':O.up_hi+'%'}</b><span class="vl">${O.n_hi}건</span></div><div><span class="vl">'45% 미만'이라 한 때 실제 상승</span><b>${O.up_lo==null?'-':O.up_lo+'%'}</b><span class="vl">${O.n_lo}건</span></div></div>
      ${(rv.latest_ol||[]).length?`<div class="tablewrap"><table class="revtbl"><thead><tr><th>대상</th><th>예측 (오를 확률)</th><th>예상 흔들림</th><th>실제 5일</th><th>방향</th><th>범위</th></tr></thead><tbody>${rv.latest_ol.map(x=>`<tr class="tn"><td><b>${esc(x.name)}</b></td><td>${x.p}%</td><td>${pctS(x.er,2)} ± ${x.band}%</td><td class="${x.act>=0?'upc':'downc'}">${pctS(x.act,2)}</td><td>${x.hit?'✅ 맞음':'❌ 틀림'}</td><td>${x.inband?'안':'밖'}</td></tr>`).join('')}</tbody></table></div><p class="vl" style="text-align:left">${esc(rv.latest_ol[0].asof)} 아침 예측 → 5거래일 뒤 결과</p>`:''}`:'<p class="vl" style="text-align:left">5거래일이 지난 예측이 아직 없습니다.</p>';
    const R=rv.rk,rkH=R?`<p>위험 50점 이상이던 날(${R.n_hi}일) 다음 5일 코스피 ${R.hi==null?'-':pctS(R.hi,2)} · 25점 미만이던 날(${R.n_lo}일) ${R.lo==null?'-':pctS(R.lo,2)}</p>`:'<p class="vl" style="text-align:left">5거래일이 지난 기록이 아직 없습니다.</p>';
    const log=`<div class="tablewrap"><table class="revtbl"><thead><tr><th>예측한 날 (기준일)</th><th>결과 날</th><th>추천 1~3위 평균</th><th>승률</th><th>시장</th></tr></thead><tbody>${rv.daylog.slice().reverse().map(d=>`<tr class="tn"><td>${d.asof}</td><td>${d.day}</td><td class="${(d.avg1||0)>=0?'upc':'downc'}"><b>${d.avg1==null?'-':pctS(d.avg1,2)}</b></td><td>${d.win1==null?'-':d.win1+'%'}</td><td>${pctS(d.b1,2)}</td></tr>`).join('')}</tbody></table></div>`;
    body=`${ld?`<section class="gsec"><h3>어제 예측 vs 실제 <span class="vl">${ld.asof} 아침 추천 → ${ld.day} 결과</span></h3>${latestH}</section>`:''}
      <section class="gsec"><h3>추천 종목 누적 성적 <span class="vl">${rv.first} ~ ${rv.last} · ${rv.n_snap}일치</span></h3>${cum}</section>
      <section class="gsec"><h3>5일 전망 검증</h3>${olH}</section>
      <section class="gsec"><h3>시장 신호등 검증</h3>${rkH}</section>
      <section class="gsec"><h3>날짜별 기록</h3>${log}</section>`;
  }
  g.innerHTML=`<a class="back" href="#">← 대시보드로 돌아가기</a><h2 style="margin-top:12px">🔧 프로그램 점검 · 수정 제안</h2>
    <p class="vl" style="text-align:left">매일 마지막 장전 갱신(보통 7시 30분) 때의 추천 종목·5일 전망·신호등을 저장하고, 다음 날 아침 실제 결과와 비교합니다. 추천 종목은 '그날 시가에 샀다면'으로 계산합니다.${D.demo?' (미리보기: 과거 시점을 다시 계산한 가상 기록)':''}</p>
    <section class="gsec"><h3>자동 진단 · 수정 제안</h3>${tipsH}
      <div class="aibox aibtn" style="margin-top:10px"><p><b style="color:var(--ink)">🤖 Claude에게 수정 방법 받기</b><br>위 결과가 입력된 채로 Claude가 열립니다. Enter를 누르면 고칠 부분과 방법을 구체적으로 알려드립니다.</p>
      <a class="btn ai" href="${claudeURL(revPrompt(),null)}" target="_blank" rel="noopener">🤖 수정 방법 받기</a></div></section>
    ${body}`;
}
function renderEvents(){
  const box=document.getElementById('events'),E=(D.outlook&&D.outlook.events)||[];
  if(!E.length){box.innerHTML='<div class="empty">앞으로 45일 안에 등록된 일정이 없습니다.</div>';return;}
  const pat=(v,up,lab)=>v==null?'':`<span class="evp"><span class="vl">${lab}</span> ${pct(v,1)} <span class="vl">· ${up}% 상승</span></span>`;
  box.innerHTML=`<div class="tablewrap"><table class="evtbl"><thead><tr><th>날짜</th><th>일정</th><th>관련</th><th>과거 패턴</th></tr></thead><tbody>
    ${E.map(e=>`<tr class="tn ${e.dday<=5?'soon':''}"><td><b>${e.date}</b> <span class="vl">(${e.dow})</span><br><span class="dd">${e.dday===0?'오늘':'D-'+e.dday}</span></td>
      <td><b>${esc(e.name)}</b>${e.tent||e.note?`<br><span class="vl">${e.tent?'예상 날짜':''}${e.note&&!e.tent?esc(e.note):''}</span>`:''}</td>
      <td>${e.secs.map(x=>`<span class="sb">${esc(x)}</span>`).join(' ')}</td>
      <td>${e.st?`${pat(e.st.pre,e.st.pre_up,'일정 전 5일')}<br>${pat(e.st.post,e.st.post_up,'일정 후 5일')}<br><span class="vl">${esc(e.st.target)} 기준 · 과거 ${e.st.n}번</span>`:'<span class="vl">과거 자료 부족</span>'}</td></tr>`).join('')}
    </tbody></table></div>`;
}
function renderOutlook(){
  const O=D.outlook,box=document.getElementById('outlook');
  if(!O){box.innerHTML='<div class="empty">오늘은 지수·섹터 데이터를 받지 못해 5일 전망을 계산하지 못했습니다.</div>';return;}
  const weak=O.hit<=O.base+1;
  const pbar=p=>{const q=Math.max(0,Math.min(100,50+(p-50)*3));return `<div class="pbar" title="50%가 가운데"><i style="left:${Math.min(q,50)}%;width:${Math.abs(q-50)}%" class="${p>=50?'u':'d'}"></i><s></s></div>`;};
  const word=p=>p>=60?'상승 우세':p>=53?'약한 상승':p>48?'방향 불분명':p>40?'약한 하락':'하락 우세';
  const tcls=t=>t==='상승'?'tu':t==='하락'?'td':'tf';
  box.innerHTML=`
    <div class="olk-note ${weak?'weak':''}">검증 적중률 <b>${O.hit}%</b> <span class="vl">· 찍기 ${O.base}%</span> ${weak?'<span class="chip">방향은 참고만 · 흔들림 폭 위주로</span>':'<span class="chip">참고용</span>'}</div>
    <div class="olk-idx">${O.idx.map(o=>`<div class="oi"><div class="oh"><b>${o.name}</b><span class="tn">${f0(o.last)}</span><span class="tr ${tcls(o.trend)}">${o.trend} 추세</span></div>
      ${cone(o)}
      <div class="op tn"><div><span class="vl">5일 뒤 오를 확률</span><b class="${o.p>=50?'upc':'downc'}">${o.p}%</b> <small>${word(o.p)}</small></div>
      <div><span class="vl">과거 비슷한 때 평균</span><b>${pctS(o.er,2)}</b></div><div><span class="vl">예상 흔들림 (68%)</span><b>±${o.band}%</b></div></div></div>`).join('')}</div>
    <h3 class="sec">섹터별 추세와 5일 전망</h3>
    <div class="tablewrap"><table class="sectbl"><thead><tr><th>섹터</th><th>최근 60일</th><th>현재 추세</th><th>5일 / 20일</th><th>코스피 대비</th><th>5일 뒤 오를 확률</th><th>예상 흔들림</th><th>다가오는 일정</th></tr></thead><tbody>
    ${O.sec.map(o=>`<tr class="tn"><td class="nm"><b>${esc(o.name)}</b><span>${esc(o.etf)}</span></td><td>${miniSpark(o.spark)}</td>
      <td><span class="tr ${tcls(o.trend)}">${o.trend}</span></td><td>${pct(o.r5,1)} / ${pct(o.r20,1)}</td><td>${o.rs==null?'-':pct(o.rs,1)}</td>
      <td><div class="pcell">${pbar(o.p)}<b class="${o.p>=50?'upc':'downc'}">${o.p}%</b></div></td><td>±${o.band}%</td><td class="vl">${esc(o.next_ev||'')}</td></tr>`).join('')}</tbody></table></div>
`;
}
function miniSpark(arr){const c=arr.filter(x=>x!=null);if(c.length<2)return '';const W=90,H=26,lo=Math.min(...c),hi=Math.max(...c);
  return `<svg viewBox="0 0 ${W} ${H}" width="90" height="26" aria-hidden="true"><polyline points="${c.map((v,i)=>(i*(W-2)/(c.length-1)+1)+','+(H-2-(v-lo)/((hi-lo)||1)*(H-4))).join(' ')}" class="ms ${c[c.length-1]>=c[0]?'u':'d'}"/></svg>`;}
function renderPulse(){
  const B=D.breadth;
  const regime=B.above20>=60?['상승 우위','추세 눌림목·거래량 돌파가 잘 맞는 장입니다.']:B.above20>=45?['중립','추세 눌림목 위주로, 종목 수를 줄이세요.']:['약세','과매도 반등 위주로, 비중을 줄이세요.'];
  const g=(t,v)=>`<div><dt>${t}</dt><dd class="tn">${v==null?'-':v.toFixed(0)+'%'}</dd><div class="gauge"><i style="width:${v||0}%"></i></div></div>`;
  document.getElementById('pulse').innerHTML=g('20일선 위 종목',B.above20)+g('60일선 위 종목',B.above60)+g('상승 마감 종목',B.rising)+
    `<div class="regime"><dt>시장 분위기</dt><dd>${regime[0]}</dd><p>${regime[1]} 아래에서 <b>${STRAT[D.default_tab].n}</b> 탭을 먼저 보여드립니다.</p></div>`;
}
let current=D.default_tab;
function renderTabs(){
  document.getElementById('tabs').innerHTML=D.tabs.map(t=>`<button class="tab" role="tab" style="--tc:var(${STRAT[t.k].v})" aria-selected="${t.k===current}" data-k="${t.k}">${t.name}${t.k===D.default_tab?'<span class="rec">오늘 추천</span>':''}</button>`).join('')+(D.yt&&D.yt.videos.length?`<a class="tab ytab" href="#yt">📺 유튜브</a>`:'');
  document.querySelectorAll('.tab[data-k]').forEach(b=>b.onclick=()=>{current=+b.dataset.k;renderTabs();renderPanel();});
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
  <div class="rr"><span>손익비 <b>1 : ${t.rr==null?'-':t.rr.toFixed(1)}</b> <button class="q" data-help="target" aria-label="설명 보기">?</button></span><span class="vl">${esc(t.note||'')}</span></div></dl>`;}
function aiHTML(x){
  return aiSlot({prompt:x.prompt,code:x.code,kind:'pick',key:`ai:${D.asof}:${x.code}:p${x.strat}`,score:x.score,pre:x.ai||null});
}
function pickCard(x){
  return `<article class="pick">
      <div class="info">
        <span class="tag ${STRAT[x.strat].c}">${STRAT[x.strat].n}${x.rank?' '+x.rank+'위':''}</span>
        <div class="name">${esc(x.name)} <span class="code tn">${x.code} ${x.market}</span></div>
        <div class="price tn">${f0(x.close)}<span>${pct(x.chg)}</span></div>
        ${storyHTML(x)}
        ${scoreHTML(x)}
        ${tgHTML(x)}
        <details class="mini-fold"><summary>점수 계산 자세히 (기본 조건 · 보조 지표)</summary>
          <div class="lbl">기본 조건 <button class="q" data-help="rules${x.strat}" aria-label="설명 보기">?</button></div>${checksHTML(x.checks)}
          <div class="lbl">보조 지표 가감점 <button class="q" data-help="aux" aria-label="설명 보기">?</button></div>${auxHTML(x.aux)}
          <dl class="kv tn"><div><dt>거래대금</dt><dd>${won(x.amount)}</dd></div><div><dt>시가총액</dt><dd>${won(x.mcap)}</dd></div><div><dt>거래량</dt><dd>${x.vr==null?'-':x.vr.toFixed(1)+'배'}</dd></div></dl>
        </details>
      </div>
      <div class="right">${x.series?'<div class="chartbox"></div>':'<p class="vl" style="text-align:left">차트는 1~6위만 제공합니다.</p>'}${newsHTML(x.code,4)}${aiHTML(x)}</div>
    </article>`;
}
let SHEET_DRAW=null;
const sheet=document.getElementById('sheet');
function openSheet(html,draw){
  document.getElementById('sheetBody').innerHTML=html;SHEET_DRAW=draw||null;
  if(!sheet.open)sheet.showModal();
  document.getElementById('sheetBody').scrollTop=0;sheet.querySelector('.shin').scrollTop=0;
  requestAnimationFrame(()=>SHEET_DRAW&&SHEET_DRAW());
}
document.getElementById('sheetClose').onclick=()=>sheet.close();
sheet.addEventListener('click',e=>{if(e.target===sheet)sheet.close();});
function openPick(k,i){
  const x=D.tabs.find(t=>t.k===k).stocks[i];
  openSheet(pickCard(x),()=>{const b=sheet.querySelector('.chartbox');if(b&&x.series)drawChart(b,x.series);});
}
function renderPanel(){
  const tab=D.tabs.find(t=>t.k===current),st=tab.stocks;
  document.getElementById('tabdesc').innerHTML=(D.risk&&D.risk.score>=50?`<span class="rwarn">⚠️ 시장 위험 '${D.risk.level}' · 매수 금액을 크게 줄이세요</span><br>`:'')+esc(STRAT[current].d);
  if(!st.length){document.getElementById('panel').innerHTML='<div class="empty">오늘은 이 전략에 맞는 종목이 없습니다.</div>';return;}
  const row=(x,i)=>`<button class="prow ${i<3?'top':''}" data-pick="${tab.k}:${i}">
      <span class="rk tn">${x.rank}</span>
      <span class="pn"><b>${esc(x.name)}</b><small class="tn">${f0(x.close)} ${pct(x.chg)}</small></span>
      ${x.series?`<span class="sp">${miniSpark(x.series.c.slice(-40))}</span>`:'<span class="sp"></span>'}
      <span class="pt tn"><small>목표</small><b class="upc">${pctS(x.tg.t1p)}</b></span>
      <span class="pt tn"><small>손절</small><b class="downc">${pctS(x.tg.sp)}</b></span>
      <span class="ps tn"><i style="width:${x.score}%"></i><b>${x.score}</b></span></button>`;
  document.getElementById('panel').innerHTML=
    `<div class="plist">${st.slice(0,3).map(row).join('')}</div>`+
    (st.length>3?`<h3 class="sec">지켜볼 종목</h3><div class="plist">${st.slice(3).map((x,i)=>row(x,i+3)).join('')}</div>`:'')+
    `<p class="vl" style="text-align:left;margin-top:8px">종목을 누르면 차트·조건·뉴스·AI 분석이 열립니다.</p>`;
}
function drawAll(){if(sheet.open&&SHEET_DRAW)SHEET_DRAW();}

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
  rerender();
}
function findCode(q){q=q.trim();if(!q)return null;if(NAMES[q])return q;const m=q.match(/\((\w{6})\)/);if(m&&NAMES[m[1]])return m[1];
  if(/^[0-9A-Z]{6}$/.test(q)&&AI_URL)return q;
  if(byName[q])return byName[q];const k=Object.keys(byName).find(n=>n.replace(/\s/g,'')===q.replace(/\s/g,''));return k?byName[k]:null;}

function analyzeHolding(code,qty,avg){
  const [name,mk,c,chg,ma20,ma60,slope,bbu,bbl,rsi,atr0,h0,h1,hi20,lo20,hi60,res,sup,pick,rs,fhi,flo,fup,patt]=snapOf(code);
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
  if(fup!=null&&fhi&&flo&&fhi>flo){
    const rr=[.382,.5,.618],lv=rr.map(r=>[r,fup?fhi-(fhi-flo)*r:flo+(fhi-flo)*r]);
    const below=lv.filter(x=>x[1]<c).sort((a,b)=>b[1]-a[1])[0],above=lv.filter(x=>x[1]>c).sort((a,b)=>a[1]-b[1])[0];
    why.push(`피보나치(최근 120일 ${fup?'상승':'하락'} 파동): `+[below?`아래 지지 ${f0(below[1])}원(${(below[0]*100).toFixed(1)}%)`:'',above?`위 저항 ${f0(above[1])}원(${(above[0]*100).toFixed(1)}%)`:''].filter(Boolean).join(', ')+'.');
  }
  if(patt&&patt!=='없음')why.push(`세력 패턴(최근 20일): ${patt}${patt.includes('이탈')?' — 큰손이 빠져나간 흔적일 수 있어 주의가 필요합니다.':' — 큰손이 사 모은 흔적으로 볼 수 있습니다.'}`);
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
function holdCard(h,saved){
  const act=saved?`<button class="btn ghost sm danger" data-rm="${h.code}">🗑 삭제</button>`:`<button class="btn ghost sm" data-save="1">＋ 저장</button>`;
  if(!snapOf(h.code)){
    const nm=esc(NAMES[h.code]||h.code),st=PENDING[h.code];
    if(AI_URL&&!st)setTimeout(()=>fetchSnap(h.code),0);
    const msg=!AI_URL?'오늘 아침 시세 수집에서 빠진 종목이라 차트 점검을 할 수 없습니다. 내일 아침 갱신 때 다시 확인해 주세요. 아래 AI 분석은 지금도 쓸 수 있습니다.':
      (typeof st==='string'&&st.startsWith('err:'))?'시세를 불러오지 못했습니다: '+esc(st.slice(4)):'시세를 불러오는 중입니다…';
    const bp=`한국 주식 ${NAMES[h.code]||h.code}(${h.code})을 ${f0(h.qty)}주, 평균 ${f0(h.avg)}원에 보유하고 있어. `+
      `현재 주가와 최근 차트 흐름(20일선·60일선, RSI, 거래량), 최근 한 달 국내 뉴스·공시와 Reuters·Bloomberg 같은 해외 영문 기사(영문은 한국어로 요약)를 웹에서 검색해서 `+
      `최근 한 달 주요 이슈를 날짜순으로 정리하고, 보유 / 일부 매도 / 전량 매도 중 어떤 전략이 좋을지 손절가·목표가, 근거 3가지, 리스크, 참고 기사 링크와 함께 알려줘.`;
    return `<div class="hold"><div><div class="hdr"><div><b>${nm}</b> <span class="code">${esc(h.code)}</span> · ${f0(h.qty)}주 · 평단 ${f0(h.avg)}</div><span class="acts">${act}</span></div>
      <p class="vl" style="text-align:left;margin:8px 0 0">${msg}</p></div><div>${aiSlot({prompt:bp,code:h.code,kind:'hold',key:`ai:${D.asof}:${h.code}:hx`,tone:'mid'})}</div></div>`;
  }
  const r=analyzeHolding(h.code,h.qty,h.avg);
  const row=(l,p,why)=>`<tr><td>${l}<br><small class="vl" style="text-align:left">${why}</small></td><td>${f0(p)}</td><td>${pctS((p/r.c-1)*100)}</td><td class="${p>=r.avg?'upc':'downc'}">${(p-r.avg)*r.qty>=0?'+':''}${f0((p-r.avg)*r.qty)}원</td></tr>`;
  return `<article class="hold">
    <div>
      <div class="hdr"><div><div class="name" style="margin:0">${esc(r.name)} <span class="code tn">${r.code} ${r.mk}</span></div>
        <div class="price tn">${f0(r.c)}<span>${pct(r.chg)}</span></div></div><span class="acts">${act}</span></div>
      <dl class="pos-grid tn">
        <div><dt>보유</dt><dd>${f0(r.qty)}주 · 평단 ${f0(r.avg)}</dd></div>
        <div><dt>평가금액</dt><dd>${f0(r.val)}원</dd></div>
        <div><dt>평가손익</dt><dd class="${r.pl>=0?'upc':'downc'}">${r.plAmt>=0?'+':''}${f0(r.plAmt)}원</dd></div>
        <div><dt>수익률</dt><dd class="${r.pl>=0?'upc':'downc'}">${pctS(r.pl,2)}</dd></div>
      </dl>
      <div class="hv"><div class="lab">📊 차트 기준 전략 <button class="q" data-help="hold" aria-label="설명 보기">?</button></div><div class="val ${r.tone}">${r.v}</div></div>
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
}
let CHECK=null;
const isSaved=code=>loadH().some(h=>h.code===code);
function openHold(h){
  const draw=()=>{};
  openSheet(holdCard(h,isSaved(h.code)),draw);
}
function renderMine(){
  const list=loadH(),box=document.getElementById('mylist');
  if(!list.length){box.innerHTML='<div class="empty">저장한 종목이 없습니다. 위에 입력하고 <b>저장</b>을 누르면 매일 자동으로 다시 점검합니다.</div>';return;}
  let tot=0,cost=0;
  const rows=list.map(h=>{
    if(!snapOf(h.code)){if(AI_URL&&!PENDING[h.code])setTimeout(()=>fetchSnap(h.code),0);
      return `<div class="mrow"><button class="mmain" data-hold="${h.code}"><span class="pn"><b>${esc(NAMES[h.code]||h.code)}</b><small>${f0(h.qty)}주 · 평단 ${f0(h.avg)}</small></span><span class="vl">데이터 없음</span></button><button class="del" data-rm="${h.code}" aria-label="${esc(NAMES[h.code]||h.code)} 삭제">✕</button></div>`;}
    const r=analyzeHolding(h.code,h.qty,h.avg);tot+=r.val;cost+=r.avg*r.qty;
    return `<div class="mrow"><button class="mmain" data-hold="${h.code}">
        <span class="pn"><b>${esc(r.name)}</b><small class="tn">${f0(r.qty)}주 · 평단 ${f0(r.avg)} · 현재 ${f0(r.c)}</small></span>
        <span class="mpl tn"><b class="${r.pl>=0?'upc':'downc'}">${pctS(r.pl,1)}</b><small class="${r.plAmt>=0?'upc':'downc'}">${r.plAmt>=0?'+':''}${f0(r.plAmt)}원</small></span>
        <span class="chip ${r.tone}">${r.v}</span></button>
      <button class="del" data-rm="${h.code}" aria-label="${esc(r.name)} 삭제">✕</button></div>`;
  }).join('');
  const pl=tot-cost;
  box.innerHTML=`<div class="msum tn"><span>저장 ${list.length}종목</span><span>평가 ${f0(tot)}원</span><b class="${pl>=0?'upc':'downc'}">${pl>=0?'+':''}${f0(pl)}원</b></div><div class="mlist">${rows}</div>
    <p class="vl" style="text-align:left;margin-top:6px">종목을 누르면 자세한 점검이 열리고, ✕를 누르면 목록에서 지워집니다.</p>`;
}
function rerender(){renderMine();if(location.hash==='#home'||location.hash==='')renderHome();}
function weather(){const R=D.risk;if(!R)return ['🌫️','정보 없음','신호등 자료를 받지 못했습니다','mid'];const s=R.score,u=R.up;
  if(s>=75)return ['⛈️','폭풍','신규 매수는 쉬고 현금을 늘리세요','bad'];
  if(s>=50)return u>=35?['🌦️','비 그칠 조짐','바닥 신호가 보여요 · 아주 조금씩만','mid']:['🌧️','비','신규 매수는 평소의 절반 이하로','bad'];
  if(s>=25)return ['⛅','구름','조금만 사고 손절가는 꼭 지키기','mid'];
  return u>=35?['☀️','맑음','매수하기 좋은 날','ok']:['🌤️','대체로 맑음','평소처럼 해도 괜찮아요','ok'];}
function renderHome(){
  const box=document.getElementById('v-home'),W=weather(),R=D.risk||{},q=(D.brief&&D.brief.quick)||[];
  const tab=D.tabs.find(t=>t.k===D.default_tab)||D.tabs[0],top=(tab?tab.stocks:[]).slice(0,3);
  const saved=loadH(),evs=((D.outlook&&D.outlook.events)||[]).filter(e=>e.dday<=7).slice(0,4),secs=(D.outlook&&D.outlook.sec)||[];
  const heat=x=>{const v=Math.max(-8,Math.min(8,x.r5||0)),a=(Math.abs(v)/8*.55+.08).toFixed(2);return v>=0?`rgba(212,42,60,${a})`:`rgba(29,91,216,${a})`;};
  const mine=saved.slice(0,4).map(h=>{const s=snapOf(h.code);if(!s)return `<button class="hrow" data-hold="${h.code}"><b>${esc(NAMES[h.code]||h.code)}</b><span class="vl">데이터 없음</span></button>`;
    const r=analyzeHolding(h.code,h.qty,h.avg);return `<button class="hrow" data-hold="${h.code}"><b>${esc(r.name)}</b><span class="tn ${r.pl>=0?'upc':'downc'}">${pctS(r.pl,1)}</span><span class="chip ${r.tone}">${r.v}</span></button>`;}).join('');
  box.innerHTML=`
    <div class="hgrid">
      <a class="card wx ${W[3]}" href="#market">
        <span class="wxi" aria-hidden="true">${W[0]}</span>
        <span class="wxb"><small>오늘 시장 날씨</small><b>${W[1]}</b><span>${W[2]}</span></span>
        <span class="wxs tn"><small>하락 위험</small><b>${R.score??'-'}</b><small>상승 신호</small><b>${R.up??'-'}</b></span>
      </a>
      ${q.length?`<div class="card strip tn">${q.map(x=>`<div><small>${esc(x.n)}</small><b class="${x.chg>=0?'upc':'downc'}">${pctS(x.chg,2)}</b></div>`).join('')}</div>`:''}
      <div class="card">
        <div class="ch"><h3>오늘의 TOP 3</h3>${tab?`<span class="tag ${STRAT[tab.k].c}">${tab.name}</span>`:''}</div>
        ${top.length?`<div class="plist">${top.map((x,i)=>`<button class="prow top" data-pick="${tab.k}:${i}"><span class="rk tn">${i+1}</span><span class="pn"><b>${esc(x.name)}</b><small class="tn">${f0(x.close)} ${pct(x.chg)}</small></span><span class="pt tn"><small>목표</small><b class="upc">${pctS(x.tg.t1p)}</b></span><span class="pt tn"><small>손절</small><b class="downc">${pctS(x.tg.sp)}</b></span></button>`).join('')}</div>`:'<p class="vl" style="text-align:left">오늘은 추천 종목이 없습니다.</p>'}
        <a class="more" href="#picks">전략별 추천 전체 →</a>
      </div>
      <div class="card">
        <div class="ch"><h3>내 종목</h3><a class="more" href="#mine">${saved.length?'관리 →':'＋ 추가'}</a></div>
        ${saved.length?`<div class="hlist">${mine}</div>${saved.length>4?`<a class="more" href="#mine">외 ${saved.length-4}종목 →</a>`:''}`:'<p class="vl" style="text-align:left;margin:0">보유 종목을 저장하면 여기서 매일 보유·매도 신호를 보여드려요.</p>'}
      </div>
      <div class="card">
        <div class="ch"><h3>이번 주 일정</h3><a class="more" href="#market">전체 →</a></div>
        ${evs.length?`<ul class="evl">${evs.map(e=>`<li><span class="dd tn">${e.dday===0?'오늘':'D-'+e.dday}</span><b>${esc(e.name)}</b><span class="vl">${esc(e.secs[0]||'')}</span></li>`).join('')}</ul>`:'<p class="vl" style="text-align:left;margin:0">이번 주 큰 일정이 없습니다.</p>'}
      </div>
      ${ytCardHTML()}
      ${D.gainers&&D.gainers.items.length?`<a class="card glink" href="#market"><span>🚀 어제 급등주 ${D.gainers.items.length}종목 해부</span><span class="vl">${D.gainers.items.slice(0,3).map(g=>esc(g.name)+' '+pctS(g.chg,1)).join(' · ')}</span></a>`:''}
      ${secs.length?`<div class="card wide">
        <div class="ch"><h3>섹터 지도 <small class="vl">최근 5일</small></h3><a class="more" href="#market">5일 전망 →</a></div>
        <div class="heat">${secs.slice().sort((a,b)=>(b.r5||0)-(a.r5||0)).map(x=>`<div class="ht2" style="background:${heat(x)}"><b>${esc(x.name)}</b><span class="tn">${pctS(x.r5,1)}</span></div>`).join('')}</div>
      </div>`:''}
    </div>`;
}
function route(){
  let h=location.hash||'#home';if(h==='#my')h='#mine';
  const views={'#home':'v-home','#picks':'v-picks','#mine':'v-mine','#market':'v-market','#more':'v-more','#review':'reviewview','#yt':'v-yt'};
  const id=h.startsWith('#guide')?'guideview':(views[h]||'v-home');
  document.querySelectorAll('.view').forEach(v=>v.hidden=v.id!==id);
  const tabKey=id==='guideview'||id==='reviewview'?'more':id==='v-yt'?'picks':id.replace('v-','');
  document.querySelectorAll('#tabbar a').forEach(a=>a.classList.toggle('on',a.dataset.v===tabKey));
  if(sheet.open)sheet.close();
  if(id==='v-home')renderHome();
  else if(id==='v-mine')renderMine();
  else if(id==='v-yt')renderYt();
  else if(id==='reviewview')renderReview();
  else if(id==='guideview'){renderGuide();const t=h.length>6&&document.getElementById(h.slice(1));if(t){t.scrollIntoView();return;}}
  scrollTo(0,0);
}
addEventListener('hashchange',route);
document.addEventListener('click',e=>{
  const yv=e.target.closest('[data-vid]');if(yv){openVideo(+yv.dataset.vid);return;}
  const ys=e.target.closest('[data-yts]');if(ys){openYtStock(ys.dataset.yts);return;}
  const pk=e.target.closest('[data-pick]');if(pk){const [k,i]=pk.dataset.pick.split(':').map(Number);openPick(k,i);return;}
  const hd=e.target.closest('[data-hold]');if(hd){const h=loadH().find(x=>x.code===hd.dataset.hold);if(h)openHold(h);return;}
  const rm=e.target.closest('[data-rm]');
  if(rm){const code=rm.dataset.rm,nm=NAMES[code]||code;if(!confirm(`${nm}을(를) 내 종목에서 삭제할까요?`))return;
    saveH(loadH().filter(h=>h.code!==code));if(sheet.open)sheet.close();rerender();return;}
  const sv=e.target.closest('[data-save]');
  if(sv&&CHECK){const l=loadH().filter(h=>h.code!==CHECK.code);l.unshift(CHECK);saveH(l);rerender();openHold(CHECK);return;}
});
function initHolds(){
  const dl=document.getElementById('stocklist');
  dl.innerHTML=Object.entries(NAMES).map(([code,n])=>`<option value="${esc(n)} (${code})"></option>`).join('');
  const err=document.getElementById('fErr');
  const read=()=>{
    const code=findCode(document.getElementById('fStock').value),qty=+document.getElementById('fQty').value,avg=+document.getElementById('fAvg').value;
    const bad=!code?'종목을 찾지 못했습니다. 목록에서 고르거나 6자리 코드로 입력해 주세요.':!(qty>0)?'수량을 입력해 주세요.':!(avg>0)?'평균 매입가를 입력해 주세요.':'';
    err.hidden=!bad;err.textContent=bad;return bad?null:{code,qty,avg};
  };
  const clear=()=>['fStock','fQty','fAvg'].forEach(id=>document.getElementById(id).value='');
  document.getElementById('fCheck').onclick=()=>{const h=read();if(!h)return;CHECK=h;openHold(h);};
  document.getElementById('fSave').onclick=()=>{const h=read();if(!h)return;CHECK=h;const l=loadH().filter(x=>x.code!==h.code);l.unshift(h);saveH(l);clear();rerender();openHold(h);};
  renderMine();
}

/* ───── 시작 ───── */
document.getElementById('meta').textContent=`${D.asof} 종가 · ${D.generated.slice(5)} 갱신`;
document.getElementById('demo').hidden=!D.demo;
document.getElementById('refresh').onclick=()=>location.reload();
renderBrief();renderScore();renderRevLink();checkFresh();setInterval(checkFresh,10*60*1000);renderRisk();renderMacro();renderOutlook();renderEvents();
renderGainers();renderTabs();renderPanel();initHolds();route();
let rt;addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(drawAll,150);});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()

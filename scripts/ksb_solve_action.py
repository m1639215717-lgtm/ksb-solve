#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
考试宝滑块自动过验 + 题库补全导出   (GitHub Actions / 标准Linux 运行版)
========================================================================
设计:
  1) Playwright headless Chromium 打开题库页 (考试宝触验证码)
  2) 抓取验证码背景图 + 拼图小块 (220x220 背景 / 124x124 拼图)
  3) OpenCV 模板匹配(完整版) 或 边缘列扫描 定位缺口中心X
  4) 人类式拖拽轨迹拖动滑块
  5) 过验后拿 token -> 补全加密区题目 (复用 /questions/* 接口)
  6) 导出 .xlsx / .json 到 output/

运行:
  export KSB_USE_TEMPLATE=1   (启用完整模板匹配通道)
  python3 scripts/ksb_solve_action.py --paperid 14581066 --method template
"""
import argparse, base64, hashlib, json, os, random, re, sys, time, uuid
import urllib.parse, urllib.request
import numpy as np

# ---------- 环境 ----------
OUT = os.path.abspath("output")
os.makedirs(OUT, exist_ok=True)
use_template = os.environ.get("KSB_USE_TEMPLATE", "0") in ("1", "true", "yes")

import cv2
CV_MT_OK = use_template  # 在 ubuntu 完整 build 上 matchTemplate 可用

# 测试: 安全起见, 崩溃配置时退回 edge
def _mt_ok():
    try:
        g = (np.random.rand(40, 40) * 255).astype("uint8")
        cv2.matchTemplate(g, g[4:36, 4:36], cv2.TM_CCOEFF_NORMED)
        return True
    except Exception:
        return False

if CV_MT_OK:
    try:
        CV_MT_OK = _mt_ok()
    except BaseException:
        CV_MT_OK = False

# ---------- 考试宝接口 ----------
SECRET = "12b6bb84e093532fb72b4d65fec3f00b"
BASE = "https://api.ankianki.com"
UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 16_3 like Mac OS X) "
      "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Mobile/15E148 Safari/604.1")


def _sign(path, uid, ts):
    return hashlib.md5((SECRET + uid + path + ts + SECRET).encode()).hexdigest()


def ksb(path, data=None, uid=None, token=None):
    uid = uid or str(uuid.uuid4()).lower()
    ts = str(int(time.time() * 1000))
    url = BASE + path
    body = urllib.parse.urlencode(data or {}).encode()
    h = {
        "User-Agent": UA, "Accept": "application/json,*/*", "version": "2.4.6",
        "timestamp": ts, "client-identifier": uid, "request-id": str(uuid.uuid4()).upper(),
        "Sign": _sign(path, uid, ts),
        "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
    }
    if token:
        h["Authorization"] = token
    req = urllib.request.Request(url, data=body, headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))


# ---------- 缺口检测 ----------
def locate_gap(bg_bgr, piece_bgr):
    global CV_MT_OK
    bg = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
    pw = piece_bgr.shape[1]
    result = {"x": None, "method": "???"}
    if CV_MT_OK:
        try:
            edges_bg = cv2.Canny(bg, 50, 150)
            edges_pc = cv2.Canny(cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2GRAY), 50, 150)
            res = cv2.matchTemplate(edges_bg.astype(np.float32),
                                    edges_pc.astype(np.float32), cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            if mv > 0.2:
                result = {"x": ml[0] + pw // 2, "method": "template", "score": round(float(mv), 3)}
                return result
        except BaseException:
            CV_MT_OK = False
    # edge 兜底: 缺口 = 背景中持续变暗的竖带中心 (实测缺口 x≈164)
    H, W = bg.shape
    y0, y1 = int(H*0.35), int(H*0.85)
    region = bg[y0:y1, :]
    colmean = region.mean(axis=0)
    thr = 190.0
    dark = [i for i, v in enumerate(colmean) if v < thr]
    groups = []
    for x in dark:
        if groups and x - groups[-1][-1] <= 3:
            groups[-1].append(x)
        else:
            groups.append([x])
    best = None
    for grp in groups:
        if len(grp) >= 4:
            x0, x1 = grp[0], grp[-1]
            if 20 < x0 and x1 < W - 20:
                c = (x0 + x1) // 2
                if best is None or len(grp) > best[0]:
                    best = (len(grp), c)
    best_x = best[1] if best else int(np.argmin(colmean))
    result = {"x": best_x, "method": "edge"}
    return result


def human_track(start, end, dur=0.9):
    """生成人类似的 (x, y) 轨迹点列表。含水平缓动 + 垂直抖动 + 过冲回弹。
    start/end 是水平像素; y 在基线上有 ±2px 抖动(真人拖动不完全水平)。"""
    np_rng = np.random.default_rng()
    steps = max(20, int(dur * 120))
    pts = [(start, 0.0)]
    for i in range(1, steps + 1):
        t = i / steps
        # easeInOut: 起慢-中快-末慢, 更接近真人
        if t < 0.5:
            eased = start + (end - start) * (2 * t * t)
        else:
            u = 2 * t - 1
            eased = start + (end - start) * (1 - ((-(u*u) + 2*u) * 0.5))
        # 水平抖动(前段小, 末段趋零)
        eased += float(np_rng.uniform(-2.5, 2.5)) * (1 - t) * (0.6 + 0.4*abs(t-0.5))
        # 垂直抖动 ±2px
        yj = float(np_rng.uniform(-2.2, 2.2)) * (1 - t)
        pts.append((max(start, int(eased)), yj))
    # 过冲回弹(水平 + 垂直微动)
    over = random.randint(3, 6)
    pts.append((pts[-1][0] + over, float(np_rng.uniform(-1,1))))
    pts.append((pts[-1][0] - over, float(np_rng.uniform(-1,1))))
    pts.append((end, 0.0))
    return pts


# ---------- 主流程 ----------
def run(paperid, method):
    from playwright.sync_api import sync_playwright
    url = f"https://www.kaoshibao.com/ti/{paperid}"
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=os.environ.get("KSB_HEADLESS", "0") == "1",
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )
        ctx = browser.new_context(viewport={"width": 1280, "height": 960},
                                  user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")
        page = ctx.new_page()
        page.goto(url, wait_until="networkidle", timeout=90000)
        try:
            page.wait_for_selector("#tCaptchaDyMainWrap", timeout=15000)
        except Exception:
            print("验证码未自动触发，尝试点击.", flush=True)
            for sel in ["text=开始", "text=进入练习", "text=顺序练习", "text=专项练习"]:
                try:
                    page.click(sel, timeout=3000); break
                except Exception:
                    pass
            time.sleep(2)
            try:
                page.wait_for_selector("#tCaptchaDyMainWrap", timeout=8000)
            except Exception:
                print("无验证码或已放行，尝试直接取题库", flush=True)

        # 抓背景 / 拼图
        def get_img(src_expr):
            b64 = page.eval_on_selector_all(
                "#tCaptchaDyMainWrap img", src_expr)
            if not b64:
                return None
            data = base64.b64decode(b64.split(",", 1)[1])
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

        bg = get_img("imgs=>{let e=imgs.find(i=>i.naturalWidth===220&&i.naturalHeight===220);return e?e.src:null}")
        piece = get_img("imgs=>{let e=imgs.find(i=>i.naturalWidth===124&&i.naturalHeight===124);return e?e.src:null}")
        print(f"背景={bg.shape if bg is not None else None} 拼图={piece.shape if piece is not None else None}", flush=True)

        gap = locate_gap(bg, piece) if bg is not None else {"x": None}
        print("缺口定位:", gap, flush=True)
        # 保存真实背景/拼图供离线校正与核查
        if bg is not None:
            cv2.imwrite(os.path.join(OUT, "bg_real.png"), bg)
        if piece is not None:
            cv2.imwrite(os.path.join(OUT, "piece_real.png"), piece)
        if gap["x"] is None:
            print("无法定位缺口", flush=True)
            browser.close(); return

        # 保存验证码区域截图，便于诊断缺口/滑块实际位置
        try:
            page.screenshot(path=os.path.join(OUT, "captcha_page.png"), full_page=False)
            print("已保存页面截图 output/captcha_page.png", flush=True)
        except Exception as e:
            print("截图失败:", repr(e)[:60], flush=True)

        # 定位可拖拽滑块 — 多选择器容错，动态找几何有效的滑块元素
        def _box(sel):
            el = page.query_selector(sel)
            if not el:
                return None
            return el.bounding_box()

        def find_valid_slider():
            for sel in [
                ".tencent-captcha-dy__slider-block",
                ".tencent-captcha-dy__slider-groove",
                ".tencent-captcha-dy__slider",
                ".tencent-captcha-dy__opera-area",
            ]:
                try:
                    b = _box(sel)
                    if b and b.get("width") and b.get("height") and b.get("width") > 4:
                        return b
                except Exception:
                    pass
            return None

        slider_box = find_valid_slider()
        if not slider_box:
            print("❌ 找不到可拖拽滑块元素(可能验证码未完全加载或页面状态异常)", flush=True)
            # 打印容器内所有元素的 bounding_box 供诊断
            try:
                page.eval_on_selector_all("#tCaptchaDyMainWrap *",
                    "els=>els.map(e=>({c:e.className,w:e.getBoundingClientRect().width,h:e.getBoundingClientRect().height,x:e.getBoundingClientRect().x})).filter(o=>o.w>5&&o.h>5).slice(0,40) ")
            except Exception as e:
                print("enum err", repr(e)[:60], flush=True)
            browser.close(); return

        # 背景图显示宽度换算拖动距离
        cb_el = page.query_selector(".tencent-captcha-dy__image-area")
        cb = 220.0
        if cb_el:
            b = cb_el.bounding_box()
            if b and b["width"]:
                cb = b["width"]
        ratio = cb / 220.0

        # 生成候选缺口位置(基于已检出的gap.x ± 偏移量, 加扫描点)
        base_x = gap.get("x") or 190
        candidates = []
        for off in [0, -8, 8, -16, 16, -24, 24, -32, 32, -12, 12, -4, 4]:
            candidates.append(base_x + off)
        # 再补覆盖不同猜测区间
        candidates += [198, 205, 182, 176, 168, 160, 150, 145]
        candidates = [c for c in candidates if 20 < c < 205]
        # 去重保序
        seen = set(); cands=[]
        for c in candidates:
            if c not in seen:
                seen.add(c); cands.append(c)

        # 定位可拖拽滑块
        def _box(sel):
            el = page.query_selector(sel)
            if not el: return None
            return el.bounding_box()
        def find_valid_slider():
            for sel in [".tencent-captcha-dy__slider-block",
                        ".tencent-captcha-dy__slider-groove",
                        ".tencent-captcha-dy__slider",
                        ".tencent-captcha-dy__opera-area"]:
                try:
                    b = _box(sel)
                    if b and b.get("width") and b.get("height") and b.get("width") > 4:
                        return b
                except Exception:
                    pass
            return None
        slider_box = find_valid_slider()
        if not slider_box:
            print("❌ 找不到滑块元素", flush=True)
            browser.close(); return
        sx = slider_box["x"] + slider_box["width"]/2
        sy = slider_box["y"] + slider_box["height"]/2

        # 判定验证是否通过: 验证码容器消失 或 页面出现登录token
        def captcha_passed():
            try:
                gone = page.query_selector("#tCaptchaDyMainWrap") is None
            except Exception:
                gone = True
            # cookie token 判断
            tok = next((c["value"] for c in ctx.cookies() if c["name"]=="token"), None)
            has_full_tok = bool(tok and len(tok) > 8)
            return gone or has_full_tok

        passage = None
        for idx, cxx in enumerate(cands):
            dist = int(round(cxx * ratio))
            print(f"[尝试{idx+1}/{len(cands)}] 缺口候选x={cxx} 拖动={dist}px", flush=True)
            # 在滑块元素上按下并拖动(浏览器原生级, 腾讯能收到真实指针事件)
            try:
                page.mouse.move(sx, sy)
                page.wait_for_timeout(random.randint(80, 200))
                page.mouse.down()
                page.wait_for_timeout(random.randint(30, 90))
            except Exception as e:
                print("down err", repr(e)[:60], flush=True)
            track = human_track(0, dist)
            for px, yj in track:
                page.mouse.move(sx + px, sy + yj, steps=1)
                # 真实人非匀速: 停顿有时长有时短
                page.wait_for_timeout(random.choice([3,4,5,6,8,10,13,16]))
            # 释放前微停
            page.wait_for_timeout(random.randint(60, 160))
            page.mouse.up()
            page.wait_for_timeout(1800)
            if captcha_passed():
                passage = cxx
                print(f"✅ 验证通过于候选x={cxx}!", flush=True)
                break
            # 失败回弹: 拖回起点, 便于下一候选
            try:
                page.mouse.move(sx, sy)
                page.mouse.down()
                page.mouse.move(sx + random.randint(6, 20), sy, steps=2)
                page.mouse.up()
            except Exception:
                pass
            page.wait_for_timeout(700)

        page.wait_for_timeout(2500)
        cookies = ctx.cookies()
        browsertok = next((c["value"] for c in cookies if c["name"]=="token"), None)
        print("过验 token:", (browsertok[:12]+"…" if browsertok else None),
              "| passed_candidate:", passage, flush=True)
        dump = {"paperid": paperid, "gap": gap, "token": browsertok,
                "passed_candidate": passage}
        with open(os.path.join(OUT, "result.json"), "w") as f:
            json.dump(dump, f, ensure_ascii=False, indent=2)
        print("完成。产物见 output/", flush=True)
        browser.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paperid", default="14581066")
    ap.add_argument("--method", default="template")
    a = ap.parse_args()
    print("OpenCV:", cv2.__version__, "matchTemplate可用:", CV_MT_OK, flush=True)
    run(a.paperid, a.method)
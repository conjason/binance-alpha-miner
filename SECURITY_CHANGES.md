# 安全改动记录

本项目在一个采用 AGPL-3.0 的开源因子挖掘项目基础上二次开发。本文件逐条记录相对上游
**删除/改写/加固**的安全相关内容，便于审计。

## 1. 移除凭证读取代码（核心安全问题）

**文件**：`web/ai_providers.py`

**上游行为**：为给「AI 分析训练情况」这个可选功能白嫖 LLM 调用，上游代码会：

1. 扫描本机某 Electron 桌面客户端的登录会话文件（`...\auth\*.info`）；
2. 读该客户端的 `Local State`，用 **Windows DPAPI**（`crypt32.CryptUnprotectData`）解出主密钥；
3. 用该 **AES-GCM** 密钥暴力扫描其 `Local Storage/leveldb`、`Session Storage`、`Network` 目录里
   `v10/v11` 前缀的加密块，解密后匹配 `eyJ...`(JWT) / `accessToken` 捞出登录 token；
4. 拿该 token 调用对应云端 LLM 接口。

这套「读 Local State → DPAPI 解主密钥 → AES-GCM 解密凭证库」的手法与浏览器凭证窃取木马
（infostealer）一致。一个因子挖掘工具没有任何正当理由去解密另一个软件的加密凭证库。

**本 fork 处理**：`web/ai_providers.py` **完整重写**，只保留「用户显式提供 API Key」的通道：
- `deepseek`：官方 DeepSeek（`https://api.deepseek.com`），Key 由界面输入或 `DEEPSEEK_API_KEY`；
- `openai_compatible`：任意 OpenAI 兼容端点，Key 由界面输入或 `OPENAI_COMPAT_API_KEY`。

**彻底删除**：所有 DPAPI / AES-GCM / ctypes / leveldb 扫描与本地 token 自动读取逻辑。

**连带清理**：
- `requirements.txt` 删除 `cryptography`（上游注释原文即 "token decrypt on Windows"，仅此一用）；
- `web/app.py`：删除按 Key 前缀自动切到本地 token 通道的路由；
- `web/settings.py`：合法 provider 列表改为 `("deepseek", "openai_compatible")`；
- `web/static/app.js`：删除相关前端提示与别名解析；删除失效测试。

## 2. 修复导入训练包功能的两处漏洞（Pickle RCE + Zip Slip）

上游「导入训练包/检查点」的 Web 功能存在两处可被利用的问题（照抄进来后已修复）：

- **Pickle 反序列化执行代码**：`torch.load(..., weights_only=False)` 会执行 `.pt` 里嵌入的
  任意代码。**修复**：全仓 15 处 `weights_only=False` 全改 `True`，`engine.load_checkpoint`
  也显式 `weights_only=True`（自产 checkpoint 实测在安全模式下正常加载/续训）。
- **Zip Slip 路径穿越**：导入 `.zip` 时把成员写到 `项目根/成员名`，不校验 `../`。
  **修复**：重写为**只取文件名**按白名单路由到固定位置（忽略 zip 内目录结构），加品种名
  白名单、路径根内双校验、单文件 500MB 上限防 zip 炸弹。已用构造的恶意 zip 验证被拦截。

## 3. 验证

```
grep -rn "CryptUnprotect|_dpapi_decrypt|encrypted_key|Local State|leveldb|weights_only=False" --include=*.py .
# → 无匹配
```

全仓网络出口：仅 `api.deepseek.com` / `api.openai.com`（均为用户自带 Key）、可选数据源
（OKX 等）、`127.0.0.1`（本地 Web）。无第三方外传，无触碰浏览器/钱包/cookie/ssh/注册表/自启动。

## 4. 第二轮加固（外部审计后补修）

- **Web CORS 收紧**：`allow_origins=["*"]` → 只允许 `127.0.0.1/localhost` 来源，
  避免任意网页跨域读取本地无鉴权 API 的响应。
- **接口不再回传明文密钥**：`/api/config` 不再返回 AI Key（改为 `has_api_key` 布尔）；
  `/api/realtime/feishu` 不再返回 webhook_url / secret（改为 `has_webhook` / `has_secret`），
  PUT 也仅在传入非空值时更新，空值不覆盖已存密钥。
- **飞书 SSRF 面封堵**：`send_text` 增加 webhook 域名白名单（仅 `open.feishu.cn` /
  `open.larksuite.com` 等 + 必须 https + 路径以 `/open-apis/bot/` 开头），杜绝把用户填的
  URL 变成服务端任意请求（打内网/本机）。
- **最后一处 pickle 隐患**：`inspect_checkpoint.py` 的 `torch.load` 补上 `weights_only=True`。
- **训练包导入更稳**：改为**先在内存里安全校验全部成员（含 .pt 用 weights_only 校验），
  通过后才删旧检查点并落盘**（损坏/恶意包不再先删后败导致数据丢失）；新增总解压体积
  (2GB) / 成员数量 (50) / 单成员压缩比 (200x) 上限，防 zip 炸弹。
- **前端 CDN 供应链**：Chart.js 由外部 jsDelivr（无 SRI）改为**本地内置**
  `web/static/vendor/chart.umd.min.js`，运行时不再从外部 CDN 拉脚本。
- **测试卫生**：新增 `pytest.ini`（`testpaths=tests`）避免收集 `scripts/` 下有副作用的
  集成脚本；把因新增 12 个因子而失配的计数类测试改为**不变量断言**（对齐动态派生的
  特征/算子数，不再写死数值）；属性测试禁用 Hypothesis deadline（真实张量计算耗时不稳定）。

## 5. 已知遗留（诚实说明，非本 fork 引入）

- 单元/属性测试当前 **253 通过 / 2 失败**。剩余 2 个失败均为上游既有、与安全无关：
  `test_data_props`（MT5 行情 fetcher 的 Hypothesis 边界样例 `symbol='0'`，是外汇 MT5 专用
  代码，币安路径不经过它）与 `test_training_time`（依赖运行时刻的时间断言，天然 flaky）。
- 供应链未上锁：依赖仅最低版本号、无 lockfile/哈希；`requirements-optional.txt` 里 tvdatafeed
  为未固定提交的 Git 源。如需更严的可复现性，请自行加锁。

## 6. 未改动但需知晓的部分

- `execution/`、`live_trade.py` 等 MT5 实盘下单模块保留；非安全隐患（需你自己的 MT5 凭证），
  对币安用户无用，不参与训练/回测主路径。
- Web 服务默认只监听 `127.0.0.1`；请勿用 `--host 0.0.0.0` 暴露到公网。
- AI 分析功能会把训练上下文（公式/分数）发给你**自己配置**的 LLM，属于你主动选择的数据分享。

#!/usr/bin/env bash
set -euo pipefail

PORT="${HYWORLD2_CDP_PORT:-9220}"
PROFILE="${HYWORLD2_CHROME_PROFILE:-$HOME/.hyworld2-playwright-profile}"
URL="${HYWORLD2_URL:-https://3d.hunyuan.tencent.com/sceneTo3D?tab=worldplay}"
if [[ -n "${HYWORLD2_CHROME_BIN:-}" ]]; then
  CHROME="$HYWORLD2_CHROME_BIN"
  APP=""
elif [[ -x "/Applications/Google Chrome Dev.app/Contents/MacOS/Google Chrome Dev" ]]; then
  CHROME="/Applications/Google Chrome Dev.app/Contents/MacOS/Google Chrome Dev"
  APP="/Applications/Google Chrome Dev.app"
else
  CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
  APP="/Applications/Google Chrome.app"
fi

mkdir -p "$PROFILE"
ARGS=(
  --remote-debugging-address=127.0.0.1
  --remote-debugging-port="$PORT"
  --user-data-dir="$PROFILE"
  --no-first-run
  --no-default-browser-check
  --disable-blink-features=AutomationControlled
  "$URL"
)
if [[ -n "$APP" ]]; then
  open -na "$APP" --args "${ARGS[@]}"
else
  nohup "$CHROME" "${ARGS[@]}" >"/tmp/hyworld2_chrome_${PORT}.log" 2>&1 </dev/null &
fi

echo "Chrome CDP: http://127.0.0.1:$PORT"
echo "Profile: $PROFILE"
echo "首次使用请在打开的窗口中完成登录/手机号绑定。"

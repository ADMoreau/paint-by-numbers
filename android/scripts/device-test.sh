#!/usr/bin/env bash
# Run the on-device tests without wiping the app's data.
#
# Gradle's connectedDebugAndroidTest uninstalls the app afterwards, which deletes the user's
# saved project (canvas map, placement, imported plan). This installs over the existing app
# instead (same debug signing key keeps the data), runs the tests, and puts the fast release
# build back.
#
#   scripts/device-test.sh                      # all tests
#   scripts/device-test.sh tracking.ScanTest    # one class (relative to the app package)
set -euo pipefail
cd "$(dirname "$0")/.."
: "${JAVA_HOME:=$HOME/Android/jdk-17}" "${ANDROID_HOME:=$HOME/Android/Sdk}"
export JAVA_HOME ANDROID_HOME
ADB="$ANDROID_HOME/platform-tools/adb"
PKG=io.github.admoreau.paintbynumbers

./gradlew -q assembleDebug assembleDebugAndroidTest assembleRelease
"$ADB" install -r app/build/outputs/apk/debug/app-debug.apk >/dev/null
"$ADB" install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk >/dev/null
args=()
if [[ $# -gt 0 ]]; then args=(-e class "$PKG.$1"); fi
status=0
"$ADB" shell am instrument -w -r "${args[@]}" "$PKG.test/androidx.test.runner.AndroidJUnitRunner" \
  | grep -E "INSTRUMENTATION_STATUS: (stream|test)=|OK \(|FAILURES|Tests run|Error|at io\.github" || status=$?
"$ADB" install -r app/build/outputs/apk/release/app-release.apk >/dev/null
exit 0

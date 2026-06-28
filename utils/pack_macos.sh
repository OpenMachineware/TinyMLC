#!/bin/bash
# pack_macos.sh - Package TinyMLC for macOS

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

APP_NAME="TinyMLC"
APP_DIR="${PROJECT_ROOT}/build/${APP_NAME}.app"
CONTENTS_DIR="${APP_DIR}/Contents"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
VERSION="0.1.0"

echo "🔨 Building TinyMLC GUI..."
cd ${PROJECT_ROOT}/TinyGUI
mkdir -p build
cd build
cmake ..
make

echo "📦 Creating app bundle structure..."
mkdir -p "${MACOS_DIR}"
mkdir -p "${RESOURCES_DIR}"

cd "${RESOURCES_DIR}"
uv venv -p 3.13 venv
source venv/bin/activate

echo "Current dir: $(pwd)"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "RESOURCES_DIR: ${RESOURCES_DIR}"
echo "Looking for uv.lock at: ${PROJECT_ROOT}/uv.lock"
cp ${PROJECT_ROOT}/uv.lock .
uv sync --frozen
#uv pip install -e "${PROJECT_ROOT}" --python "${RESOURCES_DIR}/venv/bin/python"

echo "📝 Creating launch.sh..."
echo "RESOURCES_DIR: ${RESOURCES_DIR}"
cat > "${RESOURCES_DIR}/launch.sh" << 'EOF'
#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
export PYTHONPATH="$DIR:$DIR/TinyMLC:$DIR/utils:$DIR/TinyMLC/ANG:$DIR/TinyMLC/converter"
./venv/bin/python main.py "$@"
EOF
chmod +x "${RESOURCES_DIR}/launch.sh"


echo "📄 Copying executable..."
cp ${PROJECT_ROOT}/TinyGUI/build/TinyGUI "${MACOS_DIR}/${APP_NAME}"

echo "📄 Copying Python backend..."
cp ${PROJECT_ROOT}/main.py "${RESOURCES_DIR}/"
cp ${PROJECT_ROOT}/cli.py "${RESOURCES_DIR}/"
cp ${PROJECT_ROOT}/handlers.py "${RESOURCES_DIR}/"
cp -r ${PROJECT_ROOT}/TinyMLC "${RESOURCES_DIR}/"
cp -r ${PROJECT_ROOT}/TinyMLC/ANG "${RESOURCES_DIR}/"
cp -r ${PROJECT_ROOT}/TinyMLC/converter "${RESOURCES_DIR}/"
cp -r ${PROJECT_ROOT}/utils "${RESOURCES_DIR}/"

echo "📄 Copying Info.plist..."
cat > "${CONTENTS_DIR}/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>com.tinymlc.${APP_NAME}</string>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
</dict>
</plist>
EOF

echo "🔧 Running macdeployqt..."
macdeployqt "${APP_DIR}" -dmg

echo "✅ Done! App bundle created at: ${APP_DIR}"

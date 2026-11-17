#!/usr/bin/env bash
# Run inside the builder container. /src = repo checkout, /out = output folder.
set -euo pipefail
cd /src
if [ ! -d draw-things-community/.git ]; then
  git clone --depth 1 https://github.com/drawthingsai/draw-things-community.git
fi
mkdir -p /out
cd draw-things-community
git rev-parse HEAD > /out/SOURCE_COMMIT.txt
echo "try-import %workspace%/.bazelrc.linux" > .bazelrc
ln -sf WORKSPACE.linux WORKSPACE
LLVM_PREFIX=$(llvm-config --prefix)
./Scripts/setup_clang.sh "$LLVM_PREFIX" || true
cat > .bazelrc.local <<RC
build --config=clang
build --linkopt=-Wl,-z,nostart-stop-gc
build --host_linkopt=-Wl,-z,nostart-stop-gc
RC
bazel --output_user_root=/cache/bazel build Apps:LoRAConverter --compilation_mode=opt --keep_going --spawn_strategy=local --jobs=${JOBS:-16} 2>&1 | tee /out/build.log
cp -L bazel-bin/Apps/LoRAConverter /out/LoRAConverter
echo BUILD_OK

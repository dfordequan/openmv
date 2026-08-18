# Deploying Your Own Neural Network to the OpenMV N6 — Practical Guide

**Audience:** you have an **OpenMV N6** board, an **Ubuntu laptop**, and a trained network
(optical flow, depth estimation, a custom CNN…) that you want running on the board's NPU.

**Good news:** you do **NOT** need to rebuild or reflash the firmware. The stock firmware contains
a general model loader; you compile your model on the laptop and copy the result to the board's
SD card. No ARM toolchain, no `make`.

**Time:** ~1 hour for the first model if nothing goes wrong. Budget a day for a real one — the
traps in §8 are where the time goes.

---

## 0. What you need

| Item | Notes |
|---|---|
| OpenMV **N6** board | STM32N6, 800 MHz Cortex-M55 + Neural-ART NPU. **Must be the N6 or AE3** — other OpenMV boards have no NPU. |
| **microSD card** | Strongly recommended. On-board `/flash` is only ~4 MB total; models are 1–4 MB. |
| **USB-C data cable** | A charge-only cable is the #1 "board not detected" cause. If `lsusb` shows nothing, try another cable first. |
| Ubuntu laptop | GPU only needed for *training*, not for deployment. |
| Your trained model | PyTorch/TensorFlow. Must be exportable to ONNX. |
| Representative input data | ~200 samples for int8 calibration. **Not optional** — see §4. |

---

## 1. Install

### 1.1 The compiler (`stedgeai`) — you probably already have it
The **OpenMV IDE bundles ST Edge AI Core**. Install the IDE from openmv.io, then:

```bash
ls ~/openmvide/share/qtcreator/stedgeai/Utilities/linux/stedgeai
~/openmvide/share/qtcreator/stedgeai/Utilities/linux/stedgeai --version
# ST Edge AI Core v4.0.0-...
```
Save it for later:
```bash
export STE=~/openmvide/share/qtcreator/stedgeai/Utilities/linux/stedgeai
export NEURALART=~/developer_dq/openmv/lib/stai/scripts/neuralart.json   # from the openmv repo
```
(If you don't have the IDE, ST Edge AI Core is a separate free download from ST — needs an
ST account. The IDE route is easier.)

You also need the **`neuralart.json` profile + `stm32n6.mpool`** from the openmv firmware repo
(`lib/stai/scripts/`). Clone `https://github.com/openmv/openmv` just for these two files if needed.
⚠️ `neuralart.json` references `stm32n6.mpool` by a **relative** path — keep them in the same
directory, or edit `memory_pool` to an absolute path.

### 1.2 Python environment
Use a **dedicated conda env**. Do not install this into an env you care about — the
ONNX/TF/torch dependency graph is fragile and will happily downgrade numpy underneath you.

```bash
conda create -n n6deploy python=3.11 -y && conda activate n6deploy
pip install torch onnx onnxruntime numpy pillow pyserial
# only if you train/export with ultralytics:
pip install ultralytics
```
You do **not** need TensorFlow if you go PyTorch → ONNX (recommended).

### 1.3 Serial port permissions
```bash
groups | grep dialout || sudo usermod -aG dialout $USER   # then LOG OUT and back in
```

### 1.4 Check the board
```bash
lsusb | grep -i 37c5          # 37c5:1206 = OpenMV
ls -l /dev/ttyACM0
```
Nothing? → charge-only cable, a USB hub, or the board isn't powered. Try a direct port.

---

## 2. Know the constraints BEFORE you design the network

The Neural-ART is an **int8 convolution/GEMM engine**, not a general tensor processor.

**Accelerated:** `Conv2D`, `DepthwiseConv2D`, 1x1 conv, `TransposeConv`, `FullyConnected`,
max/avg pool, `Add`/`Mul`/`Concat`, `ReLU`/`ReLU6`, resize/upsample.
→ Encoder–decoder CNNs (U-Net for depth/flow) map well. TransposeConv and resize upsampling are fine.

**Falls back to the CPU (slow):** attention/transformers, recurrent cells (GRU/LSTM), dynamic
shapes, control flow, custom ops, **and anything not int8**.

**Memory:**
- Activations → **~1.8 MB** fast on-chip AXISRAM (4 x 448 KB banks). *This is usually the binding
  limit*, and it scales with **input resolution x channel width**.
- Weights → 16 MB external flash. Practical ceiling ~3 MB per network.

**Latency scales with input resolution, and resolution is yours to choose.** The firmware
bilinear-downscales the camera frame into the model's input size, so a 192x192 model on a VGA
sensor still only pays 192x192. **Keep input small** (96–256). For reference: a YOLOv8n at 192
runs in ~21 ms; the same net at 320 took ~290 ms.

> **Dense-output nets (depth / optical flow) — plan the output too.** A 128x128 float depth map is
> 64 KB per inference and must fit in the activation budget alongside the decoder. Prefer
> predicting at reduced resolution (e.g. 64x64) and upsampling on the host if you need more.

---

## 3. Export to ONNX (+ the NHWC fix)

Export ONNX, **at the exact resolution you trained at** (see §8, trap 1):

```python
import torch
model.eval()
dummy = torch.zeros(1, C, H, W)
torch.onnx.export(model, (dummy,), "model.onnx",
                  input_names=["input"], output_names=["output"],
                  opset_version=13, dynamo=False)
```

### 3.1 Make the input NHWC
ONNX is NCHW `(1,C,H,W)`; the board wants **NHWC `(1,H,W,C)`**. Insert a transpose at the input —
this costs nothing and saves a lot of pain:

```python
import onnx
from onnx import helper, TensorProto
m = onnx.load("model.onnx"); g = m.graph
old = g.input[0]; nm = old.name
n, c, h, w = [d.dim_value for d in old.type.tensor_type.shape.dim]
g.input.remove(old)
g.input.insert(0, helper.make_tensor_value_info("input_nhwc", TensorProto.FLOAT, [n, h, w, c]))
g.node.insert(0, helper.make_node("Transpose", ["input_nhwc"], [nm],
                                  perm=[0, 3, 1, 2], name="nhwc2nchw"))
onnx.checker.check_model(m); onnx.save(m, "model_nhwc.onnx")
```

---

## 4. Quantize to int8 (mandatory for the NPU)

**Float models run 100% on the CPU.** int8 is what puts work on the NPU.

```python
import numpy as np, glob
from PIL import Image
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                      QuantType, QuantFormat)

FILES = sorted(glob.glob("calib/*.png"))[:200]     # ~200 REAL samples from your domain

class Reader(CalibrationDataReader):
    def __init__(self): self.i = iter(FILES)
    def get_next(self):
        f = next(self.i, None)
        if f is None: return None
        a = np.asarray(Image.open(f).convert("RGB").resize((W, H)), np.float32)[None] / 255.0
        return {"input_nhwc": a}                    # NHWC, EXACTLY your training preprocessing

quantize_static("model_nhwc.onnx", "model_int8.onnx", Reader(),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                per_channel=True)
```

**Rules:**
- Calibration data must be **real data from your deployment domain**, preprocessed **identically**
  to training. Bad calibration is the #1 cause of "it works on my laptop".
- **Never let one int8 scale cover outputs with different ranges.** If your model outputs
  heterogeneous quantities in one tensor (e.g. flow in pixels AND a confidence in 0–1), a single
  scale sized for the large range **quantizes the small one to all zeros**. Either output them as
  **separate tensors**, or keep the output head in float via `nodes_to_exclude=[...]`.
  For depth/flow with one homogeneous output this is usually fine — but **check** (§7).

---

## 5. Compile for the NPU

**Always `analyze` first** — it reports memory + NPU/CPU placement and needs no board:

```bash
$STE analyze --model model_int8.onnx --type onnx --target stm32n6 \
    --st-neural-art default@$NEURALART --workspace /tmp/ws --output /tmp/out
```

Read `/tmp/out/network_analyze_report.txt`:

| epoch type | meaning |
|---|---|
| **HW** | fully on the NPU |
| **EC** (meta) | **also NPU** — HW epochs merged into a command stream (the default profile enables this, so `HW: 0` is NORMAL and does not mean "no NPU") |
| **Hybrid** | partly CPU, assisted by NPU |
| **SW** | **on the Cortex-M55 CPU — this is what makes you slow** |

**Judge utilisation by `SW` (bad) vs `EC + HW + Hybrid` (good).** Lots of SW epochs → your ops
aren't supported, or the model isn't properly int8.

Also check memory: activations must fit ~1.8 MB, weights < ~3 MB.

Then build the loadable binary:

```bash
$STE generate --model model_int8.onnx --type onnx --target stm32n6 --relocatable \
    --st-neural-art default@$NEURALART --workspace /tmp/ws --output /tmp/out
# -> /tmp/out/network_rel.bin      <- this is what you copy to the board
```
`--relocatable` is **required**: it makes position-independent code the firmware can load at
runtime. Without it you'd have to rebuild firmware.

---

## 6. Get it onto the board

**Easiest:** put the SD card in your laptop, copy `network_rel.bin` onto it as e.g. `mymodel.bin`.

**Or over USB** (no mass-storage interface on this board, so use the REPL helper):
```bash
python omv_put.py /tmp/out/network_rel.bin /sdcard/mymodel.bin
```
~120–200 KB/s, so a 3 MB model takes ~20–30 s. Close the OpenMV IDE first — it holds the port.

**Labels (classifiers/detectors only):** put `mymodel.txt`, one label per line, next to the `.bin`;
`ml.Model` picks it up automatically.

---

## 7. Run it — and VERIFY it

```python
import ml, image, time
model = ml.Model('/sdcard/mymodel.bin')
print(model.input_shape, model.output_shape)      # sanity-check the contract
img = image.Image('/sdcard/test.bmp', copy_to_fb=True)
t0 = time.ticks_ms(); out = model.predict([img]); t1 = time.ticks_ms()
print(time.ticks_diff(t1, t0), 'ms')
```

### 7.1 If your input is NOT a 1- or 3-channel image
`predict([img])` runs a preprocessing step that **only accepts 1 or 3 channels** and will raise
`ValueError: Expected channels to be 1 or 3`. This bites **optical flow** (two stacked frames = 2 or
6 channels) and any multi-channel tensor input.

**Solution: build the tensor yourself and pass it directly** (this is proven to work — an
8-channel input was fed this way):
```python
from ulab import numpy as np
buf = open('/sdcard/input.bin', 'rb').read()          # raw float32, NHWC order
x   = np.frombuffer(buf, dtype=np.float).reshape((1, H, W, C))
out = model.predict([x])                              # bypasses image preprocessing
```
Multiple inputs work too: `model.predict([x1, x2, x3])`.
For live optical flow you'd grab two frames, build the stacked array, and pass it in.

### 7.2 Verify against your laptop — do not skip this
Run the **same** input through `model_int8.onnx` with onnxruntime on the laptop and compare
numbers with the board. They should match closely.

**Test with real, asymmetric data — never zeros.** A zero input cannot reveal a layout (NHWC/NCHW)
error, because zeros look the same in any layout. Ask me how I know.

---

## 8. Troubleshooting — the traps that cost real time

**Trap 1 (worst): export resolution MUST equal training resolution.**
A model trained at 320 and exported at 192 produced max confidence 0.05 and **zero output — in
float too**, so it looked like a quantization bug for hours. It wasn't. If you want a smaller,
faster model, **retrain at that size**.

| symptom | cause | fix |
|---|---|---|
| `ValueError: Expected channels to be 1 or 3` | model input is NCHW, or has ≠1/3 channels | NHWC surgery (§3.1); or feed an ndarray (§7.1) |
| `INTERNAL ERROR: 0 vs. 11` from stedgeai | you fed it a **tflite** exported by ultralytics 8.4 (it's NCHW) | use the ONNX route |
| Output is all zeros / nothing detected | one int8 scale spanning mixed output ranges | separate outputs, or keep the head float via `nodes_to_exclude` |
| Inference works but results are garbage | fed NCHW where NHWC expected (or vice versa) | compare against host with **asymmetric** data |
| Very slow (100s of ms) | lots of **SW** epochs: model is float, or uses unsupported ops | check the analyze report; ensure real int8; simplify ops |
| `RuntimeError: Failed to load network` | see below | — |
| Board not in `lsusb` | charge-only cable / hub / no power | different cable, direct port |
| `Permission denied` on `/dev/ttyACM0` | not in `dialout` | `usermod -aG dialout`, re-login |
| Port busy | OpenMV IDE has it open | close the IDE |

### Known unresolved issue: small models may refuse to load
`RuntimeError: Failed to load network` (from `ll_aton_reloc_get_info`/`ll_aton_reloc_install`)
was hit reproducibly with **small networks (26–85 KB)**, while everything **>1.5 MB loaded fine**.
Ruled out: epoch controller on/off, int8 vs float, forcing SW epochs, forcing external RAM.
**Not root-caused.** If you hit it on a small model, the practical workaround is to deploy a
**larger single network** rather than splitting into small ones. (Suspected small-model path in the
relocatable loader; a real fix means patching firmware, which *does* need a Cortex-M55 ARM GCC ~13+.)

---

## 9. Making it faster

1. **Reduce input resolution** — the biggest lever (cost scales ~quadratically). Retrain at the
   smaller size; don't just export smaller.
2. **Eliminate SW epochs** — check the analyze report and replace unsupported ops with supported
   equivalents (e.g. ReLU instead of exotic activations; resize+conv instead of odd upsamplers).
3. **Use depthwise-separable convs** (MobileNet style) — ~8-9x fewer MACs at the same width, and
   fully NPU-accelerated.
4. **Shrink channel widths** before shrinking depth — activations scale with width.
5. Predict at low resolution and upsample on the host (very effective for depth/flow).

---

## 10. Cheat sheet

```bash
export STE=~/openmvide/share/qtcreator/stedgeai/Utilities/linux/stedgeai
export NEURALART=<path>/openmv/lib/stai/scripts/neuralart.json

# 1. export ONNX at TRAINING resolution, then NHWC surgery      (§3)
# 2. int8 quantize with ~200 real calibration samples           (§4)
$STE analyze  --model model_int8.onnx --type onnx --target stm32n6 \
      --st-neural-art default@$NEURALART --workspace /tmp/ws --output /tmp/out
$STE generate --model model_int8.onnx --type onnx --target stm32n6 --relocatable \
      --st-neural-art default@$NEURALART --workspace /tmp/ws --output /tmp/out
python omv_put.py /tmp/out/network_rel.bin /sdcard/mymodel.bin
python omv_repl.py "import ml; m=ml.Model('/sdcard/mymodel.bin'); print(m.input_shape, m.output_shape)"
```

**Golden rules**
1. Export at the resolution you trained at.
2. int8 or it runs on the CPU.
3. NHWC.
4. `analyze` before you flash; verify against the host with real data before you trust it.

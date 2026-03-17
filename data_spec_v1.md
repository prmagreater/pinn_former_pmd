# 仿真数据集规范 v1（偏折主数据 + CMM锚点 + PINN/Transformer）
日期：2026-03-17

本规范用于生成**可复现实验**的仿真数据集，满足：
1) 训练信息充足、覆盖真实自由曲面可能成分；
2) **不透题**（不保存/不提供任何“隐形直接答案”给训练）；
3) 将来可制作实物工件（铝基体 + NiP + SPDT + 轻抛光）并进行 PMD（单相机+单屏幕）验证；
4) 与当前工程数据接口兼容：**四件套 + 可选 design_surface**。

---

## 0. 坐标系与单位（冻结）

- Mirror frame **M**：镜面局部坐标系，右手系，`z` 指向镜面外法向（朝相机/屏幕一侧）。
- Camera frame **C**：相机光心坐标系。
- Screen frame **S**：屏幕平面坐标系，`u,v` 为屏幕平面物理坐标。

单位：
- `x,y,z`：米 (m)
- `p=∂z/∂x, q=∂z/∂y`：无量纲（**不保存到文件**）
- `u,v`：米 (m)
- `conf`：0~1

法向定义（用于 forward/损失，必须一致）：
- `n = (-p, -q, 1) / sqrt(1+p^2+q^2)`

---

## 1. 文件结构（每个样本一个文件夹）

最小四件套（必须）：
- `surface_gt.npy`：仅用于评估/画图（训练代码**禁止读取**）
- `deflect_obs.npy`：主观测（训练使用）
- `cmm_points.npy`：辅观测（训练使用）
- `calib.json`：网格/孔径/相机屏幕几何（训练与forward使用）

可选（推荐用于工业语境与残差定义）：
- `design_surface.npy`：设计面形 `z_design(x,y)`（允许作为先验/正则，但必须做 ablation）

示例：
```
dataset_root/
  sample_A_000001/
    surface_gt.npy
    deflect_obs.npy
    cmm_points.npy
    calib.json
    design_surface.npy   # 可选
  sample_B_000001/
  sample_C_000001/
```

---

## 2. 各文件字段定义（严禁透题）

### 2.1 surface_gt.npy（仅评估用）
- dtype: float32
- shape: (H, W)
- 内容：真实面形 `z_true(x,y)`，孔径外可为 NaN。
- 约束：训练脚本不得将其作为输入/监督标签。

### 2.2 deflect_obs.npy（主观测，必须）
- dtype: float32
- shape: (H, W, 3)
- 通道：
  - `[...,0] = u_m`：屏幕命中坐标 u（m）
  - `[...,1] = v_m`：屏幕命中坐标 v（m）
  - `[...,2] = conf`：有效性/置信度（0~1）
- 生成链路：`z_true → forward → (u,v)理想 → 观测域噪声/畸变 → conf`

**不允许**在 deflect_obs 中额外保存：真实坡度/真实法向/缺陷mask/任何 forward 中间变量。

### 2.3 cmm_points.npy（辅观测锚点）
- dtype: float32
- shape: (N, 3)
- 列：`x, y, z`（Mirror frame M，单位 m）
- 备注：允许加入小噪声（模拟 CMM 测量噪声）。

### 2.4 calib.json（几何与网格）
必须包含：
- `units.length="m"`, `units.screen_uv="m"`
- `mirror_grid`: H,W,x0,y0,dx,dy
- `aperture`: type="circle", radius_m
- `camera_intrinsics`: fx,fy,cx,cy（可先占位）
- `extrinsics`: T_WM,T_WC,T_WS（4x4；v1 允许只用平移）
- `screen`: width_m,height_m,width_px,height_px,pixel_pitch_m
- `notes`：说明

### 2.5 design_surface.npy（可选，推荐）
- dtype: float32
- shape: (H, W)
- 内容：设计面形 `z_design(x,y)`（孔径外 NaN）
- 用途：定义残差 `e = μz - z_design`；或作为形状先验。
- 要求：训练/报告必须做 ablation（有/无 design 先验）。

---

## 3. “真实世界成分”分层模型（保证不透题）

定义：
`z_true = z_design + Δz_LF + Δz_MSF + Δz_defect`

### 3.1 设计基形 z_design（可制作、可复现）
- 固定三种设计族：A（温和基准）、B（挑战高梯度）、C（多尺度基准）
- 口径：清口径 D=50mm（R=25mm）
- 约束（单相机单屏幕可测）：
  - A/C：max slope ≤ 0.18（≈10°）
  - B：max slope ≤ 0.25（≈14°）
- 设计形式：低阶 XY 多项式或低阶 Zernike（实现上选其一，固定系数并记录到 manifest）。

### 3.2 低频制造误差 Δz_LF（逼迫CMM发挥价值）
- 低阶基函数扰动（例如少量低阶 Zernike/多项式项），模拟装夹漂移/低频形变。
- 幅值建议：RMS 0.5–5 µm（可在 manifest 设范围）。

### 3.3 中频误差 Δz_MSF（SPDT+轻抛光典型背景）
- 形式：
  - 少量正弦叠加（可控） + 可选带限随机场（更真实）
- 幅值建议：RMS 0.1–1 µm；周期 1–5 mm。

### 3.4 缺陷 Δz_defect（局部几何缺陷 + 非几何异常）
几何缺陷（作用于 z_true）：
- Gaussian pit/bump：|A|=0.2–5 µm，σ=0.15–0.8 mm
- Scratch groove：深0.05–1 µm，宽30–150 µm，长5–20 mm，可控方向

非几何异常（作用于观测域，不改 z_true）：
- 局部 conf 降低（低对比度/饱和/遮挡）
- 局部 uv 噪声增大（解码不稳定）
> 该类用于解释“σ高但e不高”的区域，增强工业真实性。

---

## 4. 观测域误差模型（不透题的关键）

必须在 **(u,v,conf)观测域**引入误差，而不是给模型“答案提示”：

- `uv_gaussian_noise`：u,v 加高斯噪声（等效相位解码噪声）
- `uv_lowfreq_distortion`：u,v 加低阶畸变场（模拟系统误差/标定残差）
- `conf_masking`：
  - 孔径边缘、高斜率、缺陷附近更容易无效
  - 可加随机“坏块”模拟饱和/遮挡

CMM 噪声：z 方向小高斯噪声（0.05–0.5 µm），可选对 x,y 不加噪（v1）。

---

## 5. 训练/评估约束（防止透题）

- 训练时允许读取：`deflect_obs`, `cmm_points`, `calib`, （可选）`design_surface`
- 训练时禁止读取：`surface_gt`、任何GT slope/normal、任何缺陷mask
- 若要做缺陷检测评估，缺陷mask只能放在 **独立评估脚本**生成/读取，不写入样本文件夹（避免被误用）。

---

## 6. 必须可复现的输出（每次生成数据集都要记录）

- `dataset_manifest_v1.json`：所有参数范围与随机种子
- 每个样本 `calib.json` 中记录：surface_family、噪声配置摘要、valid_ratio 等元信息（但不含GT中间量）


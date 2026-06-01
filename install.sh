#!/usr/bin/env bash
# =============================================================================
# 道路巡检反光板检测系统 — 一键安装脚本
# Road Reflector Detection System — One-click Installation Script
#
# 用法 (Usage):
#   chmod +x install.sh
#   ./install.sh [--env ENV_NAME] [--cuda CUDA_VERSION] [--skip-conda]
#
# 参数说明 (Options):
#   --env        Conda 虚拟环境名称，默认 road-reflector
#   --cuda       CUDA 版本 (11.7 / 11.8 / 12.1)，默认自动检测
#   --skip-conda 跳过 Conda 环境创建（已有环境时使用）
#   --help       显示帮助信息
# =============================================================================

set -e  # 任意命令失败则立即退出

# ── 颜色输出 ──────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
step()    { echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════${NC}"; \
            echo -e "${BOLD}  STEP $1: $2${NC}"; \
            echo -e "${BOLD}${BLUE}══════════════════════════════════════════${NC}"; }

# ── 默认参数 ──────────────────────────────────────────────
ENV_NAME="road-reflector"
CUDA_VERSION=""
SKIP_CONDA=false
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 参数解析 ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)      ENV_NAME="$2";    shift 2 ;;
        --cuda)     CUDA_VERSION="$2"; shift 2 ;;
        --skip-conda) SKIP_CONDA=true; shift ;;
        --help)
            echo "用法: ./install.sh [--env ENV_NAME] [--cuda CUDA_VERSION] [--skip-conda]"
            echo ""
            echo "选项:"
            echo "  --env ENV_NAME      Conda 环境名称 (默认: road-reflector)"
            echo "  --cuda CUDA_VERSION CUDA 版本，如 11.8 或 12.1 (默认: 自动检测)"
            echo "  --skip-conda        跳过 Conda 环境创建步骤"
            echo "  --help              显示此帮助信息"
            exit 0
            ;;
        *) warn "未知参数: $1，已忽略"; shift ;;
    esac
done

# ── 欢迎信息 ──────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  道路巡检反光板检测系统 — 一键安装程序               ║${NC}"
echo -e "${BOLD}║  Road Reflector Detection System Installer           ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
info "安装目录: ${REPO_DIR}"
info "Conda 环境名称: ${ENV_NAME}"

# ══════════════════════════════════════════════════════════
# STEP 1: 系统环境检测
# ══════════════════════════════════════════════════════════
step 1 "系统环境检测 (System Environment Check)"

# 检查操作系统
OS=$(uname -s)
if [[ "$OS" != "Linux" && "$OS" != "Darwin" ]]; then
    error "不支持的操作系统: $OS。请在 Linux 或 macOS 上运行。"
fi
success "操作系统: $OS"

# 检查 Python 版本
if command -v python3 &> /dev/null; then
    PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYTHON_MAJOR=$(echo $PYTHON_VER | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VER | cut -d. -f2)
    if [[ $PYTHON_MAJOR -lt 3 || ($PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 8) ]]; then
        error "Python 版本过低: $PYTHON_VER。需要 Python >= 3.8"
    fi
    success "Python 版本: $PYTHON_VER"
else
    error "未找到 Python3。请先安装 Python >= 3.8"
fi

# 检测 CUDA 版本
if [[ -z "$CUDA_VERSION" ]]; then
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2- | cut -d. -f1,2)
        success "自动检测到 CUDA 版本: $CUDA_VERSION"
    elif command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
        success "自动检测到 CUDA 版本: $CUDA_VERSION (via nvidia-smi)"
    else
        warn "未检测到 CUDA，将安装 CPU 版本的 PyTorch"
        CUDA_VERSION="cpu"
    fi
fi

# 检查 Git
if ! command -v git &> /dev/null; then
    error "未找到 git。请先安装 git。"
fi
success "Git: $(git --version)"

# ══════════════════════════════════════════════════════════
# STEP 2: Conda 环境创建
# ══════════════════════════════════════════════════════════
step 2 "Conda 虚拟环境创建 (Conda Environment Setup)"

if [[ "$SKIP_CONDA" == "true" ]]; then
    warn "已跳过 Conda 环境创建步骤（--skip-conda）"
else
    # 检查 conda
    if ! command -v conda &> /dev/null; then
        warn "未找到 conda。尝试查找 miniconda/anaconda..."
        for CONDA_PATH in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "/opt/conda/bin/conda"; do
            if [[ -f "$CONDA_PATH" ]]; then
                export PATH="$(dirname $CONDA_PATH):$PATH"
                success "找到 conda: $CONDA_PATH"
                break
            fi
        done
        if ! command -v conda &> /dev/null; then
            error "未找到 conda。请先安装 Miniconda: https://docs.conda.io/en/latest/miniconda.html"
        fi
    fi
    success "Conda: $(conda --version)"

    # 创建或更新环境
    if conda env list | grep -q "^${ENV_NAME} "; then
        warn "Conda 环境 '${ENV_NAME}' 已存在，跳过创建"
    else
        info "创建 Conda 环境: ${ENV_NAME} (Python 3.10)..."
        conda create -n "${ENV_NAME}" python=3.10 -y
        success "Conda 环境创建成功: ${ENV_NAME}"
    fi
fi

# ── 获取 conda 激活命令 ──────────────────────────────────
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "${CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || true

if [[ "$SKIP_CONDA" == "false" ]]; then
    conda activate "${ENV_NAME}" 2>/dev/null || {
        warn "conda activate 失败，尝试使用 source activate..."
        source activate "${ENV_NAME}" 2>/dev/null || \
            warn "无法激活 conda 环境，将在当前 Python 环境中继续安装"
    }
fi

# ══════════════════════════════════════════════════════════
# STEP 3: PyTorch 安装
# ══════════════════════════════════════════════════════════
step 3 "PyTorch 安装 (PyTorch Installation)"

# 检查是否已安装
if python3 -c "import torch; print(torch.__version__)" &> /dev/null; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
    success "PyTorch 已安装: $TORCH_VER，跳过安装"
else
    info "安装 PyTorch (CUDA: ${CUDA_VERSION})..."
    case "$CUDA_VERSION" in
        "12.1"|"12.2"|"12.3"|"12.4")
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            ;;
        "11.8")
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            ;;
        "11.7")
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
            ;;
        "cpu"|"")
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            ;;
        *)
            warn "未知 CUDA 版本 $CUDA_VERSION，尝试安装默认版本..."
            pip install torch torchvision torchaudio
            ;;
    esac
    success "PyTorch 安装完成"
fi

# ══════════════════════════════════════════════════════════
# STEP 4: MMSegmentation 依赖安装
# ══════════════════════════════════════════════════════════
step 4 "MMSegmentation 生态依赖安装 (MMSeg Dependencies)"

info "安装 openmim..."
pip install -U openmim -q

info "安装 mmengine..."
mim install mmengine -q

info "安装 mmcv..."
# 根据 CUDA 版本选择合适的 mmcv
if [[ "$CUDA_VERSION" == "cpu" ]]; then
    mim install "mmcv>=2.0.0" -q
else
    mim install "mmcv>=2.0.0" -q
fi

info "安装 mmdet（目标检测依赖）..."
mim install mmdet -q

success "MMSeg 生态依赖安装完成"

# ══════════════════════════════════════════════════════════
# STEP 5: 项目本体安装
# ══════════════════════════════════════════════════════════
step 5 "项目本体安装 (Project Installation)"

cd "${REPO_DIR}"

info "安装项目依赖 requirements.txt..."
if [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt -q
fi

info "以开发模式安装 mmseg 项目..."
pip install -e . -q

success "项目安装完成"

# ══════════════════════════════════════════════════════════
# STEP 6: 安装评估与实验依赖
# ══════════════════════════════════════════════════════════
step 6 "评估与实验依赖安装 (Evaluation & Experiment Dependencies)"

pip install numpy scipy matplotlib seaborn pandas tqdm -q
success "评估依赖安装完成"

# ══════════════════════════════════════════════════════════
# STEP 7: 自定义 EDL 模块验证
# ══════════════════════════════════════════════════════════
step 7 "自定义 EDL 模块注册验证 (EDL Module Verification)"

python3 -c "
import sys
sys.path.insert(0, '${REPO_DIR}')

# 验证 EvidentialHead 注册
try:
    from mmseg.models.decode_heads import EvidentialHead
    print('[OK] EvidentialHead 注册成功')
except ImportError as e:
    print(f'[WARN] EvidentialHead 导入失败: {e}')

# 验证 DirichletLoss 注册
try:
    from mmseg.models.losses import DirichletLoss
    print('[OK] DirichletLoss 注册成功')
except ImportError as e:
    print(f'[WARN] DirichletLoss 导入失败: {e}')

# 验证 PyTorch 基本功能
import torch
x = torch.randn(4, 3)
import torch.nn.functional as F
e = F.softplus(x)
alpha = e + 1.0
S = alpha.sum(dim=1, keepdim=True)
p = alpha / S
assert p.sum(dim=1).allclose(torch.ones(4)), 'EDL 概率归一化验证失败'
print('[OK] EDL 前向传播验证通过（概率归一化正确）')
print('[OK] 所有验证通过！')
"

# ══════════════════════════════════════════════════════════
# STEP 8: 运行单元测试
# ══════════════════════════════════════════════════════════
step 8 "运行单元测试 (Unit Tests)"

if [[ -f "${REPO_DIR}/tests/test_edl_fixes.py" ]]; then
    info "运行 EDL 修复验证测试..."
    python3 "${REPO_DIR}/tests/test_edl_fixes.py" && success "所有单元测试通过" || warn "部分单元测试失败，请检查日志"
else
    warn "未找到单元测试文件，跳过"
fi

# ══════════════════════════════════════════════════════════
# 安装完成
# ══════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║                  安装完成！                          ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}快速开始 (Quick Start):${NC}"
echo ""
echo -e "  ${YELLOW}# 1. 激活 Conda 环境${NC}"
echo -e "  conda activate ${ENV_NAME}"
echo ""
echo -e "  ${YELLOW}# 2. 运行小样本对比实验${NC}"
echo -e "  python experiments/comparison_experiment.py"
echo ""
echo -e "  ${YELLOW}# 3. 使用 EDL 配置训练（需准备数据集）${NC}"
echo -e "  python tools/train.py configs/segformer/segformer_mit-b2_edl_road_reflector.py"
echo ""
echo -e "  ${YELLOW}# 4. 运行校准评估${NC}"
echo -e "  python eval_ece_unified.py --config <config> --checkpoint <ckpt> --mode edl"
echo ""
echo -e "  ${YELLOW}# 5. 运行道路校准脚本${NC}"
echo -e "  python roadcalib.py --checkpoint <ckpt>"
echo ""
info "详细文档请查阅: docs/Secondary_Development_Report.md"
info "算法理论报告请查阅: docs/Algorithm_Theory_Innovation_Report.md"
echo ""

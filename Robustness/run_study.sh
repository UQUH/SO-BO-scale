#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMMON_DIR="${SCRIPT_DIR}/common"
SEED_FILE="${COMMON_DIR}/seeds.txt"

example_arg="all"
max_seeds=""
skip_existing=0
rebuild_truth=0

usage() {
    cat <<EOF
Usage: bash Robustness/run_study.sh [options]

Options:
  --example Ex1|Ex2|Ex3|Ex3_sigma03|all   Example to run (default: all)
  --max-seeds N               Use only the first N seeds from seeds.txt
  --skip-existing             Skip robustness outputs that already exist
  --rebuild-truth             Rebuild truth_data.mat before running
  -h, --help                  Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --example)
            example_arg="$2"
            shift 2
            ;;
        --max-seeds)
            max_seeds="$2"
            shift 2
            ;;
        --skip-existing)
            skip_existing=1
            shift
            ;;
        --rebuild-truth)
            rebuild_truth=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "${SEED_FILE}" ]]; then
    echo "Missing seed file: ${SEED_FILE}" >&2
    exit 1
fi

mapfile -t seeds < "${SEED_FILE}"
if [[ -n "${max_seeds}" ]]; then
    seeds=("${seeds[@]:0:${max_seeds}}")
fi

if [[ "${example_arg}" == "all" ]]; then
    examples=("Ex1" "Ex2" "Ex3" "Ex3_sigma03")
else
    examples=("${example_arg}")
fi

# Returns the lowercase key used in native script/file names.
# Ex3_sigma03 scripts are still named bo_ex3_*.py / bo_ex3_*.mat.
get_script_key() {
    case "$1" in
        Ex3_sigma03) echo "ex3" ;;
        *) echo "${1}" | tr '[:upper:]' '[:lower:]' ;;
    esac
}

run_truth() {
    local example="$1"
    local truth_path="${SCRIPT_DIR}/${example}/truth_data.mat"
    if [[ ${rebuild_truth} -eq 1 || ! -f "${truth_path}" ]]; then
        python "${COMMON_DIR}/build_truth_data.py" --example "${example}"
    else
        echo "Skipping existing truth data: ${truth_path}"
    fi
}

run_1d() {
    local example="$1"
    local out_path="${SCRIPT_DIR}/${example}/1D/Results/deterministic.mat"
    if [[ ${skip_existing} -eq 1 && -f "${out_path}" ]]; then
        echo "Skipping existing 1D result: ${out_path}"
        return
    fi
    python "${COMMON_DIR}/run_1d_reference.py" --example "${example}"
}

run_bo() {
    local example="$1"
    local seed="$2"
    local seed_tag
    seed_tag="$(printf '%03d' "${seed}")"
    local out_path="${SCRIPT_DIR}/${example}/BO/Results/seed_${seed_tag}.mat"
    if [[ ${skip_existing} -eq 1 && -f "${out_path}" ]]; then
        echo "Skipping existing BO result: ${out_path}"
        return
    fi
    mkdir -p "$(dirname "${out_path}")"
    matlab -batch "addpath('${COMMON_DIR}'); run_bo_seeded('${example}', ${seed}, '${out_path}');"
}

run_gp() {
    local example="$1"
    local seed="$2"
    local seed_tag
    seed_tag="$(printf '%03d' "${seed}")"
    local script_key
    script_key="$(get_script_key "${example}")"
    local out_path="${SCRIPT_DIR}/${example}/GP/Results/seed_${seed_tag}.mat"
    local raw_dir="${SCRIPT_DIR}/${example}/GP/Results/raw/seed_${seed_tag}"

    if [[ ${skip_existing} -eq 1 && -f "${out_path}" ]]; then
        echo "Skipping existing GP result: ${out_path}"
        return
    fi

    mkdir -p "${raw_dir}"
    PYTHONUNBUFFERED=1 \
    ROBUSTNESS_SEED="${seed}" \
    ROBUSTNESS_RESULTS_DIR="${raw_dir}" \
    python "${REPO_ROOT}/${example}/GP/bo_${script_key}_hetero.py"

    python "${COMMON_DIR}/run_gp_seeded.py" --example "${example}" --seed "${seed}"
}

run_sota() {
    local example="$1"
    local seed="$2"
    local seed_tag
    seed_tag="$(printf '%03d' "${seed}")"
    local script_key
    script_key="$(get_script_key "${example}")"
    local out_path="${SCRIPT_DIR}/${example}/SOTA_BO/Results/seed_${seed_tag}.mat"
    local raw_dir="${SCRIPT_DIR}/${example}/SOTA_BO/Results/raw/seed_${seed_tag}"

    if [[ ${skip_existing} -eq 1 && -f "${out_path}" ]]; then
        echo "Skipping existing SOTA result: ${out_path}"
        return
    fi

    mkdir -p "${raw_dir}"
    PYTHONUNBUFFERED=1 \
    ROBUSTNESS_SEED="${seed}" \
    ROBUSTNESS_RESULTS_DIR="${raw_dir}" \
    SOTA_ACQUISITION_FUNCTION="qlognei" \
    SOTA_N_MC_EVAL="10" \
    python "${REPO_ROOT}/${example}/SOTA_BO/bo_${script_key}_sota.py"

    python "${COMMON_DIR}/run_sota_qlognei_seeded.py" --example "${example}" --seed "${seed}"
}

run_sota_qnei() {
    local example="$1"
    local seed="$2"
    local seed_tag
    seed_tag="$(printf '%03d' "${seed}")"
    local script_key
    script_key="$(get_script_key "${example}")"
    local out_path="${SCRIPT_DIR}/${example}/SOTA_BO_QNEI/Results/seed_${seed_tag}.mat"
    local raw_dir="${SCRIPT_DIR}/${example}/SOTA_BO_QNEI/Results/raw/seed_${seed_tag}"

    if [[ ${skip_existing} -eq 1 && -f "${out_path}" ]]; then
        echo "Skipping existing SOTA qNEI result: ${out_path}"
        return
    fi

    mkdir -p "${raw_dir}"
    PYTHONUNBUFFERED=1 \
    ROBUSTNESS_SEED="${seed}" \
    ROBUSTNESS_RESULTS_DIR="${raw_dir}" \
    SOTA_ACQUISITION_FUNCTION="qnei" \
    SOTA_N_MC_EVAL="10" \
    python "${REPO_ROOT}/${example}/SOTA_BO/bo_${script_key}_sota.py"

    python "${COMMON_DIR}/run_sota_qnei_mc10_seeded.py" --example "${example}" --seed "${seed}"
}

aggregate_example() {
    local example="$1"
    python "${COMMON_DIR}/aggregate_example.py" --example "${example}"
}

plot_seed_diagnostics() {
    local example="$1"
    shift
    local all_seeds=("$@")
    local seed_vec
    seed_vec="$(printf '%s,' "${all_seeds[@]}" | sed 's/,$//')"
    matlab -batch "addpath('${COMMON_DIR}'); for s = [${seed_vec}]; plot_robustness_method_seed('${example}', 'BO', s); plot_robustness_method_seed('${example}', 'GP', s); plot_robustness_method_seed('${example}', 'SOTA_BO', s); plot_robustness_method_seed('${example}', 'SOTA_BO_QNEI', s); end"
}

for example in "${examples[@]}"; do
    echo
    echo "=== Robustness Study: ${example} ==="
    run_truth "${example}"
    run_1d "${example}"

    for seed in "${seeds[@]}"; do
        echo
        echo "[${example}] Seed ${seed}"
        run_bo "${example}" "${seed}"
        run_gp "${example}" "${seed}"
        run_sota "${example}" "${seed}"
        run_sota_qnei "${example}" "${seed}"
    done

    aggregate_example "${example}"
    plot_seed_diagnostics "${example}" "${seeds[@]}"
done

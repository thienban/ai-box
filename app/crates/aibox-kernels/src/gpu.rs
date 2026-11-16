//! Backend GPU CUDA chargé dynamiquement (`libaibox_gpu.so`) avec cache VRAM.

use crate::tensor::f32_to_f16;
use crate::{validate_matmul_inputs, BackendError, TernaryHadamardBackend};

#[cfg(target_os = "linux")]
use std::collections::HashMap;
#[cfg(target_os = "linux")]
use std::ffi::{c_char, c_int, c_void, CString};
#[cfg(target_os = "linux")]
use std::sync::Mutex;

#[cfg(target_os = "linux")]
const RTLD_NOW: c_int = 2;

#[cfg(target_os = "linux")]
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

#[cfg(target_os = "linux")]
type FnIsAvailable = unsafe extern "C" fn() -> c_int;
#[cfg(target_os = "linux")]
type FnAlloc = unsafe extern "C" fn(usize) -> *mut c_void;
#[cfg(target_os = "linux")]
type FnAllocCopy = unsafe extern "C" fn(*const c_void, usize) -> *mut c_void;
#[cfg(target_os = "linux")]
type FnFree = unsafe extern "C" fn(*mut c_void);
#[cfg(target_os = "linux")]
type FnMemcpyH2D = unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> c_int;
#[cfg(target_os = "linux")]
type FnMemcpyD2H = unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> c_int;
#[cfg(target_os = "linux")]
type FnArgmaxF32 = unsafe extern "C" fn(*const f32, c_int, *mut u32) -> c_int;
#[cfg(target_os = "linux")]
type FnFwht128F16 = unsafe extern "C" fn(*mut u16, c_int, c_int) -> c_int;
#[cfg(target_os = "linux")]
type FnMatmulTernaryF16Dev =
    unsafe extern "C" fn(*const u16, *const u8, *const u16, *mut u16, c_int, c_int, c_int, c_int) -> c_int;
#[cfg(target_os = "linux")]
type FnMatmulTernaryF16ToF32Dev =
    unsafe extern "C" fn(*const u16, *const u8, *const u16, *mut f32, c_int, c_int, c_int, c_int) -> c_int;
#[cfg(target_os = "linux")]
type FnRmsNormF16 = unsafe extern "C" fn(*const u16, *const f32, *mut u16, c_int, c_int, f32) -> c_int;
#[cfg(target_os = "linux")]
type FnRmsNormHeadsRopeF16 =
    unsafe extern "C" fn(*mut u16, *const f32, c_int, c_int, c_int, c_int, c_int, f32, f32) -> c_int;
#[cfg(target_os = "linux")]
type FnKvAppendGqaF16 = unsafe extern "C" fn(
    *const u16,
    *const u16,
    *const u16,
    *mut u16,
    *mut u16,
    *mut u16,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
) -> c_int;
#[cfg(target_os = "linux")]
type FnSwigluInplaceF16 = unsafe extern "C" fn(*mut u16, *const u16, c_int) -> c_int;
#[cfg(target_os = "linux")]
type FnLmHeadF16ActF16 = unsafe extern "C" fn(*const u16, *const u16, *mut f32, c_int, c_int) -> c_int;
#[cfg(target_os = "linux")]
type FnLmHeadF32ActF16 = unsafe extern "C" fn(*const u16, *const f32, *mut f32, c_int, c_int) -> c_int;
#[cfg(target_os = "linux")]
type FnLmHeadQuantActF16 =
    unsafe extern "C" fn(*const u16, *const u8, *const u16, *mut f32, c_int, c_int, c_int) -> c_int;
#[cfg(target_os = "linux")]
type FnMatmulTernarySwigluF16Dev = unsafe extern "C" fn(
    *const u16,
    *const u8,
    *const u16,
    *const u8,
    *const u16,
    *mut u16,
    c_int,
    c_int,
    c_int,
) -> c_int;
#[cfg(target_os = "linux")]
type FnSampleCandidatesF32 = unsafe extern "C" fn(
    *const f32,
    c_int,
    f32,
    *mut u32,
    *mut f32,
    c_int,
    *mut c_int,
) -> c_int;
#[cfg(target_os = "linux")]
type FnGraphCapture = unsafe extern "C" fn() -> c_int;
#[cfg(target_os = "linux")]
type FnGraphLaunch = unsafe extern "C" fn(*mut u16, *const u16, c_int, c_int) -> c_int;
#[cfg(target_os = "linux")]
type FnMatmulTernary = unsafe extern "C" fn(
    *const f32,
    *const u8,
    *const u16,
    *mut f32,
    c_int,
    c_int,
    c_int,
    c_int,
) -> c_int;
#[cfg(target_os = "linux")]
type FnDeinterleaveQGateF16 =
    unsafe extern "C" fn(*const u16, *mut u16, *mut u16, c_int, c_int, c_int) -> c_int;
#[cfg(target_os = "linux")]
type FnMulSigmoidInplaceF16 = unsafe extern "C" fn(*mut u16, *const u16, c_int) -> c_int;
#[cfg(target_os = "linux")]
type FnGdnConvSiluF16 =
    unsafe extern "C" fn(*const u16, *const f32, *mut f32, *mut u16, c_int, c_int) -> c_int;
#[cfg(target_os = "linux")]
type FnGdnStepF16 = unsafe extern "C" fn(
    *const u16,
    *const u16,
    *const u16,
    *const f32,
    *const f32,
    *const f32,
    *const u16,
    *mut u16,
    *mut u16,
    c_int,
    c_int,
    c_int,
    f32,
) -> c_int;
#[cfg(target_os = "linux")]
type FnMatmulDenseF16Dev =
    unsafe extern "C" fn(*const u16, *const u16, *mut u16, c_int, c_int, c_int) -> c_int;
#[cfg(target_os = "linux")]
type FnMatmulQ2AffineF16Dev = unsafe extern "C" fn(
    *const u16,
    *const u8,
    *const u16,
    *const u16,
    *mut u16,
    c_int,
    c_int,
    c_int,
) -> c_int;
#[cfg(target_os = "linux")]
type FnAllocPinned = unsafe extern "C" fn(usize) -> *mut c_void;
#[cfg(target_os = "linux")]
type FnFreePinned = unsafe extern "C" fn(*mut c_void);
#[cfg(target_os = "linux")]
type FnMemcpyH2DAsync = unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> c_int;
#[cfg(target_os = "linux")]
type FnGegluInplaceF16 = unsafe extern "C" fn(*mut u16, *const u16, c_int) -> c_int;
#[cfg(target_os = "linux")]
type FnWeightedAccF16ToF32 = unsafe extern "C" fn(*mut f32, *const u16, f32, c_int, c_int) -> c_int;
#[cfg(target_os = "linux")]
type FnVramFreeBytes = unsafe extern "C" fn() -> usize;
#[cfg(target_os = "linux")]
type FnMatmulQ2AffineGegluF16Dev = unsafe extern "C" fn(
    *const u16,
    *const u8,
    *const u16,
    *const u16,
    *const u8,
    *const u16,
    *const u16,
    *mut u16,
    c_int,
    c_int,
) -> c_int;
#[cfg(target_os = "linux")]
type FnGemma4NormRopeF16 = unsafe extern "C" fn(
    *mut u16,
    *const f32,
    *const f32,
    c_int,
    c_int,
    c_int,
    c_int,
    f32,
    f32,
) -> c_int;
#[cfg(target_os = "linux")]
type FnGemma4AttentionF16 = unsafe extern "C" fn(
    *const u16,
    *const u16,
    *const u16,
    *mut u16,
    *mut u16,
    *mut u16,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
) -> c_int;
#[repr(C)]
pub struct Gemma4MoeBatchArgs {
    pub wg: [*const u8; 8],
    pub sg: [*const u16; 8],
    pub zg: [*const u16; 8],
    pub wu: [*const u8; 8],
    pub su: [*const u16; 8],
    pub zu: [*const u16; 8],
    pub wd: [*const u8; 8],
    pub sd: [*const u16; 8],
    pub zd: [*const u16; 8],
    pub weights: [f32; 8],
    pub k_count: i32,
}

impl Default for Gemma4MoeBatchArgs {
    fn default() -> Self {
        Self {
            wg: [std::ptr::null(); 8],
            sg: [std::ptr::null(); 8],
            zg: [std::ptr::null(); 8],
            wu: [std::ptr::null(); 8],
            su: [std::ptr::null(); 8],
            zu: [std::ptr::null(); 8],
            wd: [std::ptr::null(); 8],
            sd: [std::ptr::null(); 8],
            zd: [std::ptr::null(); 8],
            weights: [0.0; 8],
            k_count: 0,
        }
    }
}

#[cfg(target_os = "linux")]
type FnCopyD2DBytes = unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> c_int;
#[cfg(target_os = "linux")]
type FnGemma4MoeQ2Batch = unsafe extern "C" fn(
    *const u16,
    *const Gemma4MoeBatchArgs,
    *mut u16,
    *mut f32,
    c_int,
    c_int,
) -> c_int;

#[cfg(target_os = "linux")]
struct GpuWeightPair {
    dev_w: *mut c_void,
    dev_scales: *mut c_void,
}

#[cfg(target_os = "linux")]
unsafe impl Send for GpuWeightPair {}
#[cfg(target_os = "linux")]
unsafe impl Sync for GpuWeightPair {}

#[cfg(target_os = "linux")]
struct RawPtr(*mut c_void);
#[cfg(target_os = "linux")]
unsafe impl Send for RawPtr {}
#[cfg(target_os = "linux")]
unsafe impl Sync for RawPtr {}

#[cfg(target_os = "linux")]
struct CudaSymbols {
    handle: *mut c_void,
    alloc: FnAlloc,
    alloc_copy: FnAllocCopy,
    free: FnFree,
    memcpy_h2d: FnMemcpyH2D,
    memcpy_d2h: FnMemcpyD2H,
    argmax_f32: FnArgmaxF32,
    fwht128_f16: FnFwht128F16,
    matmul_ternary_f16_dev: FnMatmulTernaryF16Dev,
    matmul_ternary_f16_add_dev: FnMatmulTernaryF16Dev,
    matmul_ternary_f16_to_f32_dev: FnMatmulTernaryF16ToF32Dev,
    rms_norm_f16: FnRmsNormF16,
    rms_norm_fwht128_f16: FnRmsNormF16,
    rms_norm_heads_rope_f16: FnRmsNormHeadsRopeF16,
    kv_append_gqa_f16: FnKvAppendGqaF16,
    swiglu_inplace_f16: FnSwigluInplaceF16,
    swiglu_fwht128_inplace_f16: FnSwigluInplaceF16,
    lm_head_f16_act_f16: FnLmHeadF16ActF16,
    lm_head_f32_act_f16: FnLmHeadF32ActF16,
    lm_head_quant_act_f16: FnLmHeadQuantActF16,
    matmul_ternary_swiglu_f16_dev: FnMatmulTernarySwigluF16Dev,
    sample_candidates_f32: FnSampleCandidatesF32,
    graph_begin_capture: FnGraphCapture,
    graph_end_capture: FnGraphCapture,
    graph_launch: FnGraphLaunch,
    matmul_ternary: FnMatmulTernary,
    deinterleave_q_gate_f16: FnDeinterleaveQGateF16,
    mul_sigmoid_inplace_f16: FnMulSigmoidInplaceF16,
    gdn_conv_silu_f16: FnGdnConvSiluF16,
    gdn_step_f16: FnGdnStepF16,
    matmul_dense_f16_dev: FnMatmulDenseF16Dev,
    matmul_q2_affine_f16_dev: FnMatmulQ2AffineF16Dev,
    matmul_q4_affine_f16_dev: FnMatmulQ2AffineF16Dev,
    alloc_pinned: FnAllocPinned,
    free_pinned: FnFreePinned,
    memcpy_h2d_async: FnMemcpyH2DAsync,
    geglu_inplace_f16: FnGegluInplaceF16,
    weighted_acc_f16_to_f32: FnWeightedAccF16ToF32,
    vram_free_bytes: FnVramFreeBytes,
    matmul_q2_affine_geglu_f16_dev: FnMatmulQ2AffineGegluF16Dev,
    gemma4_norm_rope_f16: FnGemma4NormRopeF16,
    gemma4_attention_f16: FnGemma4AttentionF16,
    copy_d2d_bytes: FnCopyD2DBytes,
    gemma4_moe_q2_batch: FnGemma4MoeQ2Batch,
}

#[cfg(target_os = "linux")]
unsafe impl Send for CudaSymbols {}
#[cfg(target_os = "linux")]
unsafe impl Sync for CudaSymbols {}

#[cfg(target_os = "linux")]
type WeightCacheKey = (usize, usize, usize, usize);

/// Tampon typé alloué en VRAM avec libération automatique (RAII).
pub struct GpuBuffer<'a, T = f32> {
    #[cfg(target_os = "linux")]
    gpu: &'a GpuBackend,
    #[cfg(target_os = "linux")]
    ptr: *mut T,
    #[cfg(not(target_os = "linux"))]
    _phantom: std::marker::PhantomData<&'a T>,
    len: usize,
}

#[cfg(target_os = "linux")]
impl<T> Drop for GpuBuffer<'_, T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { (self.gpu.symbols.free)(self.ptr as *mut c_void) };
        }
    }
}

impl<'a, T> GpuBuffer<'a, T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(target_os = "linux")]
impl<'a, T> GpuBuffer<'a, T> {
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

#[cfg(target_os = "linux")]
impl<'a, T> GpuBuffer<'a, T> {
    pub fn download(&self, out: &mut [T]) -> Result<(), BackendError> {
        if out.len() > self.len {
            return Err(BackendError::BufferLengthMismatch(format!(
                "download {} > GpuBuffer len {}",
                out.len(),
                self.len
            )));
        }
        let status = unsafe {
            (self.gpu.symbols.memcpy_d2h)(
                out.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                std::mem::size_of_val(out),
            )
        };
        check_cuda_status("cudaMemcpy D2H", status)
    }
}

#[cfg(target_os = "linux")]
impl<'a> GpuBuffer<'a, f32> {
    pub fn upload_f32(&mut self, data: &[f32]) -> Result<(), BackendError> {
        if data.len() > self.len {
            return Err(BackendError::BufferLengthMismatch(format!(
                "upload {} > GpuBuffer len {}",
                data.len(),
                self.len
            )));
        }
        let status = unsafe {
            (self.gpu.symbols.memcpy_h2d)(
                self.ptr as *mut c_void,
                data.as_ptr() as *const c_void,
                std::mem::size_of_val(data),
            )
        };
        check_cuda_status("cudaMemcpy H2D", status)
    }
}

#[cfg(target_os = "linux")]
impl<'a> GpuBuffer<'a, u16> {
    pub fn upload_f32_as_f16(&mut self, data: &[f32]) -> Result<(), BackendError> {
        if data.len() > self.len {
            return Err(BackendError::BufferLengthMismatch(format!(
                "upload {} > GpuBuffer len {}",
                data.len(),
                self.len
            )));
        }
        let halfs: Vec<u16> = data.iter().map(|&v| f32_to_f16(v)).collect();
        let status = unsafe {
            (self.gpu.symbols.memcpy_h2d)(
                self.ptr as *mut c_void,
                halfs.as_ptr() as *const c_void,
                std::mem::size_of_val(halfs.as_slice()),
            )
        };
        check_cuda_status("cudaMemcpy H2D (f16)", status)
    }

    pub fn copy_row_to_front(&mut self, row_idx: usize, row_len: usize) -> Result<(), BackendError> {
        if row_idx == 0 {
            return Ok(());
        }
        let offset = row_idx * row_len;
        if offset + row_len > self.len {
            return Err(BackendError::BufferLengthMismatch(format!(
                "copy_row_to_front {} > GpuBuffer len {}",
                offset + row_len,
                self.len
            )));
        }
        let mut host = vec![0u16; row_len];
        let bytes = row_len * std::mem::size_of::<u16>();
        let d2h = unsafe {
            (self.gpu.symbols.memcpy_d2h)(
                host.as_mut_ptr() as *mut c_void,
                self.ptr.add(offset) as *const c_void,
                bytes,
            )
        };
        check_cuda_status("cudaMemcpy D2H (row)", d2h)?;
        let h2d = unsafe {
            (self.gpu.symbols.memcpy_h2d)(
                self.ptr as *mut c_void,
                host.as_ptr() as *const c_void,
                bytes,
            )
        };
        check_cuda_status("cudaMemcpy H2D (row)", h2d)
    }

    pub fn download_as_f32(&self, out: &mut [f32]) -> Result<(), BackendError> {
        if out.len() > self.len {
            return Err(BackendError::BufferLengthMismatch(format!(
                "download {} > GpuBuffer len {}",
                out.len(),
                self.len
            )));
        }
        let mut halfs = vec![0u16; out.len()];
        let status = unsafe {
            (self.gpu.symbols.memcpy_d2h)(
                halfs.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                std::mem::size_of_val(halfs.as_slice()),
            )
        };
        check_cuda_status("cudaMemcpy D2H (f16)", status)?;
        for (dst, &bits) in out.iter_mut().zip(halfs.iter()) {
            *dst = crate::tensor::f16_to_f32(bits);
        }
        Ok(())
    }
}

#[cfg(not(target_os = "linux"))]
impl<'a, T> GpuBuffer<'a, T> {
    pub fn as_ptr(&self) -> *const T {
        std::ptr::null()
    }

    pub fn download(&self, _out: &mut [T]) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
}

#[cfg(not(target_os = "linux"))]
impl<'a> GpuBuffer<'a, f32> {
    pub fn upload_f32(&mut self, _data: &[f32]) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
}

#[cfg(not(target_os = "linux"))]
impl<'a> GpuBuffer<'a, u16> {
    pub fn upload_f32_as_f16(&mut self, data: &[f32]) -> Result<(), BackendError> {
        let _ = f32_to_f16(data.first().copied().unwrap_or(0.0));
        Err(linux_only_err())
    }

    pub fn copy_row_to_front(&mut self, _row_idx: usize, _row_len: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }

    pub fn download_as_f32(&self, _out: &mut [f32]) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
}

/// Backend d'inférence ternaire sur GPU CUDA (via `libaibox_gpu.so`).
pub struct GpuBackend {
    #[cfg(target_os = "linux")]
    symbols: CudaSymbols,
    #[cfg(target_os = "linux")]
    weight_cache: Mutex<HashMap<WeightCacheKey, GpuWeightPair>>,
    #[cfg(target_os = "linux")]
    raw_cache: Mutex<HashMap<(usize, usize), RawPtr>>,
}

#[cfg(not(target_os = "linux"))]
fn linux_only_err() -> BackendError {
    BackendError::UnsupportedFeature(
        "CUDA dynamic library backend is only supported on Linux".to_string(),
    )
}

#[cfg(target_os = "linux")]
fn check_cuda_status(op: &str, status: c_int) -> Result<(), BackendError> {
    if status == 0 {
        Ok(())
    } else {
        Err(BackendError::UnsupportedFeature(format!(
            "{op} failed with status {status}"
        )))
    }
}

#[cfg(target_os = "linux")]
unsafe fn load_symbol<T>(handle: *mut c_void, name: &str) -> Result<T, BackendError> {
    let c_name = CString::new(name)
        .map_err(|_| BackendError::UnsupportedFeature(format!("Invalid symbol name: {name}")))?;
    let ptr = unsafe { dlsym(handle, c_name.as_ptr()) };
    if ptr.is_null() {
        return Err(BackendError::UnsupportedFeature(format!(
            "Missing CUDA symbol `{name}` in libaibox_gpu.so"
        )));
    }
    Ok(unsafe { std::mem::transmute_copy(&ptr) })
}

#[cfg(target_os = "linux")]
fn open_cuda_library() -> Result<*mut c_void, BackendError> {
    let mut candidates = Vec::new();
    if let Ok(custom) = std::env::var("AIBOX_GPU_LIB") {
        candidates.push(custom);
    }
    candidates.extend([
        "libaibox_gpu.so".to_string(),
        "./libaibox_gpu.so".to_string(),
        "/usr/local/lib/libaibox_gpu.so".to_string(),
    ]);

    for path in &candidates {
        let Ok(c_path) = CString::new(path.as_str()) else {
            continue;
        };
        let handle = unsafe { dlopen(c_path.as_ptr(), RTLD_NOW) };
        if !handle.is_null() {
            return Ok(handle);
        }
    }

    Err(BackendError::UnsupportedFeature(format!(
        "Could not load CUDA library (tried: {})",
        candidates.join(", ")
    )))
}

#[cfg(target_os = "linux")]
impl GpuBackend {
    pub fn try_new() -> Result<Self, BackendError> {
        let handle = open_cuda_library()?;
        let load_result: Result<CudaSymbols, BackendError> = (|| unsafe {
            let is_available: FnIsAvailable = load_symbol(handle, "aibox_gpu_is_available")?;
            if is_available() == 0 {
                return Err(BackendError::UnsupportedFeature(
                    "No CUDA device available".to_string(),
                ));
            }
            Ok(CudaSymbols {
                handle,
                alloc: load_symbol(handle, "aibox_gpu_alloc")?,
                alloc_copy: load_symbol(handle, "aibox_gpu_alloc_copy")?,
                free: load_symbol(handle, "aibox_gpu_free")?,
                memcpy_h2d: load_symbol(handle, "aibox_gpu_memcpy_h2d")?,
                memcpy_d2h: load_symbol(handle, "aibox_gpu_memcpy_d2h")?,
                argmax_f32: load_symbol(handle, "aibox_gpu_argmax_f32")?,
                fwht128_f16: load_symbol(handle, "aibox_gpu_fwht128_f16")?,
                matmul_ternary_f16_dev: load_symbol(handle, "aibox_gpu_matmul_ternary_f16_dev")?,
                matmul_ternary_f16_add_dev: load_symbol(handle, "aibox_gpu_matmul_ternary_f16_add_dev")?,
                matmul_ternary_f16_to_f32_dev: load_symbol(handle, "aibox_gpu_matmul_ternary_f16_to_f32_dev")?,
                rms_norm_f16: load_symbol(handle, "aibox_gpu_rms_norm_f16")?,
                rms_norm_fwht128_f16: load_symbol(handle, "aibox_gpu_rms_norm_fwht128_f16")?,
                rms_norm_heads_rope_f16: load_symbol(handle, "aibox_gpu_rms_norm_heads_rope_f16")?,
                kv_append_gqa_f16: load_symbol(handle, "aibox_gpu_kv_append_gqa_f16")?,
                swiglu_inplace_f16: load_symbol(handle, "aibox_gpu_swiglu_inplace_f16")?,
                swiglu_fwht128_inplace_f16: load_symbol(handle, "aibox_gpu_swiglu_fwht128_inplace_f16")?,
                lm_head_f16_act_f16: load_symbol(handle, "aibox_gpu_lm_head_f16_act_f16")?,
                lm_head_f32_act_f16: load_symbol(handle, "aibox_gpu_lm_head_f32_act_f16")?,
                lm_head_quant_act_f16: load_symbol(handle, "aibox_gpu_lm_head_quant_act_f16")?,
                matmul_ternary_swiglu_f16_dev: load_symbol(handle, "aibox_gpu_matmul_ternary_swiglu_f16_dev")?,
                sample_candidates_f32: load_symbol(handle, "aibox_gpu_sample_candidates_f32")?,
                graph_begin_capture: load_symbol(handle, "aibox_gpu_graph_begin_capture")?,
                graph_end_capture: load_symbol(handle, "aibox_gpu_graph_end_capture")?,
                graph_launch: load_symbol(handle, "aibox_gpu_graph_launch")?,
                matmul_ternary: load_symbol(handle, "aibox_gpu_matmul_ternary")?,
                deinterleave_q_gate_f16: load_symbol(handle, "aibox_gpu_deinterleave_q_gate_f16")?,
                mul_sigmoid_inplace_f16: load_symbol(handle, "aibox_gpu_mul_sigmoid_inplace_f16")?,
                gdn_conv_silu_f16: load_symbol(handle, "aibox_gpu_gdn_conv_silu_f16")?,
                gdn_step_f16: load_symbol(handle, "aibox_gpu_gdn_step_f16")?,
                matmul_dense_f16_dev: load_symbol(handle, "aibox_gpu_matmul_dense_f16_dev")?,
                matmul_q2_affine_f16_dev: load_symbol(handle, "aibox_gpu_matmul_q2_affine_f16_dev")?,
                matmul_q4_affine_f16_dev: load_symbol(handle, "aibox_gpu_matmul_q4_affine_f16_dev")?,
                alloc_pinned: load_symbol(handle, "aibox_gpu_alloc_pinned")?,
                free_pinned: load_symbol(handle, "aibox_gpu_free_pinned")?,
                memcpy_h2d_async: load_symbol(handle, "aibox_gpu_memcpy_h2d_async")?,
                geglu_inplace_f16: load_symbol(handle, "aibox_gpu_geglu_inplace_f16")?,
                weighted_acc_f16_to_f32: load_symbol(handle, "aibox_gpu_weighted_acc_f16_to_f32")?,
                vram_free_bytes: load_symbol(handle, "aibox_gpu_vram_free_bytes")?,
                matmul_q2_affine_geglu_f16_dev: load_symbol(handle, "aibox_gpu_matmul_q2_affine_geglu_f16_dev")?,
                gemma4_norm_rope_f16: load_symbol(handle, "aibox_gpu_gemma4_norm_rope_f16")?,
                gemma4_attention_f16: load_symbol(handle, "aibox_gpu_gemma4_attention_f16")?,
                copy_d2d_bytes: load_symbol(handle, "aibox_gpu_copy_d2d_bytes")?,
                gemma4_moe_q2_batch: load_symbol(handle, "aibox_gpu_gemma4_moe_q2_batch")?,
            })
        })();

        match load_result {
            Ok(symbols) => Ok(Self {
                symbols,
                weight_cache: Mutex::new(HashMap::new()),
                raw_cache: Mutex::new(HashMap::new()),
            }),
            Err(err) => {
                unsafe { dlclose(handle) };
                Err(err)
            }
        }
    }

    fn alloc_typed<T>(&self, len: usize) -> Result<GpuBuffer<'_, T>, BackendError> {
        let bytes = len.max(1) * std::mem::size_of::<T>();
        let ptr = unsafe { (self.symbols.alloc)(bytes) } as *mut T;
        if ptr.is_null() {
            return Err(BackendError::UnsupportedFeature(format!(
                "cudaMalloc failed for {len} elements"
            )));
        }
        Ok(GpuBuffer { gpu: self, ptr, len })
    }

    pub fn alloc_f32(&self, len: usize) -> Result<GpuBuffer<'_, f32>, BackendError> {
        self.alloc_typed(len)
    }

    pub fn alloc_f16(&self, len: usize) -> Result<GpuBuffer<'_, u16>, BackendError> {
        self.alloc_typed(len)
    }

    pub fn argmax_f32(&self, logits_dev: &GpuBuffer<'_, f32>, len: usize) -> Result<usize, BackendError> {
        let mut best_idx: u32 = 0;
        let status = unsafe { (self.symbols.argmax_f32)(logits_dev.as_ptr(), len as c_int, &mut best_idx) };
        check_cuda_status("aibox_gpu_argmax_f32", status)?;
        Ok(best_idx as usize)
    }

    pub fn fwht128_inplace_f16(&self, buf: &mut GpuBuffer<'_, u16>, m: usize, k: usize) -> Result<(), BackendError> {
        let status = unsafe { (self.symbols.fwht128_f16)(buf.as_mut_ptr(), m as c_int, k as c_int) };
        check_cuda_status("aibox_gpu_fwht128_f16", status)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_matmul_ternary_f16_dev(
        &self,
        op_name: &str,
        func: FnMatmulTernaryF16Dev,
        x: &GpuBuffer<'_, u16>,
        packed_weights: &[u8],
        scales_fp16: &[u16],
        out: &mut GpuBuffer<'_, u16>,
        m: usize,
        n: usize,
        k: usize,
        bits: u8,
    ) -> Result<(), BackendError> {
        let (dev_w, dev_scales) = self.get_or_upload_weights(packed_weights, scales_fp16)?;
        let status = unsafe {
            func(x.as_ptr(), dev_w, dev_scales, out.as_mut_ptr(), m as c_int, n as c_int, k as c_int, bits as c_int)
        };
        check_cuda_status(op_name, status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_ternary_f16_dev(
        &self,
        x: &GpuBuffer<'_, u16>,
        packed_weights: &[u8],
        scales_fp16: &[u16],
        out: &mut GpuBuffer<'_, u16>,
        m: usize,
        n: usize,
        k: usize,
        bits: u8,
    ) -> Result<(), BackendError> {
        self.run_matmul_ternary_f16_dev(
            "aibox_gpu_matmul_ternary_f16_dev",
            self.symbols.matmul_ternary_f16_dev,
            x,
            packed_weights,
            scales_fp16,
            out,
            m,
            n,
            k,
            bits,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_ternary_f16_add_dev(
        &self,
        x: &GpuBuffer<'_, u16>,
        packed_weights: &[u8],
        scales_fp16: &[u16],
        out: &mut GpuBuffer<'_, u16>,
        m: usize,
        n: usize,
        k: usize,
        bits: u8,
    ) -> Result<(), BackendError> {
        self.run_matmul_ternary_f16_dev(
            "aibox_gpu_matmul_ternary_f16_add_dev",
            self.symbols.matmul_ternary_f16_add_dev,
            x,
            packed_weights,
            scales_fp16,
            out,
            m,
            n,
            k,
            bits,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_ternary_f16_to_f32_dev(
        &self,
        x: &GpuBuffer<'_, u16>,
        packed_weights: &[u8],
        scales_fp16: &[u16],
        out: &mut GpuBuffer<'_, f32>,
        m: usize,
        n: usize,
        k: usize,
        bits: u8,
    ) -> Result<(), BackendError> {
        let (dev_w, dev_scales) = self.get_or_upload_weights(packed_weights, scales_fp16)?;
        let status = unsafe {
            (self.symbols.matmul_ternary_f16_to_f32_dev)(
                x.as_ptr(),
                dev_w,
                dev_scales,
                out.as_mut_ptr(),
                m as c_int,
                n as c_int,
                k as c_int,
                bits as c_int,
            )
        };
        check_cuda_status("aibox_gpu_matmul_ternary_f16_to_f32_dev", status)
    }

    pub fn rms_norm_f16(
        &self,
        x: &GpuBuffer<'_, u16>,
        weight: &[f32],
        out: &mut GpuBuffer<'_, u16>,
        m: usize,
        dim: usize,
        eps: f32,
    ) -> Result<(), BackendError> {
        let dev_w = self.get_or_upload_raw_f32(weight)?;
        let status = unsafe {
            (self.symbols.rms_norm_f16)(x.as_ptr(), dev_w, out.as_mut_ptr(), m as c_int, dim as c_int, eps)
        };
        check_cuda_status("aibox_gpu_rms_norm_f16", status)
    }

    pub fn rms_norm_fwht128_f16(
        &self,
        x: &GpuBuffer<'_, u16>,
        weight: &[f32],
        out: &mut GpuBuffer<'_, u16>,
        m: usize,
        dim: usize,
        eps: f32,
    ) -> Result<(), BackendError> {
        let dev_w = self.get_or_upload_raw_f32(weight)?;
        let status = unsafe {
            (self.symbols.rms_norm_fwht128_f16)(x.as_ptr(), dev_w, out.as_mut_ptr(), m as c_int, dim as c_int, eps)
        };
        check_cuda_status("aibox_gpu_rms_norm_fwht128_f16", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_heads_rope_f16(
        &self,
        vec: &mut GpuBuffer<'_, u16>,
        norm_w: Option<&[f32]>,
        m: usize,
        num_heads: usize,
        head_dim: usize,
        rope_dim: usize,
        start_pos: usize,
        theta: f32,
        eps: f32,
    ) -> Result<(), BackendError> {
        let dev_norm = match norm_w {
            Some(w) => self.get_or_upload_raw_f32(w)?,
            None => std::ptr::null(),
        };
        let status = unsafe {
            (self.symbols.rms_norm_heads_rope_f16)(
                vec.as_mut_ptr(),
                dev_norm,
                m as c_int,
                num_heads as c_int,
                head_dim as c_int,
                rope_dim as c_int,
                start_pos as c_int,
                theta,
                eps,
            )
        };
        check_cuda_status("aibox_gpu_rms_norm_heads_rope_f16", status)
    }

    pub fn deinterleave_q_gate_f16(
        &self,
        q_full: &GpuBuffer<'_, u16>,
        q_out: &mut GpuBuffer<'_, u16>,
        gate_out: &mut GpuBuffer<'_, u16>,
        m: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<(), BackendError> {
        let status = unsafe {
            (self.symbols.deinterleave_q_gate_f16)(
                q_full.as_ptr(),
                q_out.as_mut_ptr(),
                gate_out.as_mut_ptr(),
                m as c_int,
                num_heads as c_int,
                head_dim as c_int,
            )
        };
        check_cuda_status("aibox_gpu_deinterleave_q_gate_f16", status)
    }

    pub fn mul_sigmoid_inplace_f16(
        &self,
        out: &mut GpuBuffer<'_, u16>,
        gate: &GpuBuffer<'_, u16>,
        len: usize,
    ) -> Result<(), BackendError> {
        let status = unsafe { (self.symbols.mul_sigmoid_inplace_f16)(out.as_mut_ptr(), gate.as_ptr(), len as c_int) };
        check_cuda_status("aibox_gpu_mul_sigmoid_inplace_f16", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gdn_conv_silu_f16(
        &self,
        qkv_in: &GpuBuffer<'_, u16>,
        qkv_in_offset: usize,
        conv_w: &[f32],
        conv_state: &mut GpuBuffer<'_, f32>,
        conv_state_offset: usize,
        qkv_out: &mut GpuBuffer<'_, u16>,
        conv_channels: usize,
        kernel: usize,
    ) -> Result<(), BackendError> {
        let dev_conv_w = self.get_or_upload_raw_f32(conv_w)?;
        let status = unsafe {
            (self.symbols.gdn_conv_silu_f16)(
                qkv_in.as_ptr().add(qkv_in_offset),
                dev_conv_w,
                conv_state.as_mut_ptr().add(conv_state_offset),
                qkv_out.as_mut_ptr(),
                conv_channels as c_int,
                kernel as c_int,
            )
        };
        check_cuda_status("aibox_gpu_gdn_conv_silu_f16", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gdn_step_f16(
        &self,
        qkv: &GpuBuffer<'_, u16>,
        alpha: &GpuBuffer<'_, u16>,
        alpha_offset: usize,
        beta: &GpuBuffer<'_, u16>,
        beta_offset: usize,
        ssm_a: &[f32],
        ssm_dt_bias: &[f32],
        ssm_norm: &[f32],
        z: &GpuBuffer<'_, u16>,
        z_offset: usize,
        ssm_state: &mut GpuBuffer<'_, u16>,
        ssm_state_offset: usize,
        out: &mut GpuBuffer<'_, u16>,
        out_offset: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<(), BackendError> {
        let dev_a = self.get_or_upload_raw_f32(ssm_a)?;
        let dev_dt = self.get_or_upload_raw_f32(ssm_dt_bias)?;
        let dev_norm = self.get_or_upload_raw_f32(ssm_norm)?;
        let status = unsafe {
            (self.symbols.gdn_step_f16)(
                qkv.as_ptr(),
                alpha.as_ptr().add(alpha_offset),
                beta.as_ptr().add(beta_offset),
                dev_a,
                dev_dt,
                dev_norm,
                z.as_ptr().add(z_offset),
                ssm_state.as_mut_ptr().add(ssm_state_offset),
                out.as_mut_ptr().add(out_offset),
                num_k_heads as c_int,
                num_v_heads as c_int,
                head_dim as c_int,
                eps,
            )
        };
        check_cuda_status("aibox_gpu_gdn_step_f16", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_dense_f16_dev(
        &self,
        x: &GpuBuffer<'_, u16>,
        w_f16: &[u16],
        out: &mut GpuBuffer<'_, u16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BackendError> {
        let dev_w = self.get_or_upload_raw_u16(w_f16)?;
        let status = unsafe {
            (self.symbols.matmul_dense_f16_dev)(
                x.as_ptr(),
                dev_w,
                out.as_mut_ptr(),
                m as c_int,
                n as c_int,
                k as c_int,
            )
        };
        check_cuda_status("aibox_gpu_matmul_dense_f16_dev", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kv_append_gqa_layer_f16(
        &self,
        q: &GpuBuffer<'_, u16>,
        k: &GpuBuffer<'_, u16>,
        v: &GpuBuffer<'_, u16>,
        k_cache: &mut GpuBuffer<'_, u16>,
        v_cache: &mut GpuBuffer<'_, u16>,
        layer_offset: usize,
        out: &mut GpuBuffer<'_, u16>,
        start_pos: usize,
        m: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(), BackendError> {
        let status = unsafe {
            (self.symbols.kv_append_gqa_f16)(
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                k_cache.as_mut_ptr().add(layer_offset),
                v_cache.as_mut_ptr().add(layer_offset),
                out.as_mut_ptr(),
                start_pos as c_int,
                m as c_int,
                num_q_heads as c_int,
                num_kv_heads as c_int,
                head_dim as c_int,
            )
        };
        check_cuda_status("aibox_gpu_kv_append_gqa_f16", status)
    }

    pub fn swiglu_inplace_f16(
        &self,
        gate: &mut GpuBuffer<'_, u16>,
        up: &GpuBuffer<'_, u16>,
        len: usize,
    ) -> Result<(), BackendError> {
        let status = unsafe { (self.symbols.swiglu_inplace_f16)(gate.as_mut_ptr(), up.as_ptr(), len as c_int) };
        check_cuda_status("aibox_gpu_swiglu_inplace_f16", status)
    }

    pub fn swiglu_fwht128_inplace_f16(
        &self,
        gate: &mut GpuBuffer<'_, u16>,
        up: &GpuBuffer<'_, u16>,
        len: usize,
    ) -> Result<(), BackendError> {
        let status = unsafe {
            (self.symbols.swiglu_fwht128_inplace_f16)(gate.as_mut_ptr(), up.as_ptr(), len as c_int)
        };
        check_cuda_status("aibox_gpu_swiglu_fwht128_inplace_f16", status)
    }

    pub fn lm_head_f16_act_f16(
        &self,
        hidden: &GpuBuffer<'_, u16>,
        emb_f16: &[u16],
        logits_dev: &mut GpuBuffer<'_, f32>,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<(), BackendError> {
        let dev_emb = self.get_or_upload_raw_u16(emb_f16)?;
        let status = unsafe {
            (self.symbols.lm_head_f16_act_f16)(
                hidden.as_ptr(),
                dev_emb,
                logits_dev.as_mut_ptr(),
                vocab_size as c_int,
                hidden_size as c_int,
            )
        };
        check_cuda_status("aibox_gpu_lm_head_f16_act_f16", status)
    }

    pub fn lm_head_f32_act_f16(
        &self,
        hidden: &GpuBuffer<'_, u16>,
        emb_f32: &[f32],
        logits_dev: &mut GpuBuffer<'_, f32>,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<(), BackendError> {
        let dev_emb = self.get_or_upload_raw_f32(emb_f32)?;
        let status = unsafe {
            (self.symbols.lm_head_f32_act_f16)(
                hidden.as_ptr(),
                dev_emb,
                logits_dev.as_mut_ptr(),
                vocab_size as c_int,
                hidden_size as c_int,
            )
        };
        check_cuda_status("aibox_gpu_lm_head_f32_act_f16", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn lm_head_quant_act_f16(
        &self,
        hidden: &GpuBuffer<'_, u16>,
        packed_weights: &[u8],
        scales_fp16: &[u16],
        logits_dev: &mut GpuBuffer<'_, f32>,
        vocab_size: usize,
        hidden_size: usize,
        bits: u8,
    ) -> Result<(), BackendError> {
        let (dev_w, dev_scales) = self.get_or_upload_weights(packed_weights, scales_fp16)?;
        let status = unsafe {
            (self.symbols.lm_head_quant_act_f16)(
                hidden.as_ptr(),
                dev_w,
                dev_scales,
                logits_dev.as_mut_ptr(),
                vocab_size as c_int,
                hidden_size as c_int,
                bits as c_int,
            )
        };
        check_cuda_status("aibox_gpu_lm_head_quant_act_f16", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_ternary_swiglu_f16_dev(
        &self,
        x: &GpuBuffer<'_, u16>,
        gate_w: &[u8],
        gate_scales: &[u16],
        up_w: &[u8],
        up_scales: &[u16],
        out: &mut GpuBuffer<'_, u16>,
        n: usize,
        k: usize,
        bits: u8,
    ) -> Result<(), BackendError> {
        let (dev_wg, dev_sg) = self.get_or_upload_weights(gate_w, gate_scales)?;
        let (dev_wu, dev_su) = self.get_or_upload_weights(up_w, up_scales)?;
        let status = unsafe {
            (self.symbols.matmul_ternary_swiglu_f16_dev)(
                x.as_ptr(),
                dev_wg,
                dev_sg,
                dev_wu,
                dev_su,
                out.as_mut_ptr(),
                n as c_int,
                k as c_int,
                bits as c_int,
            )
        };
        check_cuda_status("aibox_gpu_matmul_ternary_swiglu_f16_dev", status)
    }

    pub fn sample_candidates_f32(
        &self,
        logits: &GpuBuffer<'_, f32>,
        len: usize,
        margin: f32,
        max_out: usize,
    ) -> Result<Option<Vec<(usize, f32)>>, BackendError> {
        const STACK_CAP: usize = 1024;
        if max_out > STACK_CAP {
            return Ok(None);
        }
        let mut idxs = [0u32; STACK_CAP];
        let mut vals = [0.0f32; STACK_CAP];
        let mut count: c_int = 0;
        let status = unsafe {
            (self.symbols.sample_candidates_f32)(
                logits.as_ptr(),
                len as c_int,
                margin,
                idxs.as_mut_ptr(),
                vals.as_mut_ptr(),
                max_out as c_int,
                &mut count,
            )
        };
        check_cuda_status("aibox_gpu_sample_candidates_f32", status)?;
        let count_usize = count.max(0) as usize;
        if count_usize > max_out {
            return Ok(None);
        }
        Ok(Some(
            idxs[..count_usize]
                .iter()
                .zip(vals[..count_usize].iter())
                .map(|(&idx, &val)| (idx as usize, val))
                .collect(),
        ))
    }

    pub fn graph_begin_capture(&self) -> Result<(), BackendError> {
        let status = unsafe { (self.symbols.graph_begin_capture)() };
        check_cuda_status("aibox_gpu_graph_begin_capture", status)
    }

    pub fn graph_end_capture(&self) -> Result<(), BackendError> {
        let status = unsafe { (self.symbols.graph_end_capture)() };
        check_cuda_status("aibox_gpu_graph_end_capture", status)
    }

    pub fn graph_launch(
        &self,
        hidden: &mut GpuBuffer<'_, u16>,
        emb_f32: &[f32],
        pos: usize,
    ) -> Result<(), BackendError> {
        let halfs: Vec<u16> = emb_f32.iter().map(|&v| f32_to_f16(v)).collect();
        let status = unsafe {
            (self.symbols.graph_launch)(
                hidden.as_mut_ptr(),
                halfs.as_ptr(),
                halfs.len() as c_int,
                pos as c_int,
            )
        };
        check_cuda_status("aibox_gpu_graph_launch", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q2_affine_f16_dev(
        &self,
        x_dev: &GpuBuffer<'_, u16>,
        packed_weights: &[u8],
        scales_fp16: &[u16],
        zeros_fp16: &[u16],
        out_dev: &mut GpuBuffer<'_, u16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BackendError> {
        let dev_w = self.get_or_upload_raw_bytes(packed_weights.as_ptr() as *const c_void, packed_weights.len())?;
        let dev_s = self.get_or_upload_raw_bytes(
            scales_fp16.as_ptr() as *const c_void,
            std::mem::size_of_val(scales_fp16),
        )?;
        let dev_z = self.get_or_upload_raw_bytes(
            zeros_fp16.as_ptr() as *const c_void,
            std::mem::size_of_val(zeros_fp16),
        )?;
        let status = unsafe {
            (self.symbols.matmul_q2_affine_f16_dev)(
                x_dev.as_ptr(),
                dev_w as *const u8,
                dev_s as *const u16,
                dev_z as *const u16,
                out_dev.as_mut_ptr(),
                m as c_int,
                n as c_int,
                k as c_int,
            )
        };
        check_cuda_status("aibox_gpu_matmul_q2_affine_f16_dev", status)
    }

    pub fn alloc_u8(&self, len: usize) -> Result<GpuBuffer<'_, u8>, BackendError> {
        let ptr = unsafe { (self.symbols.alloc)(len.max(1)) };
        if ptr.is_null() {
            return Err(BackendError::UnsupportedFeature("cudaMalloc u8 failed".to_string()));
        }
        Ok(GpuBuffer { gpu: self, ptr: ptr as *mut u8, len })
    }

    pub fn alloc_u16(&self, len: usize) -> Result<GpuBuffer<'_, u16>, BackendError> {
        self.alloc_f16(len)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q4_affine_f16_dev(
        &self,
        x_dev: &GpuBuffer<'_, u16>,
        packed_weights: &[u8],
        scales_fp16: &[u16],
        zeros_fp16: &[u16],
        out_dev: &mut GpuBuffer<'_, u16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BackendError> {
        let dev_w = self.get_or_upload_raw_bytes(packed_weights.as_ptr() as *const c_void, packed_weights.len())?;
        let dev_s = self.get_or_upload_raw_bytes(
            scales_fp16.as_ptr() as *const c_void,
            std::mem::size_of_val(scales_fp16),
        )?;
        let dev_z = self.get_or_upload_raw_bytes(
            zeros_fp16.as_ptr() as *const c_void,
            std::mem::size_of_val(zeros_fp16),
        )?;
        let status = unsafe {
            (self.symbols.matmul_q4_affine_f16_dev)(
                x_dev.as_ptr(),
                dev_w as *const u8,
                dev_s as *const u16,
                dev_z as *const u16,
                out_dev.as_mut_ptr(),
                m as c_int,
                n as c_int,
                k as c_int,
            )
        };
        check_cuda_status("aibox_gpu_matmul_q4_affine_f16_dev", status)
    }

    /// Streame les poids `q4_affine` depuis la RAM (ou RAM pinnée) vers des tampons VRAM réutilisables
    /// sans saturer `raw_cache` (pour les experts froids MoE sur GPU 6 Go).
    ///
    /// # Safety
    /// `pinned_host` doit pointer vers un tampon valide d'au moins `packed_len + scales_bytes + zeros_bytes`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_q4_affine_f16_streaming_dev(
        &self,
        x_dev: &GpuBuffer<'_, u16>,
        pinned_host: *const u8,
        packed_len: usize,
        scales_len: usize,
        stage_w: &mut GpuBuffer<'_, u8>,
        stage_s: &mut GpuBuffer<'_, u16>,
        stage_z: &mut GpuBuffer<'_, u16>,
        out_dev: &mut GpuBuffer<'_, u16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BackendError> {
        let s_bytes = scales_len * std::mem::size_of::<u16>();
        unsafe {
            self.memcpy_h2d_async(stage_w, pinned_host, packed_len)?;
            let st_s = (self.symbols.memcpy_h2d_async)(
                stage_s.as_mut_ptr() as *mut c_void,
                pinned_host.add(packed_len) as *const c_void,
                s_bytes,
            );
            check_cuda_status("aibox_gpu_memcpy_h2d_async (scales)", st_s)?;
            let st_z = (self.symbols.memcpy_h2d_async)(
                stage_z.as_mut_ptr() as *mut c_void,
                pinned_host.add(packed_len + s_bytes) as *const c_void,
                s_bytes,
            );
            check_cuda_status("aibox_gpu_memcpy_h2d_async (zeros)", st_z)?;
        }
        let status = unsafe {
            (self.symbols.matmul_q4_affine_f16_dev)(
                x_dev.as_ptr(),
                stage_w.as_ptr(),
                stage_s.as_ptr(),
                stage_z.as_ptr(),
                out_dev.as_mut_ptr(),
                m as c_int,
                n as c_int,
                k as c_int,
            )
        };
        check_cuda_status("aibox_gpu_matmul_q4_affine_f16_dev (streaming)", status)
    }

    pub fn alloc_pinned_bytes(&self, bytes: usize) -> Result<*mut u8, BackendError> {
        let ptr = unsafe { (self.symbols.alloc_pinned)(bytes) };
        if ptr.is_null() {
            return Err(BackendError::UnsupportedFeature("cudaHostAlloc failed".to_string()));
        }
        Ok(ptr as *mut u8)
    }

    /// # Safety
    /// `ptr` must have been allocated by `alloc_pinned_bytes`.
    pub unsafe fn free_pinned_bytes(&self, ptr: *mut u8) {
        if !ptr.is_null() {
            unsafe { (self.symbols.free_pinned)(ptr as *mut c_void) };
        }
    }

    /// # Safety
    /// `host_src` must be valid for `bytes` and `dev_dst` must have at least `bytes` capacity.
    pub unsafe fn memcpy_h2d_async(
        &self,
        dev_dst: &mut GpuBuffer<'_, u8>,
        host_src: *const u8,
        bytes: usize,
    ) -> Result<(), BackendError> {
        let status = unsafe {
            (self.symbols.memcpy_h2d_async)(dev_dst.as_mut_ptr() as *mut c_void, host_src as *const c_void, bytes)
        };
        check_cuda_status("aibox_gpu_memcpy_h2d_async", status)
    }

    pub fn upload_u8_slice(&self, dev_dst: &mut GpuBuffer<'_, u8>, src: &[u8]) -> Result<(), BackendError> {
        if dev_dst.len() < src.len() {
            return Err(BackendError::BufferLengthMismatch("upload_u8_slice overflow".to_string()));
        }
        let status = unsafe {
            (self.symbols.memcpy_h2d_async)(
                dev_dst.as_mut_ptr() as *mut c_void,
                src.as_ptr() as *const c_void,
                src.len(),
            )
        };
        check_cuda_status("aibox_gpu_memcpy_h2d_async (u8)", status)
    }

    pub fn upload_u16_slice(&self, dev_dst: &mut GpuBuffer<'_, u16>, src: &[u16]) -> Result<(), BackendError> {
        if dev_dst.len() < src.len() {
            return Err(BackendError::BufferLengthMismatch("upload_u16_slice overflow".to_string()));
        }
        let status = unsafe {
            (self.symbols.memcpy_h2d_async)(
                dev_dst.as_mut_ptr() as *mut c_void,
                src.as_ptr() as *const c_void,
                std::mem::size_of_val(src),
            )
        };
        check_cuda_status("aibox_gpu_memcpy_h2d_async (u16)", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q4_affine_buffers_dev(
        &self,
        x_dev: &GpuBuffer<'_, u16>,
        w_dev: &GpuBuffer<'_, u8>,
        s_dev: &GpuBuffer<'_, u16>,
        z_dev: &GpuBuffer<'_, u16>,
        out_dev: &mut GpuBuffer<'_, u16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BackendError> {
        let status = unsafe {
            (self.symbols.matmul_q4_affine_f16_dev)(
                x_dev.as_ptr(),
                w_dev.as_ptr(),
                s_dev.as_ptr(),
                z_dev.as_ptr(),
                out_dev.as_mut_ptr(),
                m as c_int,
                n as c_int,
                k as c_int,
            )
        };
        check_cuda_status("aibox_gpu_matmul_q4_affine_f16_dev (buffers)", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q2_affine_buffers_dev(
        &self,
        x_dev: &GpuBuffer<'_, u16>,
        w_dev: &GpuBuffer<'_, u8>,
        s_dev: &GpuBuffer<'_, u16>,
        z_dev: &GpuBuffer<'_, u16>,
        out_dev: &mut GpuBuffer<'_, u16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BackendError> {
        let status = unsafe {
            (self.symbols.matmul_q2_affine_f16_dev)(
                x_dev.as_ptr(),
                w_dev.as_ptr(),
                s_dev.as_ptr(),
                z_dev.as_ptr(),
                out_dev.as_mut_ptr(),
                m as c_int,
                n as c_int,
                k as c_int,
            )
        };
        check_cuda_status("aibox_gpu_matmul_q2_affine_f16_dev (buffers)", status)
    }

    pub fn geglu_inplace_f16(
        &self,
        gate_dev: &mut GpuBuffer<'_, u16>,
        up_dev: &GpuBuffer<'_, u16>,
        len: usize,
    ) -> Result<(), BackendError> {
        let status = unsafe {
            (self.symbols.geglu_inplace_f16)(gate_dev.as_mut_ptr(), up_dev.as_ptr(), len as c_int)
        };
        check_cuda_status("aibox_gpu_geglu_inplace_f16", status)
    }

    pub fn weighted_acc_f16_to_f32(
        &self,
        acc_dev: &mut GpuBuffer<'_, f32>,
        x_dev: &GpuBuffer<'_, u16>,
        weight: f32,
        len: usize,
        reset: bool,
    ) -> Result<(), BackendError> {
        let status = unsafe {
            (self.symbols.weighted_acc_f16_to_f32)(
                acc_dev.as_mut_ptr(),
                x_dev.as_ptr(),
                weight,
                len as c_int,
                i32::from(reset),
            )
        };
        check_cuda_status("aibox_gpu_weighted_acc_f16_to_f32", status)
    }

    pub fn vram_free_bytes(&self) -> usize {
        unsafe { (self.symbols.vram_free_bytes)() }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q2_affine_geglu_buffers_dev(
        &self,
        x_dev: &GpuBuffer<'_, u16>,
        wg_dev: &GpuBuffer<'_, u8>,
        sg_dev: &GpuBuffer<'_, u16>,
        zg_dev: &GpuBuffer<'_, u16>,
        wu_dev: &GpuBuffer<'_, u8>,
        su_dev: &GpuBuffer<'_, u16>,
        zu_dev: &GpuBuffer<'_, u16>,
        out_dev: &mut GpuBuffer<'_, u16>,
        n: usize,
        k: usize,
    ) -> Result<(), BackendError> {
        let status = unsafe {
            (self.symbols.matmul_q2_affine_geglu_f16_dev)(
                x_dev.as_ptr(),
                wg_dev.as_ptr(),
                sg_dev.as_ptr(),
                zg_dev.as_ptr(),
                wu_dev.as_ptr(),
                su_dev.as_ptr(),
                zu_dev.as_ptr(),
                out_dev.as_mut_ptr(),
                n as c_int,
                k as c_int,
            )
        };
        check_cuda_status("aibox_gpu_matmul_q2_affine_geglu_f16_dev", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemma4_norm_rope_f16(
        &self,
        vec_dev: &mut GpuBuffer<'_, u16>,
        norm_w: Option<&[f32]>,
        rope_freqs: Option<&[f32]>,
        num_heads: usize,
        head_dim: usize,
        rope_dim: usize,
        pos: usize,
        theta: f32,
        eps: f32,
    ) -> Result<(), BackendError> {
        let dev_norm = match norm_w {
            Some(w) => self.get_or_upload_raw_bytes(w.as_ptr() as *const c_void, std::mem::size_of_val(w))? as *const f32,
            None => std::ptr::null(),
        };
        let dev_freqs = match rope_freqs {
            Some(rf) => self.get_or_upload_raw_bytes(rf.as_ptr() as *const c_void, std::mem::size_of_val(rf))? as *const f32,
            None => std::ptr::null(),
        };
        let status = unsafe {
            (self.symbols.gemma4_norm_rope_f16)(
                vec_dev.as_mut_ptr(),
                dev_norm,
                dev_freqs,
                num_heads as c_int,
                head_dim as c_int,
                rope_dim as c_int,
                pos as c_int,
                theta,
                eps,
            )
        };
        check_cuda_status("aibox_gpu_gemma4_norm_rope_f16", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemma4_attention_f16(
        &self,
        q_dev: &GpuBuffer<'_, u16>,
        k_new_dev: &GpuBuffer<'_, u16>,
        v_new_dev: &GpuBuffer<'_, u16>,
        k_cache_dev: &mut GpuBuffer<'_, u16>,
        v_cache_dev: &mut GpuBuffer<'_, u16>,
        out_dev: &mut GpuBuffer<'_, u16>,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        pos: usize,
        window_size: usize,
    ) -> Result<(), BackendError> {
        let status = unsafe {
            (self.symbols.gemma4_attention_f16)(
                q_dev.as_ptr(),
                k_new_dev.as_ptr(),
                v_new_dev.as_ptr(),
                k_cache_dev.as_mut_ptr(),
                v_cache_dev.as_mut_ptr(),
                out_dev.as_mut_ptr(),
                num_q_heads as c_int,
                num_kv_heads as c_int,
                head_dim as c_int,
                pos as c_int,
                window_size as c_int,
            )
        };
        check_cuda_status("aibox_gpu_gemma4_attention_f16", status)
    }

    pub fn copy_d2d_u16(
        &self,
        dst_dev: &mut GpuBuffer<'_, u16>,
        src_dev: &GpuBuffer<'_, u16>,
        len: usize,
    ) -> Result<(), BackendError> {
        let status = unsafe {
            (self.symbols.copy_d2d_bytes)(
                dst_dev.as_mut_ptr() as *mut c_void,
                src_dev.as_ptr() as *const c_void,
                len * std::mem::size_of::<u16>(),
            )
        };
        check_cuda_status("aibox_gpu_copy_d2d_bytes", status)
    }

    pub fn gemma4_moe_q2_batch(
        &self,
        x_dev: &GpuBuffer<'_, u16>,
        batch: &Gemma4MoeBatchArgs,
        gate_scratch_dev: &mut GpuBuffer<'_, u16>,
        acc_out_dev: &mut GpuBuffer<'_, f32>,
        inter_dim: usize,
        hidden_dim: usize,
    ) -> Result<(), BackendError> {
        let status = unsafe {
            (self.symbols.gemma4_moe_q2_batch)(
                x_dev.as_ptr(),
                batch as *const Gemma4MoeBatchArgs,
                gate_scratch_dev.as_mut_ptr(),
                acc_out_dev.as_mut_ptr(),
                inter_dim as c_int,
                hidden_dim as c_int,
            )
        };
        check_cuda_status("aibox_gpu_gemma4_moe_q2_batch", status)
    }

    fn get_or_upload_raw_bytes(&self, host_ptr: *const c_void, byte_len: usize) -> Result<*mut c_void, BackendError> {
        let key = (host_ptr as usize, byte_len);
        let mut guard = self.raw_cache.lock().map_err(|_| {
            BackendError::UnsupportedFeature("GPU raw cache mutex poisoned".to_string())
        })?;
        if let Some(entry) = guard.get(&key) {
            return Ok(entry.0);
        }
        let dev_ptr = unsafe { (self.symbols.alloc_copy)(host_ptr, byte_len) };
        if dev_ptr.is_null() {
            return Err(BackendError::UnsupportedFeature(
                "cudaMalloc/cudaMemcpy failed for raw tensor".to_string(),
            ));
        }
        guard.insert(key, RawPtr(dev_ptr));
        Ok(dev_ptr)
    }

    fn get_or_upload_raw_f32(&self, slice: &[f32]) -> Result<*const f32, BackendError> {
        let ptr = self.get_or_upload_raw_bytes(slice.as_ptr() as *const c_void, std::mem::size_of_val(slice))?;
        Ok(ptr as *const f32)
    }

    fn get_or_upload_raw_u16(&self, slice: &[u16]) -> Result<*const u16, BackendError> {
        let ptr = self.get_or_upload_raw_bytes(slice.as_ptr() as *const c_void, std::mem::size_of_val(slice))?;
        Ok(ptr as *const u16)
    }

    fn alloc_weight_pair(&self, packed_weights: &[u8], scales_fp16: &[u16]) -> Result<GpuWeightPair, BackendError> {
        let dev_w = unsafe {
            (self.symbols.alloc_copy)(packed_weights.as_ptr() as *const c_void, std::mem::size_of_val(packed_weights))
        };
        if dev_w.is_null() {
            return Err(BackendError::UnsupportedFeature(
                "cudaMalloc/cudaMemcpy failed for packed_weights".to_string(),
            ));
        }

        let dev_scales = unsafe {
            (self.symbols.alloc_copy)(scales_fp16.as_ptr() as *const c_void, std::mem::size_of_val(scales_fp16))
        };
        if dev_scales.is_null() {
            unsafe { (self.symbols.free)(dev_w) };
            return Err(BackendError::UnsupportedFeature(
                "cudaMalloc/cudaMemcpy failed for scales_fp16".to_string(),
            ));
        }

        Ok(GpuWeightPair { dev_w, dev_scales })
    }

    fn get_or_upload_weights(
        &self,
        packed_weights: &[u8],
        scales_fp16: &[u16],
    ) -> Result<(*const u8, *const u16), BackendError> {
        let key: WeightCacheKey = (
            packed_weights.as_ptr() as usize,
            packed_weights.len(),
            scales_fp16.as_ptr() as usize,
            scales_fp16.len(),
        );
        let mut guard = self.weight_cache.lock().map_err(|_| {
            BackendError::UnsupportedFeature("GPU weight cache mutex poisoned".to_string())
        })?;
        if let Some(entry) = guard.get(&key) {
            return Ok((entry.dev_w as *const u8, entry.dev_scales as *const u16));
        }

        let pair = self.alloc_weight_pair(packed_weights, scales_fp16)?;
        let ptrs = (pair.dev_w as *const u8, pair.dev_scales as *const u16);
        guard.insert(key, pair);
        Ok(ptrs)
    }
}

#[cfg(not(target_os = "linux"))]
impl GpuBackend {
    pub fn try_new() -> Result<Self, BackendError> {
        Err(linux_only_err())
    }
    pub fn alloc_f32(&self, _len: usize) -> Result<GpuBuffer<'_, f32>, BackendError> {
        Err(linux_only_err())
    }
    pub fn argmax_f32(&self, _logits_dev: &GpuBuffer<'_, f32>, _len: usize) -> Result<usize, BackendError> {
        Err(linux_only_err())
    }
    pub fn alloc_f16(&self, _len: usize) -> Result<GpuBuffer<'_, u16>, BackendError> {
        Err(linux_only_err())
    }
    pub fn fwht128_inplace_f16(&self, _buf: &mut GpuBuffer<'_, u16>, _m: usize, _k: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_ternary_f16_dev(&self, _x: &GpuBuffer<'_, u16>, _pw: &[u8], _sc: &[u16], _out: &mut GpuBuffer<'_, u16>, _m: usize, _n: usize, _k: usize, _bits: u8) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_ternary_f16_add_dev(&self, _x: &GpuBuffer<'_, u16>, _pw: &[u8], _sc: &[u16], _out: &mut GpuBuffer<'_, u16>, _m: usize, _n: usize, _k: usize, _bits: u8) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_ternary_f16_to_f32_dev(&self, _x: &GpuBuffer<'_, u16>, _pw: &[u8], _sc: &[u16], _out: &mut GpuBuffer<'_, f32>, _m: usize, _n: usize, _k: usize, _bits: u8) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn rms_norm_f16(&self, _x: &GpuBuffer<'_, u16>, _w: &[f32], _out: &mut GpuBuffer<'_, u16>, _m: usize, _dim: usize, _eps: f32) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn rms_norm_fwht128_f16(&self, _x: &GpuBuffer<'_, u16>, _w: &[f32], _out: &mut GpuBuffer<'_, u16>, _m: usize, _dim: usize, _eps: f32) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_heads_rope_f16(&self, _vec: &mut GpuBuffer<'_, u16>, _nw: Option<&[f32]>, _m: usize, _nh: usize, _hd: usize, _rd: usize, _pos: usize, _theta: f32, _eps: f32) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn deinterleave_q_gate_f16(&self, _qf: &GpuBuffer<'_, u16>, _qo: &mut GpuBuffer<'_, u16>, _go: &mut GpuBuffer<'_, u16>, _m: usize, _nh: usize, _hd: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn mul_sigmoid_inplace_f16(&self, _out: &mut GpuBuffer<'_, u16>, _gate: &GpuBuffer<'_, u16>, _len: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_conv_silu_f16(&self, _qi: &GpuBuffer<'_, u16>, _qio: usize, _cw: &[f32], _cs: &mut GpuBuffer<'_, f32>, _cso: usize, _qo: &mut GpuBuffer<'_, u16>, _cc: usize, _k: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_step_f16(&self, _qkv: &GpuBuffer<'_, u16>, _a: &GpuBuffer<'_, u16>, _ao: usize, _b: &GpuBuffer<'_, u16>, _bo: usize, _sa: &[f32], _sdt: &[f32], _sn: &[f32], _z: &GpuBuffer<'_, u16>, _zo: usize, _ss: &mut GpuBuffer<'_, u16>, _sso: usize, _out: &mut GpuBuffer<'_, u16>, _oo: usize, _nkh: usize, _nvh: usize, _hd: usize, _eps: f32) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_dense_f16_dev(&self, _x: &GpuBuffer<'_, u16>, _w: &[u16], _out: &mut GpuBuffer<'_, u16>, _m: usize, _n: usize, _k: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn kv_append_gqa_layer_f16(&self, _q: &GpuBuffer<'_, u16>, _k: &GpuBuffer<'_, u16>, _v: &GpuBuffer<'_, u16>, _kc: &mut GpuBuffer<'_, u16>, _vc: &mut GpuBuffer<'_, u16>, _off: usize, _out: &mut GpuBuffer<'_, u16>, _pos: usize, _m: usize, _nq: usize, _nkv: usize, _hd: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn swiglu_inplace_f16(&self, _gate: &mut GpuBuffer<'_, u16>, _up: &GpuBuffer<'_, u16>, _len: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn swiglu_fwht128_inplace_f16(&self, _gate: &mut GpuBuffer<'_, u16>, _up: &GpuBuffer<'_, u16>, _len: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn lm_head_f16_act_f16(&self, _hidden: &GpuBuffer<'_, u16>, _emb: &[u16], _logits: &mut GpuBuffer<'_, f32>, _v: usize, _h: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn lm_head_f32_act_f16(&self, _hidden: &GpuBuffer<'_, u16>, _emb: &[f32], _logits: &mut GpuBuffer<'_, f32>, _v: usize, _h: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn lm_head_quant_act_f16(&self, _hidden: &GpuBuffer<'_, u16>, _pw: &[u8], _sc: &[u16], _logits: &mut GpuBuffer<'_, f32>, _v: usize, _h: usize, _bits: u8) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_ternary_swiglu_f16_dev(&self, _x: &GpuBuffer<'_, u16>, _gw: &[u8], _gs: &[u16], _uw: &[u8], _us: &[u16], _out: &mut GpuBuffer<'_, u16>, _n: usize, _k: usize, _bits: u8) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn sample_candidates_f32(&self, _logits: &GpuBuffer<'_, f32>, _len: usize, _margin: f32, _max_out: usize) -> Result<Option<Vec<(usize, f32)>>, BackendError> {
        Err(linux_only_err())
    }
    pub fn graph_begin_capture(&self) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn graph_end_capture(&self) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn graph_launch(&self, _hidden: &mut GpuBuffer<'_, u16>, _emb: &[f32], _pos: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q2_affine_f16_dev(&self, _x: &GpuBuffer<'_, u16>, _pw: &[u8], _sc: &[u16], _zr: &[u16], _out: &mut GpuBuffer<'_, u16>, _m: usize, _n: usize, _k: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn alloc_u8(&self, _len: usize) -> Result<GpuBuffer<'_, u8>, BackendError> {
        Err(linux_only_err())
    }
    pub fn alloc_u16(&self, _len: usize) -> Result<GpuBuffer<'_, u16>, BackendError> {
        Err(linux_only_err())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q4_affine_f16_dev(&self, _x: &GpuBuffer<'_, u16>, _pw: &[u8], _sc: &[u16], _zr: &[u16], _out: &mut GpuBuffer<'_, u16>, _m: usize, _n: usize, _k: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    /// # Safety
    /// Linux-only CUDA streaming q4_affine stub.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_q4_affine_f16_streaming_dev(&self, _x: &GpuBuffer<'_, u16>, _ph: *const u8, _pl: usize, _sl: usize, _sw: &mut GpuBuffer<'_, u8>, _ss: &mut GpuBuffer<'_, u16>, _sz: &mut GpuBuffer<'_, u16>, _out: &mut GpuBuffer<'_, u16>, _m: usize, _n: usize, _k: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn alloc_pinned_bytes(&self, _bytes: usize) -> Result<*mut u8, BackendError> {
        Err(linux_only_err())
    }
    /// # Safety
    /// Linux-only CUDA pinned memory free stub.
    pub unsafe fn free_pinned_bytes(&self, _ptr: *mut u8) {}
    /// # Safety
    /// Linux-only CUDA async H2D copy stub.
    pub unsafe fn memcpy_h2d_async(&self, _dev_dst: &mut GpuBuffer<'_, u8>, _host_src: *const u8, _bytes: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn upload_u8_slice(&self, _dev_dst: &mut GpuBuffer<'_, u8>, _src: &[u8]) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn upload_u16_slice(&self, _dev_dst: &mut GpuBuffer<'_, u16>, _src: &[u16]) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q4_affine_buffers_dev(
        &self,
        _x: &GpuBuffer<'_, u16>,
        _w: &GpuBuffer<'_, u8>,
        _s: &GpuBuffer<'_, u16>,
        _z: &GpuBuffer<'_, u16>,
        _out: &mut GpuBuffer<'_, u16>,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q2_affine_buffers_dev(
        &self,
        _x: &GpuBuffer<'_, u16>,
        _w: &GpuBuffer<'_, u8>,
        _s: &GpuBuffer<'_, u16>,
        _z: &GpuBuffer<'_, u16>,
        _out: &mut GpuBuffer<'_, u16>,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn geglu_inplace_f16(
        &self,
        _gate: &mut GpuBuffer<'_, u16>,
        _up: &GpuBuffer<'_, u16>,
        _len: usize,
    ) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn weighted_acc_f16_to_f32(
        &self,
        _acc: &mut GpuBuffer<'_, f32>,
        _x: &GpuBuffer<'_, u16>,
        _weight: f32,
        _len: usize,
        _reset: bool,
    ) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn vram_free_bytes(&self) -> usize {
        0
    }
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q2_affine_geglu_buffers_dev(
        &self,
        _x: &GpuBuffer<'_, u16>,
        _wg: &GpuBuffer<'_, u8>,
        _sg: &GpuBuffer<'_, u16>,
        _zg: &GpuBuffer<'_, u16>,
        _wu: &GpuBuffer<'_, u8>,
        _su: &GpuBuffer<'_, u16>,
        _zu: &GpuBuffer<'_, u16>,
        _out: &mut GpuBuffer<'_, u16>,
        _n: usize,
        _k: usize,
    ) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gemma4_norm_rope_f16(
        &self,
        _vec: &mut GpuBuffer<'_, u16>,
        _norm_w: Option<&[f32]>,
        _rope_freqs: Option<&[f32]>,
        _num_heads: usize,
        _head_dim: usize,
        _rope_dim: usize,
        _pos: usize,
        _theta: f32,
        _eps: f32,
    ) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gemma4_attention_f16(
        &self,
        _q: &GpuBuffer<'_, u16>,
        _k_new: &GpuBuffer<'_, u16>,
        _v_new: &GpuBuffer<'_, u16>,
        _k_cache: &mut GpuBuffer<'_, u16>,
        _v_cache: &mut GpuBuffer<'_, u16>,
        _out: &mut GpuBuffer<'_, u16>,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _pos: usize,
        _window_size: usize,
    ) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn copy_d2d_u16(
        &self,
        _dst: &mut GpuBuffer<'_, u16>,
        _src: &GpuBuffer<'_, u16>,
        _len: usize,
    ) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn gemma4_moe_q2_batch(
        &self,
        _x: &GpuBuffer<'_, u16>,
        _batch: &Gemma4MoeBatchArgs,
        _gate_scratch: &mut GpuBuffer<'_, u16>,
        _acc_out: &mut GpuBuffer<'_, f32>,
        _inter_dim: usize,
        _hidden_dim: usize,
    ) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
}

#[cfg(target_os = "linux")]
impl Drop for GpuBackend {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.weight_cache.lock() {
            for (_, pair) in guard.drain() {
                unsafe {
                    (self.symbols.free)(pair.dev_w);
                    (self.symbols.free)(pair.dev_scales);
                }
            }
        }
        if let Ok(mut guard) = self.raw_cache.lock() {
            for (_, raw) in guard.drain() {
                unsafe { (self.symbols.free)(raw.0) };
            }
        }
        unsafe { dlclose(self.symbols.handle) };
    }
}

impl TernaryHadamardBackend for GpuBackend {
    fn as_gpu(&self) -> Option<&GpuBackend> {
        Some(self)
    }

    fn matmul_hadamard_ternary(
        &self,
        activations: &[f32],
        packed_weights: &[u8],
        scales_fp16: &[u16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, BackendError> {
        self.matmul_ternary(activations, packed_weights, scales_fp16, m, n, k, true)
    }

    fn matmul_ternary(
        &self,
        activations: &[f32],
        packed_weights: &[u8],
        scales_fp16: &[u16],
        m: usize,
        n: usize,
        k: usize,
        use_hadamard: bool,
    ) -> Result<Vec<f32>, BackendError> {
        validate_matmul_inputs(activations, packed_weights, scales_fp16, m, n, k)?;

        #[cfg(target_os = "linux")]
        {
            let (dev_w, dev_scales) = self.get_or_upload_weights(packed_weights, scales_fp16)?;
            let mut out = vec![0.0f32; m * n];
            let status = unsafe {
                (self.symbols.matmul_ternary)(
                    activations.as_ptr(),
                    dev_w,
                    dev_scales,
                    out.as_mut_ptr(),
                    m as c_int,
                    n as c_int,
                    k as c_int,
                    if use_hadamard { 1 } else { 0 },
                )
            };
            check_cuda_status("aibox_gpu_matmul_ternary", status)?;
            Ok(out)
        }

        #[cfg(not(target_os = "linux"))]
        {
            let _ = use_hadamard;
            Err(linux_only_err())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_backend_reports_unavailable_cleanly_without_library() {
        #[cfg(target_os = "linux")]
        unsafe {
            std::env::set_var("AIBOX_GPU_LIB", "/nonexistent/libaibox_gpu_missing.so");
        }
        let err = GpuBackend::try_new().err().expect("expected error without CUDA lib");
        assert!(matches!(err, BackendError::UnsupportedFeature(_)));
    }

    #[test]
    fn gpu_q2_affine_matches_scalar_when_available() {
        use crate::scalar::matmul_q2_affine;
        use crate::tensor::quantize_q2_affine_f32;

        let Ok(gpu) = GpuBackend::try_new() else {
            return;
        };
        let m = 1usize;
        let n = 4usize;
        let k = 192usize;
        let x: Vec<f32> = (0..k).map(|i| ((i % 15) as f32 - 7.0) * 0.1).collect();
        let w: Vec<f32> = (0..(n * k)).map(|i| ((i % 19) as f32 - 9.0) * 0.1).collect();
        let q2 = quantize_q2_affine_f32(&w, n, k).unwrap();
        let expected = matmul_q2_affine(
            &x,
            &q2.packed_weights,
            &q2.scales_fp16,
            &q2.zeros_fp16,
            m,
            n,
            k,
        )
        .unwrap();

        let mut x_dev = gpu.alloc_f16(k).unwrap();
        x_dev.upload_f32_as_f16(&x).unwrap();
        let mut out_dev = gpu.alloc_f16(m * n).unwrap();
        gpu.matmul_q2_affine_f16_dev(
            &x_dev,
            &q2.packed_weights,
            &q2.scales_fp16,
            &q2.zeros_fp16,
            &mut out_dev,
            m,
            n,
            k,
        )
        .unwrap();
        let mut got = vec![0.0f32; m * n];
        out_dev.download_as_f32(&mut got).unwrap();
        for (i, (&exp, &g)) in expected.iter().zip(got.iter()).enumerate() {
            assert!((exp - g).abs() < 0.05, "idx={i} expected={exp} got={g}");
        }
    }

    #[test]
    fn gpu_q4_affine_matches_scalar_when_available() {
        use crate::scalar::matmul_q4_affine;
        use crate::tensor::quantize_q4_affine_f32;

        let Ok(gpu) = GpuBackend::try_new() else {
            return;
        };
        let m = 1usize;
        let n = 4usize;
        let k = 192usize;
        let x: Vec<f32> = (0..k).map(|i| ((i % 15) as f32 - 7.0) * 0.1).collect();
        let w: Vec<f32> = (0..(n * k)).map(|i| ((i % 19) as f32 - 9.0) * 0.1).collect();
        let q4 = quantize_q4_affine_f32(&w, n, k).unwrap();
        let expected = matmul_q4_affine(
            &x,
            &q4.packed_weights,
            &q4.scales_fp16,
            &q4.zeros_fp16,
            m,
            n,
            k,
        )
        .unwrap();

        let mut x_dev = gpu.alloc_f16(k).unwrap();
        x_dev.upload_f32_as_f16(&x).unwrap();
        let mut out_dev = gpu.alloc_f16(m * n).unwrap();
        gpu.matmul_q4_affine_f16_dev(
            &x_dev,
            &q4.packed_weights,
            &q4.scales_fp16,
            &q4.zeros_fp16,
            &mut out_dev,
            m,
            n,
            k,
        )
        .unwrap();
        let mut got = vec![0.0f32; m * n];
        out_dev.download_as_f32(&mut got).unwrap();
        for (i, (&exp, &g)) in expected.iter().zip(got.iter()).enumerate() {
            assert!((exp - g).abs() < 0.05, "q4 idx={i} expected={exp} got={g}");
        }
    }
}

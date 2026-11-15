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
    unsafe extern "C" fn(*mut u16, *const f32, c_int, c_int, c_int, c_int, f32, f32) -> c_int;
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
    graph_begin_capture: FnGraphCapture,
    graph_end_capture: FnGraphCapture,
    graph_launch: FnGraphLaunch,
    matmul_ternary: FnMatmulTernary,
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
impl<'a> GpuBuffer<'a, f32> {
    pub fn download(&self, out: &mut [f32]) -> Result<(), BackendError> {
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
}

#[cfg(not(target_os = "linux"))]
impl<'a> GpuBuffer<'a, f32> {
    pub fn download(&self, _out: &mut [f32]) -> Result<(), BackendError> {
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
                graph_begin_capture: load_symbol(handle, "aibox_gpu_graph_begin_capture")?,
                graph_end_capture: load_symbol(handle, "aibox_gpu_graph_end_capture")?,
                graph_launch: load_symbol(handle, "aibox_gpu_graph_launch")?,
                matmul_ternary: load_symbol(handle, "aibox_gpu_matmul_ternary")?,
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
                start_pos as c_int,
                theta,
                eps,
            )
        };
        check_cuda_status("aibox_gpu_rms_norm_heads_rope_f16", status)
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
    pub fn rms_norm_heads_rope_f16(&self, _vec: &mut GpuBuffer<'_, u16>, _nw: Option<&[f32]>, _m: usize, _nh: usize, _hd: usize, _pos: usize, _theta: f32, _eps: f32) -> Result<(), BackendError> {
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
    pub fn graph_begin_capture(&self) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn graph_end_capture(&self) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn graph_launch(&self, _hidden: &mut GpuBuffer<'_, u16>, _emb: &[f32], _pos: usize) -> Result<(), BackendError> {
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
}

//! Backend GPU CUDA chargé dynamiquement (`libaibox_gpu.so`) avec cache VRAM.

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
type FnFwht128 = unsafe extern "C" fn(*mut f32, c_int, c_int) -> c_int;
#[cfg(target_os = "linux")]
type FnMatmulTernaryDev =
    unsafe extern "C" fn(*const f32, *const u8, *const u16, *mut f32, c_int, c_int, c_int) -> c_int;
#[cfg(target_os = "linux")]
type FnRmsNorm = unsafe extern "C" fn(*const f32, *const f32, *mut f32, c_int, c_int, f32) -> c_int;
#[cfg(target_os = "linux")]
type FnRmsNormHeadsRope =
    unsafe extern "C" fn(*mut f32, *const f32, c_int, c_int, c_int, f32, f32) -> c_int;
#[cfg(target_os = "linux")]
type FnKvAppendGqa = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *const f32,
    *mut f32,
    *mut f32,
    *mut f32,
    c_int,
    c_int,
    c_int,
    c_int,
) -> c_int;
#[cfg(target_os = "linux")]
type FnSwigluInplace = unsafe extern "C" fn(*mut f32, *const f32, c_int) -> c_int;
#[cfg(target_os = "linux")]
type FnAddInplace = unsafe extern "C" fn(*mut f32, *const f32, c_int) -> c_int;
#[cfg(target_os = "linux")]
type FnLmHeadF16 = unsafe extern "C" fn(*const f32, *const u16, *mut f32, c_int, c_int) -> c_int;
#[cfg(target_os = "linux")]
type FnLmHeadF32 = unsafe extern "C" fn(*const f32, *const f32, *mut f32, c_int, c_int) -> c_int;
#[cfg(target_os = "linux")]
type FnArgmaxF32 = unsafe extern "C" fn(*const f32, c_int, *mut u32) -> c_int;
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
    fwht128: FnFwht128,
    matmul_ternary_dev: FnMatmulTernaryDev,
    matmul_ternary_add_dev: FnMatmulTernaryDev,
    rms_norm: FnRmsNorm,
    rms_norm_fwht128: FnRmsNorm,
    rms_norm_heads_rope: FnRmsNormHeadsRope,
    kv_append_gqa: FnKvAppendGqa,
    swiglu_inplace: FnSwigluInplace,
    swiglu_fwht128_inplace: FnSwigluInplace,
    add_inplace: FnAddInplace,
    lm_head_f16: FnLmHeadF16,
    lm_head_f32: FnLmHeadF32,
    argmax_f32: FnArgmaxF32,
    matmul_ternary: FnMatmulTernary,
}

#[cfg(target_os = "linux")]
unsafe impl Send for CudaSymbols {}
#[cfg(target_os = "linux")]
unsafe impl Sync for CudaSymbols {}

#[cfg(target_os = "linux")]
type WeightCacheKey = (usize, usize, usize, usize);

/// Tampon f32 alloué en VRAM avec libération automatique (RAII).
pub struct GpuBuffer<'a> {
    #[cfg(target_os = "linux")]
    gpu: &'a GpuBackend,
    #[cfg(target_os = "linux")]
    ptr: *mut f32,
    #[cfg(not(target_os = "linux"))]
    _phantom: std::marker::PhantomData<&'a ()>,
    len: usize,
}

#[cfg(target_os = "linux")]
impl Drop for GpuBuffer<'_> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { (self.gpu.symbols.free)(self.ptr as *mut c_void) };
        }
    }
}

impl<'a> GpuBuffer<'a> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(target_os = "linux")]
impl<'a> GpuBuffer<'a> {
    pub fn as_ptr(&self) -> *const f32 {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr
    }

    pub fn upload(&mut self, data: &[f32]) -> Result<(), BackendError> {
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

#[cfg(not(target_os = "linux"))]
impl<'a> GpuBuffer<'a> {
    pub fn upload(&mut self, _data: &[f32]) -> Result<(), BackendError> {
        Err(linux_only_err())
    }

    pub fn download(&self, _out: &mut [f32]) -> Result<(), BackendError> {
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
                fwht128: load_symbol(handle, "aibox_gpu_fwht128")?,
                matmul_ternary_dev: load_symbol(handle, "aibox_gpu_matmul_ternary_dev")?,
                matmul_ternary_add_dev: load_symbol(handle, "aibox_gpu_matmul_ternary_add_dev")?,
                rms_norm: load_symbol(handle, "aibox_gpu_rms_norm")?,
                rms_norm_fwht128: load_symbol(handle, "aibox_gpu_rms_norm_fwht128")?,
                rms_norm_heads_rope: load_symbol(handle, "aibox_gpu_rms_norm_heads_rope")?,
                kv_append_gqa: load_symbol(handle, "aibox_gpu_kv_append_gqa")?,
                swiglu_inplace: load_symbol(handle, "aibox_gpu_swiglu_inplace")?,
                swiglu_fwht128_inplace: load_symbol(handle, "aibox_gpu_swiglu_fwht128_inplace")?,
                add_inplace: load_symbol(handle, "aibox_gpu_add_inplace")?,
                lm_head_f16: load_symbol(handle, "aibox_gpu_lm_head_f16")?,
                lm_head_f32: load_symbol(handle, "aibox_gpu_lm_head_f32")?,
                argmax_f32: load_symbol(handle, "aibox_gpu_argmax_f32")?,
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

    pub fn alloc_f32(&self, len: usize) -> Result<GpuBuffer<'_>, BackendError> {
        let bytes = len.max(1) * std::mem::size_of::<f32>();
        let ptr = unsafe { (self.symbols.alloc)(bytes) };
        if ptr.is_null() {
            return Err(BackendError::UnsupportedFeature(
                "cudaMalloc failed for GpuBuffer".to_string(),
            ));
        }
        Ok(GpuBuffer {
            gpu: self,
            ptr: ptr as *mut f32,
            len,
        })
    }

    pub fn fwht128_inplace(&self, buf: &mut GpuBuffer<'_>, m: usize, k: usize) -> Result<(), BackendError> {
        let status = unsafe { (self.symbols.fwht128)(buf.as_mut_ptr(), m as c_int, k as c_int) };
        check_cuda_status("aibox_gpu_fwht128", status)
    }

    fn run_matmul_ternary_dev(
        &self,
        op_name: &str,
        func: FnMatmulTernaryDev,
        x: &GpuBuffer<'_>,
        packed_weights: &[u8],
        scales_fp16: &[u16],
        out: &mut GpuBuffer<'_>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BackendError> {
        let (dev_w, dev_scales) = self.get_or_upload_weights(packed_weights, scales_fp16)?;
        let status = unsafe {
            func(
                x.as_ptr(),
                dev_w,
                dev_scales,
                out.as_mut_ptr(),
                m as c_int,
                n as c_int,
                k as c_int,
            )
        };
        check_cuda_status(op_name, status)
    }

    pub fn matmul_ternary_dev(
        &self,
        x: &GpuBuffer<'_>,
        packed_weights: &[u8],
        scales_fp16: &[u16],
        out: &mut GpuBuffer<'_>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BackendError> {
        self.run_matmul_ternary_dev(
            "aibox_gpu_matmul_ternary_dev",
            self.symbols.matmul_ternary_dev,
            x,
            packed_weights,
            scales_fp16,
            out,
            m,
            n,
            k,
        )
    }

    pub fn matmul_ternary_add_dev(
        &self,
        x: &GpuBuffer<'_>,
        packed_weights: &[u8],
        scales_fp16: &[u16],
        out: &mut GpuBuffer<'_>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BackendError> {
        self.run_matmul_ternary_dev(
            "aibox_gpu_matmul_ternary_add_dev",
            self.symbols.matmul_ternary_add_dev,
            x,
            packed_weights,
            scales_fp16,
            out,
            m,
            n,
            k,
        )
    }

    pub fn rms_norm(
        &self,
        x: &GpuBuffer<'_>,
        weight: &[f32],
        out: &mut GpuBuffer<'_>,
        m: usize,
        dim: usize,
        eps: f32,
    ) -> Result<(), BackendError> {
        let dev_w = self.get_or_upload_raw_f32(weight)?;
        let status = unsafe {
            (self.symbols.rms_norm)(
                x.as_ptr(),
                dev_w,
                out.as_mut_ptr(),
                m as c_int,
                dim as c_int,
                eps,
            )
        };
        check_cuda_status("aibox_gpu_rms_norm", status)
    }

    pub fn rms_norm_fwht128(
        &self,
        x: &GpuBuffer<'_>,
        weight: &[f32],
        out: &mut GpuBuffer<'_>,
        m: usize,
        dim: usize,
        eps: f32,
    ) -> Result<(), BackendError> {
        let dev_w = self.get_or_upload_raw_f32(weight)?;
        let status = unsafe {
            (self.symbols.rms_norm_fwht128)(
                x.as_ptr(),
                dev_w,
                out.as_mut_ptr(),
                m as c_int,
                dim as c_int,
                eps,
            )
        };
        check_cuda_status("aibox_gpu_rms_norm_fwht128", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_heads_rope(
        &self,
        vec: &mut GpuBuffer<'_>,
        norm_w: Option<&[f32]>,
        num_heads: usize,
        head_dim: usize,
        pos: usize,
        theta: f32,
        eps: f32,
    ) -> Result<(), BackendError> {
        let dev_norm = match norm_w {
            Some(w) => self.get_or_upload_raw_f32(w)?,
            None => std::ptr::null(),
        };
        let status = unsafe {
            (self.symbols.rms_norm_heads_rope)(
                vec.as_mut_ptr(),
                dev_norm,
                num_heads as c_int,
                head_dim as c_int,
                pos as c_int,
                theta,
                eps,
            )
        };
        check_cuda_status("aibox_gpu_rms_norm_heads_rope", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kv_append_gqa_layer(
        &self,
        q: &GpuBuffer<'_>,
        k: &GpuBuffer<'_>,
        v: &GpuBuffer<'_>,
        k_cache: &mut GpuBuffer<'_>,
        v_cache: &mut GpuBuffer<'_>,
        layer_offset: usize,
        out: &mut GpuBuffer<'_>,
        pos: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(), BackendError> {
        let status = unsafe {
            (self.symbols.kv_append_gqa)(
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                k_cache.as_mut_ptr().add(layer_offset),
                v_cache.as_mut_ptr().add(layer_offset),
                out.as_mut_ptr(),
                pos as c_int,
                num_q_heads as c_int,
                num_kv_heads as c_int,
                head_dim as c_int,
            )
        };
        check_cuda_status("aibox_gpu_kv_append_gqa", status)
    }

    pub fn swiglu_inplace(
        &self,
        gate: &mut GpuBuffer<'_>,
        up: &GpuBuffer<'_>,
        len: usize,
    ) -> Result<(), BackendError> {
        let status = unsafe { (self.symbols.swiglu_inplace)(gate.as_mut_ptr(), up.as_ptr(), len as c_int) };
        check_cuda_status("aibox_gpu_swiglu_inplace", status)
    }

    pub fn swiglu_fwht128_inplace(
        &self,
        gate: &mut GpuBuffer<'_>,
        up: &GpuBuffer<'_>,
        len: usize,
    ) -> Result<(), BackendError> {
        let status = unsafe { (self.symbols.swiglu_fwht128_inplace)(gate.as_mut_ptr(), up.as_ptr(), len as c_int) };
        check_cuda_status("aibox_gpu_swiglu_fwht128_inplace", status)
    }

    pub fn add_inplace(
        &self,
        target: &mut GpuBuffer<'_>,
        delta: &GpuBuffer<'_>,
        len: usize,
    ) -> Result<(), BackendError> {
        let status = unsafe { (self.symbols.add_inplace)(target.as_mut_ptr(), delta.as_ptr(), len as c_int) };
        check_cuda_status("aibox_gpu_add_inplace", status)
    }

    pub fn lm_head_f16(
        &self,
        hidden: &GpuBuffer<'_>,
        emb_f16: &[u16],
        logits_dev: &mut GpuBuffer<'_>,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<(), BackendError> {
        let dev_emb = self.get_or_upload_raw_u16(emb_f16)?;
        let status = unsafe {
            (self.symbols.lm_head_f16)(
                hidden.as_ptr(),
                dev_emb,
                logits_dev.as_mut_ptr(),
                vocab_size as c_int,
                hidden_size as c_int,
            )
        };
        check_cuda_status("aibox_gpu_lm_head_f16", status)
    }

    pub fn lm_head_f32(
        &self,
        hidden: &GpuBuffer<'_>,
        emb_f32: &[f32],
        logits_dev: &mut GpuBuffer<'_>,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<(), BackendError> {
        let dev_emb = self.get_or_upload_raw_f32(emb_f32)?;
        let status = unsafe {
            (self.symbols.lm_head_f32)(
                hidden.as_ptr(),
                dev_emb,
                logits_dev.as_mut_ptr(),
                vocab_size as c_int,
                hidden_size as c_int,
            )
        };
        check_cuda_status("aibox_gpu_lm_head_f32", status)
    }

    pub fn argmax_f32(&self, logits_dev: &GpuBuffer<'_>, len: usize) -> Result<usize, BackendError> {
        let mut best_idx: u32 = 0;
        let status = unsafe { (self.symbols.argmax_f32)(logits_dev.as_ptr(), len as c_int, &mut best_idx) };
        check_cuda_status("aibox_gpu_argmax_f32", status)?;
        Ok(best_idx as usize)
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
    pub fn alloc_f32(&self, _len: usize) -> Result<GpuBuffer<'_>, BackendError> {
        Err(linux_only_err())
    }
    pub fn fwht128_inplace(&self, _buf: &mut GpuBuffer<'_>, _m: usize, _k: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn matmul_ternary_dev(
        &self,
        _x: &GpuBuffer<'_>,
        _packed_weights: &[u8],
        _scales_fp16: &[u16],
        _out: &mut GpuBuffer<'_>,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn matmul_ternary_add_dev(
        &self,
        _x: &GpuBuffer<'_>,
        _packed_weights: &[u8],
        _scales_fp16: &[u16],
        _out: &mut GpuBuffer<'_>,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn rms_norm(
        &self,
        _x: &GpuBuffer<'_>,
        _weight: &[f32],
        _out: &mut GpuBuffer<'_>,
        _m: usize,
        _dim: usize,
        _eps: f32,
    ) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn rms_norm_fwht128(
        &self,
        _x: &GpuBuffer<'_>,
        _weight: &[f32],
        _out: &mut GpuBuffer<'_>,
        _m: usize,
        _dim: usize,
        _eps: f32,
    ) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_heads_rope(
        &self,
        _vec: &mut GpuBuffer<'_>,
        _norm_w: Option<&[f32]>,
        _num_heads: usize,
        _head_dim: usize,
        _pos: usize,
        _theta: f32,
        _eps: f32,
    ) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn kv_append_gqa_layer(
        &self,
        _q: &GpuBuffer<'_>,
        _k: &GpuBuffer<'_>,
        _v: &GpuBuffer<'_>,
        _k_cache: &mut GpuBuffer<'_>,
        _v_cache: &mut GpuBuffer<'_>,
        _layer_offset: usize,
        _out: &mut GpuBuffer<'_>,
        _pos: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
    ) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn swiglu_inplace(&self, _gate: &mut GpuBuffer<'_>, _up: &GpuBuffer<'_>, _len: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn swiglu_fwht128_inplace(&self, _gate: &mut GpuBuffer<'_>, _up: &GpuBuffer<'_>, _len: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn add_inplace(&self, _target: &mut GpuBuffer<'_>, _delta: &GpuBuffer<'_>, _len: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn lm_head_f16(&self, _hidden: &GpuBuffer<'_>, _emb_f16: &[u16], _logits_dev: &mut GpuBuffer<'_>, _v: usize, _h: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn lm_head_f32(&self, _hidden: &GpuBuffer<'_>, _emb_f32: &[f32], _logits_dev: &mut GpuBuffer<'_>, _v: usize, _h: usize) -> Result<(), BackendError> {
        Err(linux_only_err())
    }
    pub fn argmax_f32(&self, _logits_dev: &GpuBuffer<'_>, _len: usize) -> Result<usize, BackendError> {
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

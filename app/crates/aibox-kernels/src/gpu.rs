//! Backend GPU CUDA chargé dynamiquement (`libaibox_gpu.so`) avec cache VRAM des poids.
//!
//! Zéro dépendance externe : compile sans SDK CUDA sur Windows/macOS/Linux
//! et charge la bibliothèque partagée CUDA au runtime sous Linux.

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
type FnAllocCopy = unsafe extern "C" fn(*const c_void, usize) -> *mut c_void;
#[cfg(target_os = "linux")]
type FnFree = unsafe extern "C" fn(*mut c_void);
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
struct CudaSymbols {
    handle: *mut c_void,
    alloc_copy: FnAllocCopy,
    free: FnFree,
    matmul_ternary: FnMatmulTernary,
}

#[cfg(target_os = "linux")]
unsafe impl Send for CudaSymbols {}
#[cfg(target_os = "linux")]
unsafe impl Sync for CudaSymbols {}

#[cfg(target_os = "linux")]
type WeightCacheKey = (usize, usize, usize, usize);

/// Backend d'inférence ternaire sur GPU CUDA (via `libaibox_gpu.so`).
pub struct GpuBackend {
    #[cfg(target_os = "linux")]
    symbols: CudaSymbols,
    #[cfg(target_os = "linux")]
    cache: Mutex<HashMap<WeightCacheKey, GpuWeightPair>>,
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

impl GpuBackend {
    /// Tente de charger `libaibox_gpu.so` et de vérifier qu'un GPU CUDA est disponible.
    pub fn try_new() -> Result<Self, BackendError> {
        #[cfg(target_os = "linux")]
        {
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
                    alloc_copy: load_symbol(handle, "aibox_gpu_alloc_copy")?,
                    free: load_symbol(handle, "aibox_gpu_free")?,
                    matmul_ternary: load_symbol(handle, "aibox_gpu_matmul_ternary")?,
                })
            })();

            match load_result {
                Ok(symbols) => Ok(Self {
                    symbols,
                    cache: Mutex::new(HashMap::new()),
                }),
                Err(err) => {
                    unsafe { dlclose(handle) };
                    Err(err)
                }
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err(BackendError::UnsupportedFeature(
                "CUDA dynamic library backend is only supported on Linux".to_string(),
            ))
        }
    }

    #[cfg(target_os = "linux")]
    fn alloc_weight_pair(
        &self,
        packed_weights: &[u8],
        scales_fp16: &[u16],
    ) -> Result<GpuWeightPair, BackendError> {
        let dev_w = unsafe {
            (self.symbols.alloc_copy)(
                packed_weights.as_ptr() as *const c_void,
                std::mem::size_of_val(packed_weights),
            )
        };
        if dev_w.is_null() {
            return Err(BackendError::UnsupportedFeature(
                "cudaMalloc/cudaMemcpy failed for packed_weights".to_string(),
            ));
        }

        let dev_scales = unsafe {
            (self.symbols.alloc_copy)(
                scales_fp16.as_ptr() as *const c_void,
                std::mem::size_of_val(scales_fp16),
            )
        };
        if dev_scales.is_null() {
            unsafe { (self.symbols.free)(dev_w) };
            return Err(BackendError::UnsupportedFeature(
                "cudaMalloc/cudaMemcpy failed for scales_fp16".to_string(),
            ));
        }

        Ok(GpuWeightPair { dev_w, dev_scales })
    }

    #[cfg(target_os = "linux")]
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
        let mut guard = self.cache.lock().map_err(|_| {
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

#[cfg(target_os = "linux")]
impl Drop for GpuBackend {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.cache.lock() {
            for (_, pair) in guard.drain() {
                unsafe {
                    (self.symbols.free)(pair.dev_w);
                    (self.symbols.free)(pair.dev_scales);
                }
            }
        }
        unsafe {
            dlclose(self.symbols.handle);
        }
    }
}

impl TernaryHadamardBackend for GpuBackend {
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
            if status != 0 {
                return Err(BackendError::UnsupportedFeature(format!(
                    "aibox_gpu_matmul_ternary failed with status {status}"
                )));
            }
            Ok(out)
        }

        #[cfg(not(target_os = "linux"))]
        {
            let _ = use_hadamard;
            Err(BackendError::UnsupportedFeature(
                "CUDA dynamic library backend is only supported on Linux".to_string(),
            ))
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

// ═══════════════════════════════════════════════════════════════
//   arena.rs — Bump Allocator for Zero-Fragmentation Inference
// ═══════════════════════════════════════════════════════════════
//
// Pre-allocates a fixed buffer (default 80MB). Inference scratches
// are bumped forward; reset() returns cursor to zero.
//
// Benefits: O(1) alloc, zero fragmentation, no malloc/free overhead.
// Used for: all intermediate tensors during forward pass.
//
// Agent 1 — Week 1

/// Fixed-size bump allocator
pub struct Arena {
    buffer: Vec<u8>,
    cursor: usize,
    capacity: usize,
    peak_usage: usize,
}

impl Arena {
    /// Create arena with given capacity in bytes
    pub fn new(capacity: usize) -> Self {
        Arena {
            buffer: vec![0u8; capacity],
            cursor: 0,
            capacity,
            peak_usage: 0,
        }
    }

    /// Create arena with default 80MB capacity
    pub fn default_80mb() -> Self {
        Self::new(80 * 1024 * 1024)
    }

    /// Allocate `size` bytes, aligned to `align` boundary.
    /// Returns a mutable slice of the allocated region.
    /// Panics if out of memory.
    pub fn alloc(&mut self, size: usize, align: usize) -> &mut [u8] {
        // Align cursor
        let aligned = (self.cursor + align - 1) & !(align - 1);
        let end = aligned + size;

        if end > self.capacity {
            panic!(
                "Arena OOM: requested {}B at cursor {}, capacity {}B (peak: {}B)",
                size, self.cursor, self.capacity, self.peak_usage
            );
        }

        self.cursor = end;
        if self.cursor > self.peak_usage {
            self.peak_usage = self.cursor;
        }

        &mut self.buffer[aligned..end]
    }

    /// Allocate space for `count` f32 values (aligned to 32 for AVX2)
    pub fn alloc_f32(&mut self, count: usize) -> &mut [f32] {
        let bytes = self.alloc(count * 4, 32);
        unsafe {
            std::slice::from_raw_parts_mut(
                bytes.as_mut_ptr() as *mut f32,
                count,
            )
        }
    }

    /// Allocate space for `count` i8 values (aligned to 32 for AVX2)
    pub fn alloc_i8(&mut self, count: usize) -> &mut [i8] {
        let bytes = self.alloc(count, 32);
        unsafe {
            std::slice::from_raw_parts_mut(
                bytes.as_mut_ptr() as *mut i8,
                count,
            )
        }
    }

    /// Reset cursor to zero (free all allocations at once)
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Current usage in bytes
    pub fn used(&self) -> usize {
        self.cursor
    }

    /// Peak usage in bytes (high watermark)
    pub fn peak(&self) -> usize {
        self.peak_usage
    }

    /// Remaining capacity
    pub fn remaining(&self) -> usize {
        self.capacity - self.cursor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_alloc() {
        let mut arena = Arena::new(1024);
        let a = arena.alloc_f32(10);
        assert_eq!(a.len(), 10);
        a[0] = 42.0;
        assert_eq!(a[0], 42.0);
        assert!(arena.used() >= 40);
    }

    #[test]
    fn test_reset() {
        let mut arena = Arena::new(1024);
        arena.alloc_f32(100);
        assert!(arena.used() > 0);
        arena.reset();
        assert_eq!(arena.used(), 0);
    }

    #[test]
    #[should_panic(expected = "Arena OOM")]
    fn test_oom() {
        let mut arena = Arena::new(64);
        arena.alloc_f32(100); // 400 bytes > 64 → panic
    }
}

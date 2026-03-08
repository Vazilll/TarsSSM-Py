/*
 * TarsArena — Zero-fragmentation bump allocator for TARS inference.
 * 
 * All intermediate buffers during inference are allocated from the arena.
 * At the end of each generation step, call reset() to reclaim all memory.
 * Result: ZERO memory fragmentation, ZERO malloc/free overhead.
 *
 * Agent 1 — Week 1
 */

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef _WIN32
    #ifdef TARS_BUILDING_DLL
        #define TARS_API __declspec(dllexport)
    #else
        #define TARS_API __declspec(dllimport)
    #endif
#else
    #define TARS_API __attribute__((visibility("default")))
#endif

/// Default arena capacity: 80 MB (enough for d_model=1280, batch=1)
static constexpr size_t TARS_ARENA_DEFAULT_CAPACITY = 80ULL * 1024 * 1024;

class TARS_API TarsArena {
public:
    /// Create arena with given capacity (bytes). Uses OS-level virtual memory.
    explicit TarsArena(size_t capacity = TARS_ARENA_DEFAULT_CAPACITY);
    
    /// Destructor — releases virtual memory.
    ~TarsArena();
    
    // Non-copyable
    TarsArena(const TarsArena&) = delete;
    TarsArena& operator=(const TarsArena&) = delete;
    
    /// Allocate `bytes` from the arena. Returns aligned pointer.
    /// Returns nullptr if arena is exhausted.
    void* alloc(size_t bytes, size_t alignment = 32);
    
    /// Allocate typed array: N elements of type T with proper alignment.
    template<typename T>
    T* alloc_array(size_t count) {
        return static_cast<T*>(alloc(count * sizeof(T), alignof(T) < 32 ? 32 : alignof(T)));
    }
    
    /// Reset the arena — reclaim all memory (ptr = base). O(1).
    void reset();
    
    /// Bytes currently used.
    size_t used() const { return offset_; }
    
    /// Total capacity.
    size_t capacity() const { return capacity_; }
    
    /// Bytes remaining.
    size_t remaining() const { return capacity_ - offset_; }

private:
    uint8_t*  base_;       // Start of allocated region
    size_t    offset_;     // Current bump pointer offset
    size_t    capacity_;   // Total capacity
    bool      owns_memory_;// True if we allocated via VirtualAlloc/mmap
};
